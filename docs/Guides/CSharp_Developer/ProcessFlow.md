---
title: Process Flow — Control Flow & Execution Pipeline
parent: "[[info]]"
related: "[[ImplementationPlan]]"
audience: Experienced C# developers
tags: [guide, architecture, control-flow, pipeline]
---

# Process Flow — Control Flow & Execution Pipeline

This document traces the actual execution paths through pvtt's Phase 1 codebase. Where [[ImplementationPlan]] covers the *what* and *why*, this covers the *how* — call chains, data flow, and runtime behavior.

C# equivalents are noted where they clarify the pattern.

---

## Entry Point

```
pyproject.toml entry point:  pvtt = pvtt.cli.app:main
```

`cli/app.py:main()` is the process entry point — equivalent to `Program.Main()`. It wraps the Typer app in a try/catch boundary:

```python
def main() -> None:
    try:
        app()                          # Typer dispatches to subcommand
    except PvttError as exc:
        print_error(str(exc))          # Rich-formatted to stderr
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
```

**C# equivalent**: This is the global exception filter pattern — like `app.UseExceptionHandler()` in ASP.NET middleware, but at the process level. All domain exceptions (`PvttError` subtypes) are caught here and converted to clean stderr output + exit code.

The `@app.callback()` on `main_callback` runs before every subcommand, setting up logging verbosity from `-v` flags. Think of it as middleware that runs before the controller action.

---

## Command Flow: `pvtt transcribe file test.wav`

### Full Call Chain

```
cli/app.py:main()
  → Typer dispatch
    → cli/app.py:main_callback(verbose)     # setup logging
      → util/logging.py:setup_logging()
    → cli/transcribe.py:transcribe_file()   # subcommand handler
      → config/loader.py:load_config()      # build merged config
      → core/transcriber.py:Transcriber()   # construct orchestrator
        → engine/registry.py:get_engine()   # factory → FasterWhisperEngine
        → core/model_manager.py:ModelManager()
      → Transcriber.transcribe_file()
        → audio/file_reader.py:validate_audio_file()
        → Transcriber._ensure_model_loaded()
          → ModelManager.get_model_path()   # resolve name → local path
          → util/hardware.py:resolve_device_and_compute()
          → FasterWhisperEngine.load_model()
            → faster_whisper.WhisperModel()  # lazy import, C++ init
        → FasterWhisperEngine.transcribe()
          → WhisperModel.transcribe()        # CTranslate2 inference
          → yield TranscriptionSegment(...)  # convert to pvtt types
      → Transcriber.format_output()
        → export/registry.py:get_exporter()  # factory → PlainTextExporter
        → PlainTextExporter.format()
      → output_console.print(formatted)      # stdout
```

### Step-by-Step Breakdown

**1. CLI flag parsing** (`cli/transcribe.py:transcribe_file`)

Typer parses `--model`, `--device`, `--format`, etc. into typed Python values. Each flag maps to a config override dict:

```python
cli_overrides: dict[str, dict[str, object]] = {}
if model is not None:
    cli_overrides.setdefault("model", {})["name"] = model
```

This is equivalent to building an `IConfigurationBuilder` overlay. The dict structure mirrors the TOML/Pydantic schema sections: `{"model": {"name": "tiny.en"}, "transcription": {"language": "en"}}`.

**2. Config resolution** (`config/loader.py:load_config`)

Four layers, last wins:

```
Layer 1: Pydantic defaults  (schema.py — ModelConfig.name = "large-v3-turbo")
Layer 2: Global TOML         (load_toml → %LOCALAPPDATA%\pvtt\config.toml)
Layer 3: Env vars            (load_env_overrides → PVTT_MODEL_NAME=tiny.en)
Layer 4: CLI overrides       (deep_merge with cli_overrides dict)
```

`deep_merge()` recursively merges dicts — same semantics as `IConfiguration` layering in ASP.NET. The merged dict is validated by `PvttConfig.model_validate()` (Pydantic v2), which acts as both deserialization and validation in one call.

**3. Transcriber construction** (`core/transcriber.py:__init__`)

```python
self._engine = engine or get_engine()         # resolve from registry
self._model_manager = model_manager or ModelManager()
```

Dependencies are constructor-injected with fallback to defaults — poor man's DI. Tests inject mocks; production uses defaults. `get_engine()` returns a new `FasterWhisperEngine` from the registry factory.

**4. Audio validation** (`audio/file_reader.py:validate_audio_file`)

Guards: file exists, is a file (not directory), has supported extension. Raises `AudioError` on failure. No audio decoding at this stage — just path validation.

**5. Lazy model loading** (`core/transcriber.py:_ensure_model_loaded`)

The model loads on first transcription, not at construction. This is the critical path:

```
_ensure_model_loaded()
  ├── engine.is_loaded? → return early
  ├── ModelManager.get_model_path("tiny.en")
  │     ├── Is it an absolute path to a valid dir? → return it
  │     └── Look in models_dir / "tiny.en" → found? return : raise ModelNotFoundError
  ├── resolve_device_and_compute("auto", "auto")
  │     ├── detect_gpu() → try ctranslate2, fallback nvidia-smi
  │     ├── device = "cuda" if GPU found else "cpu"
  │     └── compute_type = based on VRAM (≥8GB → float16, ≥4GB → int8_float16, else int8)
  └── engine.load_model(path, device, compute_type, local_files_only=True)
        └── WhisperModel(path, device, compute_type, local_files_only=True)  # C++ init
```

**6. Inference** (`engine/faster_whisper.py:transcribe`)

```python
segments_iter, _info = self._model.transcribe(
    str(audio), language=..., beam_size=..., temperature=...,
    initial_prompt=..., vad_filter=..., word_timestamps=...
)
for seg in segments_iter:
    yield TranscriptionSegment(start=seg.start, end=seg.end, text=seg.text, ...)
```

The engine wraps faster-whisper's iterator, converting each segment to pvtt's frozen dataclass. This is an adapter pattern — the engine translates between the third-party API and pvtt's internal types.

Temperature has a quirk: faster-whisper expects a list, so a scalar `0.0` is wrapped as `[0.0]`.

**7. Result assembly** (`core/transcriber.py:transcribe_file`)

```python
segments = list(self._engine.transcribe(validated_path, options))
full_text = " ".join(seg.text.strip() for seg in segments)
return TranscriptionResult(segments=segments, text=full_text)
```

The iterator is fully consumed into a list, segments are joined with spaces. `TranscriptionResult` is a frozen dataclass — immutable after creation.

**8. Output** (`cli/transcribe.py` → `output_console.print`)

If `--output` is provided, the exporter writes to file. Otherwise, formatted text goes to `output_console` (stdout). Status messages (`print_info`) go to `console` (stderr). This separation enables piping:

```cmd
pvtt transcribe file audio.wav > transcript.txt
```

---

## Command Flow: `pvtt model download tiny.en`

### Full Call Chain

```
cli/app.py:main()
  → cli/model.py:download_model("tiny.en")
    → ModelManager()
    → console.status("Downloading tiny.en...")    # spinner on stderr
    → ModelManager.download("tiny.en")
      → MODEL_REPO_MAP["tiny.en"] → "Systran/faster-whisper-tiny.en"
      → from huggingface_hub import snapshot_download  # lazy import
      → snapshot_download(repo_id, local_dir=models_dir/"tiny.en")
    → print_success(...)                           # stderr
```

### Key Details

**Model name resolution**: `MODEL_REPO_MAP` maps short names to HuggingFace repo IDs. This is the only place that knows about the Systran namespace:

```python
MODEL_REPO_MAP = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    # ... 16 entries
}
```

Unknown names pass through as-is (assumed to be full repo IDs).

**Network isolation**: `snapshot_download` is the sole network call in the entire application. It's lazy-imported inside the `download()` method — the `huggingface_hub` module is never loaded during transcription.

**Model validation**: `_is_valid_model_dir()` checks for `model.bin` — the CTranslate2 binary format marker. This is used by `list_models()`, `get_model_path()`, and `get_model_info()`.

---

## Configuration Resolution Pipeline

```
                    ┌──────────────────┐
                    │  Pydantic Defaults│  ModelConfig.name = "large-v3-turbo"
                    │  (schema.py)      │  ModelConfig.local_files_only = True
                    └────────┬─────────┘
                             │ deep_merge
                    ┌────────▼─────────┐
                    │  Global TOML     │  %LOCALAPPDATA%\pvtt\config.toml
                    │  (load_toml)     │  [model]
                    └────────┬─────────┘  name = "small.en"
                             │ deep_merge
                    ┌────────▼─────────┐
                    │  Env Vars        │  PVTT_MODEL_NAME=tiny.en
                    │  (load_env)      │  PVTT_TRANSCRIPTION_LANGUAGE=en
                    └────────┬─────────┘
                             │ deep_merge
                    ┌────────▼─────────┐
                    │  CLI Overrides   │  --model base.en --beam-size 3
                    │  (cli_overrides) │
                    └────────┬─────────┘
                             │ PvttConfig.model_validate()
                    ┌────────▼─────────┐
                    │  Validated Config │  PvttConfig instance
                    │  (Pydantic v2)   │  with field_validators applied
                    └──────────────────┘
```

**Env var convention**: `PVTT_SECTION_FIELD` → `{"section": {"field": value}}`. Split on first `_` after prefix. Example: `PVTT_MODEL_DEVICE=cpu` → `{"model": {"device": "cpu"}}`.

**C# equivalent**: This is `ConfigurationBuilder.AddJsonFile().AddEnvironmentVariables().AddCommandLine().Build()` — same layered override semantics, but using dict merging instead of the `IConfiguration` provider chain.

---

## Engine Abstraction & Registry

### Factory Pattern

```
engine/registry.py                     engine/base.py
┌─────────────────────────┐            ┌──────────────────────────┐
│ _ENGINE_REGISTRY = {    │            │ class InferenceEngine    │
│   "faster-whisper":     │──creates──→│   (Protocol):            │
│     FasterWhisperEngine │            │   load_model()           │
│ }                       │            │   transcribe() → Iterator│
│                         │            │   is_loaded: bool        │
│ get_engine(name=None)   │            │   model_name: str | None │
│   → factory()           │            └──────────────────────────┘
└─────────────────────────┘                       ▲
                                                  │ structural typing
                                       ┌──────────┴───────────────┐
                                       │ FasterWhisperEngine      │
                                       │   (no inheritance)       │
                                       │   _model: WhisperModel   │
                                       │   load_model() → lazy    │
                                       │     import + C++ init    │
                                       │   transcribe() → adapt   │
                                       │     segments to pvtt     │
                                       │     types                │
                                       └──────────────────────────┘
```

**Auto-registration**: `_register_builtins()` is called at module load time (bottom of `registry.py`). The lazy import inside `register_engine()` means `faster_whisper` is only imported when `FasterWhisperEngine.load_model()` is actually called.

**C# equivalent**: This is `IServiceCollection.AddSingleton<IInferenceEngine, FasterWhisperEngine>()` — but using a string-keyed dictionary instead of the DI container. The Protocol (structural typing) means no `class FasterWhisperEngine : IInferenceEngine` declaration. If the methods match, mypy accepts it.

The export registry (`export/registry.py`) follows the identical pattern for `Exporter` Protocol implementations.

---

## Error Handling Chain

Exceptions propagate from the innermost layer to the CLI boundary:

```
faster_whisper.WhisperModel raises
  → FasterWhisperEngine classifies:
      "not found" in message → ModelNotFoundError
      anything else          → EngineError
    → Transcriber propagates (no catch)
      → cli/transcribe.py propagates (no catch)
        → cli/app.py:main() catches PvttError
          → print_error() to stderr
          → typer.Exit(code=1)
```

### Exception Hierarchy

```
PvttError                    ← base (like ApplicationException)
├── ConfigError              ← TOML parse failure, Pydantic validation
├── ModelNotFoundError       ← model not downloaded locally
├── ModelDownloadError       ← network/HuggingFace failure
├── EngineError              ← inference runtime failure
│   └── EngineNotFoundError  ← unknown engine name in registry
├── AudioError               ← bad file path or unsupported format
├── ExportError              ← output formatting/write failure
└── HardwareError            ← GPU detection failure
```

**Design rule**: The CLI layer (`cli/`) never catches exceptions. It delegates entirely to core, which raises typed exceptions. `main()` is the sole catch point. This is the "let it bubble" pattern — the same philosophy as letting exceptions propagate to ASP.NET middleware rather than catching in controllers.

---

## Privacy Enforcement Points

Privacy is enforced at three architectural levels, not just configuration:

```
Level 1: Config Default          schema.py    ModelConfig.local_files_only = True
                                              (Pydantic default — must be explicitly overridden)

Level 2: Engine Constructor      faster_whisper.py   WhisperModel(..., local_files_only=True)
                                              (passed through from config to C++ runtime)

Level 3: Architecture            model_manager.py    download() is the ONLY method that imports
                                              huggingface_hub. No other code path can reach
                                              the network. The import is inside the method body.
```

Even if `local_files_only` were accidentally set to `False` in config, the architecture prevents implicit downloads because:
- `Transcriber._ensure_model_loaded()` calls `get_model_path()` which only checks local disk
- If the model isn't found locally, it raises `ModelNotFoundError` — it never falls through to download
- `huggingface_hub` is only imported inside `ModelManager.download()`, never at module level

---

## Dual-Console I/O Architecture

```
cli/formatters.py:
    console = Console(stderr=True)      # status, errors, spinners
    output_console = Console()           # transcription text (stdout)

                    ┌─────────────────────────────────────┐
                    │              Process                  │
                    │                                       │
                    │  print_info("Transcribing...")        │
                    │    → console (stderr)                 │
                    │                                       │
                    │  print_error("Model not found")       │
                    │    → console (stderr)                 │
                    │                                       │
                    │  output_console.print(transcript)     │
                    │    → stdout                           │
                    └──────────┬────────────┬──────────────┘
                               │            │
                          stderr ↓      stdout ↓
                        ┌────────┐    ┌──────────────┐
                        │ screen │    │ transcript.txt│
                        └────────┘    │ (via > or |)  │
                                      └──────────────┘
```

**Why this matters**: `pvtt transcribe file audio.wav > out.txt` writes only transcription text to the file. Progress spinners, error messages, and model loading status stay on screen (stderr). This is the Unix filter pattern — same reason `curl` shows progress on stderr.

**C# equivalent**: Writing to `Console.Error` vs `Console.Out`. In ASP.NET terms, it's separating the response body (stdout) from diagnostic logging (stderr).

---

## Data Types Through the Pipeline

```
CLI input (string args)
  ↓  Typer parsing
dict[str, dict[str, object]]       # cli_overrides
  ↓  load_config() + model_validate()
PvttConfig                          # Pydantic model (validated, typed)
  ↓  Transcriber reads config sections
TranscribeOptions                   # frozen dataclass (beam_size, language, etc.)
  ↓  engine.transcribe()
Iterator[TranscriptionSegment]      # frozen dataclass (start, end, text, logprob)
  ↓  list() + join
TranscriptionResult                 # frozen dataclass (segments, text, duration)
  ↓  PlainTextExporter.format()
str                                 # plain text output
  ↓  output_console.print()
stdout
```

**Boundary rule**: Pydantic models exist only at the config boundary. Internal data types are frozen dataclasses — lightweight, immutable, no validation overhead. This is like using DTOs at the API boundary but plain records internally.

---

## References

- [[ImplementationPlan|Implementation Plan]] — tech stack mapping, AI concepts, framework deep dives
- [[info]] — audience context for this guide
