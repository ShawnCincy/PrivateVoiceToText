---
title: Implementation Plan — C# Developer Guide
parent: "[[info]]"
audience: Experienced C# developers
tags: [guide, architecture, AI, implementation]
---

# PrivateVoiceToText — Implementation Plan for C# Developers

This guide maps the pvtt project's Python/AI stack to concepts you already know from C# and .NET. It covers the frameworks, AI techniques, and architecture decisions — with enough depth to contribute or port components, not just consume them.

See [[ImplementationPlan|the main Implementation Plan]] for the full roadmap and framework research.

---

## Tech Stack — C# Equivalents

| pvtt Component | C# / .NET Equivalent | Notes |
|---|---|---|
| **Python 3.10+** | .NET 8+ | Both use type annotations; Python's are optional at runtime |
| **Typer (CLI)** | `System.CommandLine` | Typer uses type hints to generate CLI args, similar to `System.CommandLine`'s binding |
| **Rich (terminal UI)** | `Spectre.Console` | Nearly identical API philosophy — panels, tables, progress bars |
| **Pydantic v2 (config/validation)** | `System.Text.Json` + FluentValidation | Pydantic combines serialization and validation in one model class |
| **TOML config files** | `appsettings.json` / `IConfiguration` | TOML is the Python ecosystem's JSON-with-comments |
| **pytest** | xUnit / NUnit | pytest fixtures ≈ xUnit class fixtures + DI |
| **ruff (linter)** | Roslyn analyzers | ruff replaces flake8, isort, black — one tool for everything |
| **mypy (type checker)** | The C# compiler itself | Python's type system is opt-in; mypy adds compile-time-like checking |
| **Protocol (structural typing)** | Implicit interface satisfaction (duck typing) | Like if C# interfaces didn't require `implements` — match by shape |
| **sounddevice (PortAudio)** | NAudio | Both wrap native audio APIs; sounddevice binds PortAudio's C callbacks |

---

## Project Architecture

### Protocol-Based Engine Abstraction

pvtt uses Python's `Protocol` — the closest analog to a C# interface that uses structural typing rather than explicit implementation.

**C# mental model:**
```csharp
// C# — explicit interface
public interface IInferenceEngine
{
    IEnumerable<Segment> Transcribe(string audioPath, TranscribeOptions opts);
    void LoadModel(string name, string device, string computeType);
}

public class FasterWhisperEngine : IInferenceEngine { ... }
```

**Python equivalent in pvtt:**
```python
# engine/base.py — structural typing, no inheritance required
class InferenceEngine(Protocol):
    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Iterator[Segment]: ...
    def load_model(self, name: str, device: str, compute_type: str) -> None: ...

# engine/faster_whisper.py — satisfies Protocol by shape alone
class FasterWhisperEngine:
    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Iterator[Segment]:
        segments, info = self.model.transcribe(str(audio), **opts.to_dict())
        yield from segments
```

No `class FasterWhisperEngine(InferenceEngine)` needed. If the methods match, mypy accepts it. This is how Go interfaces work too.

### Layered Configuration

The config resolution chain mirrors ASP.NET's `IConfiguration` layering:

```
Pydantic defaults    →  appsettings.json defaults
Global TOML          →  appsettings.Production.json
Profile TOML         →  User Secrets
Env vars (PVTT_*)    →  Environment variables
CLI args             →  Command-line args
```

Each layer overrides the previous. Pydantic v2 validates the merged result — think FluentValidation baked into the model class:

```python
class ModelConfig(BaseModel):
    name: str = "large-v3-turbo"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal["auto", "float16", "int8", "float32"] = "auto"
    local_files_only: bool = True  # privacy: no network calls

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in KNOWN_MODELS:
            raise ValueError(f"Unknown model: {v}")
        return v
```

### Three-Thread Streaming Model

The real-time pipeline uses three threads. If you've worked with `System.Threading.Channels`, this maps directly:

```
Audio Thread          Pipeline Thread         Main Thread (CLI)
─────────────        ──────────────          ─────────────────
PortAudio callback   Read from queue         Rich live display
  → queue.put()      → Silero VAD filter     User input (Ctrl+C)
                     → Faster-Whisper
                     → Output formatter
```

Python's `queue.Queue` is the equivalent of `Channel<T>`. The GIL isn't a concern because the hot paths (PortAudio capture, CTranslate2 inference) execute in C/C++ and release the GIL.

### Thin CLI Layer

The `cli/` package contains zero business logic — it parses arguments via Typer and delegates to `core/`. This is the same pattern as keeping controllers thin in ASP.NET MVC:

```python
# cli/transcribe.py — thin delegation
@app.command()
def file(
    input_path: Path = typer.Argument(...),
    model: str = typer.Option("large-v3-turbo"),
    format: str = typer.Option("text"),
) -> None:
    config = load_config(model=model)
    transcriber = Transcriber(config)
    result = transcriber.transcribe_file(input_path)
    formatter = get_exporter(format)
    console.print(formatter.format(result))
```

---

## AI Concepts for C# Developers

### Whisper — The Core Model

Whisper is an **encoder-decoder Transformer** trained on 680,000 hours of multilingual audio. For C# developers, think of it as a function:

```
f(audio_samples[]) → text
```

Under the hood:
1. **Encoder**: Converts raw audio into a sequence of feature vectors (embeddings). Similar to how a CNN extracts features from images, but for spectrograms.
2. **Decoder**: Autoregressively generates text tokens, one at a time, conditioned on the encoder output. Each token prediction considers all previous tokens (like `StringBuilder.Append` but probabilistic).

Key parameters you'll interact with:

| Parameter | Type | Effect |
|---|---|---|
| `beam_size` | int | Number of hypotheses to track during decoding. Higher = more accurate, slower. Default 5. |
| `temperature` | float | Controls randomness. 0.0 = greedy (deterministic). Higher = more diverse but less accurate. |
| `initial_prompt` | string | Up to 224 tokens of context that biases the decoder toward specific vocabulary. |
| `language` | string? | Force a language. `null` = auto-detect. |
| `compute_type` | string | Quantization: `float16`, `int8_float16`, `int8`. INT8 halves VRAM with negligible accuracy loss. |

### CTranslate2 — The Inference Engine

CTranslate2 is to Whisper what CoreRT/NativeAOT is to .NET — it compiles the model into an optimized inference runtime. Written in C++, it provides:

- **INT8 quantization**: Reduces model weights from 32-bit floats to 8-bit integers. ~50% VRAM reduction, <1% accuracy loss.
- **Fused operations**: Multiple neural network layers merged into single GPU kernel calls. Like how the JIT inlines method chains.
- **Batch parallelism**: Process multiple audio chunks simultaneously on one GPU.

Faster-Whisper wraps CTranslate2 with a Python API. The `WhisperModel` class loads a converted model and exposes `transcribe()`.

### Quantization — INT8 vs FP16

Quantization maps high-precision weights to lower precision:

```
FP32 (4 bytes per weight)  →  standard training precision
FP16 (2 bytes per weight)  →  half precision, ~6GB VRAM for large-v3-turbo
INT8 (1 byte per weight)   →  ~3.5GB VRAM, negligible accuracy loss
```

Think of it like changing `decimal` to `float` to `Half` — you lose precision but gain speed and memory. CTranslate2 handles the conversion transparently. The model files on disk are already quantized; no runtime conversion cost.

### Voice Activity Detection (VAD)

Silero VAD is a tiny neural network (~2MB) that classifies audio frames as speech or silence. It gates the pipeline:

```
Audio stream → VAD → [speech frames only] → Whisper
```

Without VAD, Whisper would hallucinate text during silence (a known Whisper behavior). VAD also enables chunking — split continuous audio into utterances at silence boundaries, then transcribe each chunk independently.

### Personalization Stack

pvtt personalizes transcription in three layers, each adding complexity:

**Layer 1 — Prompt Engineering (zero training)**

Whisper accepts an `initial_prompt` that biases the decoder. Supply a vocabulary list:

```python
initial_prompt = "Kubernetes, kubectl, etcd, gRPC, Istio, Envoy"
```

The decoder becomes more likely to produce these tokens. Analogous to providing a custom dictionary to a spell-checker. No GPU, no training — works immediately.

**Layer 2 — N-gram Language Model (minutes of CPU time)**

KenLM trains a statistical model on the user's own writing (emails, docs, code comments). During beam search, each hypothesis is scored:

```
score = acoustic_score + α × lm_score + β × sequence_length
```

Where `α` and `β` are tunable weights. This is **shallow fusion** — the acoustic model and language model are independent; scores are combined at decode time. Think of it as a weighted voting system between "what it heard" and "what the user typically writes."

KenLM trains in seconds on CPU. The resulting binary is a few MB.

**Layer 3 — LoRA Fine-Tuning (hours of GPU time)**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all 809M parameters:

1. Freeze the base model weights entirely.
2. Inject small trainable matrices (rank 16–32) into the attention layers.
3. Train only these matrices (~0.1–1% of total parameters) on the user's voice+transcript pairs.

```
Original weight matrix W (d×d):  frozen
LoRA decomposition:              W + BA  where B is d×r and A is r×d, r << d
Trainable parameters:            2 × d × r  (e.g., 2 × 1024 × 16 = 32,768 per layer)
```

In C# terms, LoRA is like wrapping a sealed class with a decorator that adds a small learned correction to each method call, without modifying the original class.

The adapter weights are ~60MB. They're merged into the base model and converted to CTranslate2 format at training time (not inference time), because CTranslate2 doesn't support runtime adapter loading.

### Model Sizes and VRAM Budget

| Model | Params | VRAM (FP16) | VRAM (INT8) | Speed | Recommended? |
|---|---|---|---|---|---|
| tiny.en | 39M | ~1 GB | <1 GB | ~10x | Testing only |
| small.en | 244M | ~2 GB | ~1.5 GB | ~4x | CPU fallback |
| medium.en | 769M | ~5 GB | ~3 GB | ~2x | Training target |
| **large-v3-turbo** | **809M** | **~6 GB** | **~3.5 GB** | **~8x** | **Default** |
| large-v3 | 1,550M | ~10 GB | ~5 GB | 1x | INT8 only on 8GB |

`large-v3-turbo` is the default: near-large-v3 accuracy at 8x the speed and 60% less VRAM.

---

## Key Frameworks Deep Dive

### Faster-Whisper

The primary inference backend. Reimplements Whisper using CTranslate2 for 2–6x speedup over PyTorch.

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="int8_float16",
    local_files_only=True,       # privacy: never phone home
)

segments, info = model.transcribe(
    "audio.wav",
    beam_size=5,
    initial_prompt="pvtt, CTranslate2, Whisper",
    vad_filter=True,             # built-in Silero VAD
)

for segment in segments:
    print(f"[{segment.start:.1f}s → {segment.end:.1f}s] {segment.text}")
```

C# developers can also access Whisper via **whisper.cpp** which has official C# bindings (Whisper.net on NuGet). The pvtt project plans a `whisper_cpp.py` engine backend as a future CPU-only option.

### HuggingFace Transformers + PEFT

The fine-tuning stack. PEFT (Parameter-Efficient Fine-Tuning) provides the LoRA implementation.

```python
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo",
    load_in_8bit=True,           # quantized training, fits in 8GB VRAM
)

lora_config = LoraConfig(
    r=16,                        # rank — lower = fewer params, less capacity
    lora_alpha=32,               # scaling factor
    target_modules=["q_proj", "v_proj"],  # which attention matrices to adapt
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 3,932,160 || all params: 809,000,000 || trainable%: 0.486
```

After training, the adapter is merged back into the base model and converted to CTranslate2 format for Faster-Whisper inference.

### KenLM

A fast C++ n-gram language model library. Trains on plain text, produces a compact binary.

```bash
# Build a 3-gram model from the user's writing
lmplz -o 3 < user_corpus.txt > user.arpa
build_binary user.arpa user.binary
```

At decode time, each beam search hypothesis is rescored:

```python
import kenlm
lm = kenlm.Model("user.binary")
score = lm.score("Kubernetes cluster autoscaling")  # log probability
```

The resulting score is combined with the acoustic score via shallow fusion. This biases transcription toward the user's vocabulary and phrasing without any neural network training.

### Silero VAD

A tiny (~2MB) ONNX model that classifies 30ms audio frames as speech or non-speech. Runs on CPU in microseconds.

```python
import torch
model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
get_speech_timestamps = utils[0]

timestamps = get_speech_timestamps(audio_tensor, model, threshold=0.5)
# [{'start': 1200, 'end': 4800}, {'start': 6000, 'end': 15000}]
```

In the streaming pipeline, VAD determines when to send accumulated audio to Whisper. Without it, Whisper hallucinates during silence.

---

## Build Phases — What Ships When

| Phase | Delivers | C# Analogy |
|---|---|---|
| **1. Foundation** | Config, engine abstraction, `pvtt transcribe file`, `pvtt model download` | Scaffold project, DI container, first endpoint |
| **2. Streaming** | Audio capture, VAD, `pvtt transcribe live`, SRT/VTT export, `pvtt doctor` | Add SignalR real-time pipeline |
| **3. Profiles** | Profile CRUD, vocabulary-driven `initial_prompt` personalization | User identity + per-user settings |
| **4. N-gram LM** | KenLM training, shallow fusion during beam search | Custom spell-check dictionary |
| **5. LoRA** | Data collection, PEFT training, adapter merge, evaluation | ML model fine-tuning pipeline |
| **6. Polish** | Profile import/export, CI, docs, error handling | Release prep |

---

## Privacy Guarantees

These are non-negotiable project constraints:

- `local_files_only=True` on all model loading — no network calls during inference.
- Model downloads are explicit (`pvtt model download`), never implicit.
- All audio, transcriptions, adapters, and config stay on local disk.
- No telemetry, analytics, or crash reporting.
- Audio content and transcription text are never logged at INFO or above.

For a C# developer evaluating this for enterprise use: the privacy model is equivalent to an air-gapped deployment. The only network call the application ever makes is the explicit model download, which fetches from HuggingFace Hub.

---

## Platform Data Layout

```
Windows:   %LOCALAPPDATA%\pvtt\
Linux:     ~/.local/share/pvtt/
macOS:     ~/Library/Application Support/pvtt/

<root>/
    config.toml              # Global config
    models/                  # Downloaded Whisper models (CTranslate2 format)
    profiles/
        default/
            config.toml      # Profile overrides
            vocabulary.txt   # Prompt engineering word list
            ngram.binary     # KenLM model
            adapter/         # Merged CTranslate2 model (personalized)
            training-data/   # Audio + transcript pairs for LoRA
```

Comparable to `%APPDATA%\<AppName>\` in a .NET desktop app, but using `platformdirs` (Python equivalent of `Environment.GetFolderPath`).

---

## References

- [[ImplementationPlan|Main Implementation Plan]] — full framework research and roadmap
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — primary inference backend
- [whisper.cpp C# bindings](https://github.com/ggml-org/whisper.cpp) — alternative backend with NuGet package (Whisper.net)
- [HuggingFace PEFT](https://github.com/huggingface/peft) — LoRA/DoRA implementation
- [KenLM](https://github.com/kpu/kenlm) — n-gram language model
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — optimized inference engine
