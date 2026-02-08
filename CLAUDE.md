# CLAUDE.md — PrivateVoiceToText (pvtt)

## Project

Local-only, privacy-first voice-to-text CLI. Runs Whisper inference entirely on user hardware. No cloud APIs, no telemetry, no data leaves the machine. **Mission: empower everyone to create without fear of censorship.**

- **Package**: `pvtt` (install via `pip install -e .` or `pip install -e ".[train,lm]"`)
- **Python**: 3.10+
- **Default model**: `large-v3-turbo` (809M params, ~6GB VRAM FP16, ~3.5GB INT8)
- **Target GPU**: NVIDIA 8–12GB VRAM (RTX 3060/3070/4060/4070)
- **Status**: Phase 1 (foundation) — see `docs/plans/ImplementationPlan.md`

## Tech Stack

| Role | Tool |
|------|------|
| CLI | Typer + Rich |
| Config | Pydantic v2 + TOML |
| Inference | Faster-Whisper (CTranslate2) |
| Audio capture | sounddevice (PortAudio) |
| VAD | Silero VAD |
| Fine-tuning | HuggingFace Transformers + PEFT (LoRA/DoRA) |
| N-gram LM | KenLM |
| Testing | pytest |
| Linting | ruff + mypy |

## Project Structure

```
src/pvtt/
    __init__.py
    __main__.py                # python -m pvtt entry point
    cli/                       # Thin Typer layer — delegates to core
        app.py                 # Root app, global options
        transcribe.py          # pvtt transcribe {live,file}
        train.py               # pvtt train {collect,finetune,evaluate,build-lm}
        model.py               # pvtt model {download,list,info,remove}
        profile.py             # pvtt profile {create,list,show,switch,delete,export,import}
        config_cmd.py          # pvtt config {show,set,path}
        formatters.py          # Rich output helpers
    core/                      # Business logic (no CLI/IO knowledge)
        transcriber.py, streaming.py, batch.py, trainer.py
        personalizer.py, model_manager.py, profile_manager.py
    engine/                    # Inference backend abstraction
        base.py                # InferenceEngine Protocol
        faster_whisper.py      # Primary backend
        whisper_cpp.py         # Future CPU-only backend
        registry.py            # Engine factory
    audio/                     # Audio I/O
        capture.py, file_reader.py, vad.py, preprocessing.py
    personalization/           # Personalization pipeline
        prompt_builder.py, ngram_lm.py, lora_adapter.py
        training/              # Data collection, PEFT training, evaluation
    export/                    # Output formatters (plain, SRT, VTT, JSON)
        base.py, plain_text.py, srt.py, vtt.py, json_export.py, registry.py
    config/
        schema.py              # Pydantic models for all config sections
        loader.py              # Layered config resolution
        paths.py               # Platform-aware data dirs
    util/
        hardware.py            # GPU/VRAM detection, compute type selection
        logging.py, types.py
tests/
    unit/                      # Fast, no GPU/model needed
    integration/               # CliRunner + tiny model tests
    fixtures/                  # Test audio files
docs/                          # Obsidian-compatible markdown vault
    plans/                     # Architecture & implementation plans
    guides/                    # User guides (audience-specific subdirs)
```

## Architecture Decisions

1. **Engine Protocol** — `engine/base.py` defines a Protocol. Swap backends (Faster-Whisper, whisper.cpp) without touching core. New backend = implement Protocol + register in `engine/registry.py`.
2. **LoRA merge at train time** — CTranslate2 cannot load LoRA at runtime. Merge adapter into base, convert to CT2 format, store per-profile. Each personalized model is ~1–3GB (INT8).
3. **Three-thread streaming** — Main thread (CLI) / Audio thread (PortAudio C callback → queue) / Pipeline thread (queue → VAD → inference → output). GIL not a bottleneck — heavy work is in C/C++.
4. **Layered config** — Defaults (Pydantic) → global TOML → profile TOML → env vars (`PVTT_*`) → CLI args. All validated through Pydantic v2.
5. **`local_files_only=True` default** — No network calls during inference. Models explicitly downloaded via `pvtt model download`.
6. **Tiered optional deps** — `pvtt` = core (lightweight). `pvtt[train]` adds PyTorch + PEFT. `pvtt[lm]` adds KenLM.

## Privacy Requirements (Non-Negotiable)

- **No network calls during inference or transcription.** `local_files_only=True` must be the default for all model loading.
- **All audio, transcriptions, adapters, and config stay on local disk.** Never transmit user data.
- **Model downloads are explicit** — only via `pvtt model download`, never implicit/lazy.
- **No telemetry, analytics, or crash reporting.** No phone-home of any kind.
- **Log nothing sensitive.** Never log audio content or transcription text at INFO level or above.
- When adding a dependency, verify it has no hidden network calls in its default configuration.

## Development Workflow

```bash
# Install editable with all optional deps
pip install -e ".[train,lm,dev]"

# Run CLI
pvtt doctor
pvtt transcribe file test.wav
pvtt transcribe live

# Tests
pytest tests/unit/                        # Fast unit tests (no GPU)
pytest tests/integration/                 # Needs tiny model
pytest tests/ -m "not slow"               # Skip slow-marked tests
pytest --cov=pvtt --cov-report=term-missing tests/

# Lint & type check
ruff check src/ tests/
ruff format src/ tests/
mypy src/pvtt/
```

## Coding Conventions

- **Type hints everywhere.** All function signatures, return types, class attributes. Use `from __future__ import annotations` in every module.
- **Protocols over ABCs** for engine/exporter interfaces.
- **Pydantic v2 models** for all config, validated data, and API boundaries.
- **`pathlib.Path`** — never raw string paths.
- **Docstrings**: Google-style. Required on all public classes, methods, and functions.
- **Error handling**: Raise domain-specific exceptions. CLI layer catches and presents with Rich. Never bare `except:`.
- **Imports**: absolute only (`from pvtt.config.schema import ...`), never relative.
- **No global mutable state.** Pass config/dependencies explicitly.
- **CLI layer is thin.** No business logic in `cli/` — it parses args and calls `core/`.
- **Constants**: UPPER_SNAKE_CASE. Define in the module that owns them.
- **Max line length**: 88 (ruff default).

## Testing Conventions

- **Unit tests**: `tests/unit/test_<module>.py` mirrors `src/pvtt/<module>.py`.
- **Integration tests**: `tests/integration/` — may need models, audio devices.
- **Fixtures**: shared fixtures in `tests/conftest.py`. Use `tmp_path` for file tests.
- **Naming**: `test_<unit>_<scenario>_<expected>` (e.g., `test_loader_missing_file_returns_defaults`).
- **Mocking**: mock external I/O (filesystem, audio devices, GPU). Never mock the unit under test.
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`.
- **AAA pattern**: Arrange / Act / Assert with blank lines separating sections.
- See `.agents/skills/python-testing-patterns/SKILL.md` for detailed patterns.

## Documentation Rules (Obsidian Vault)

The `docs/` directory is an Obsidian-compatible markdown vault.

- **YAML frontmatter** required in every markdown file.
- **`[[wikilinks]]`** for all internal cross-references between docs.
- **No orphan files** — every doc must be linked from a parent or index.
- **Folder conventions**: `docs/plans/` for architecture/implementation plans, `docs/guides/` for user guides.
- **Guide subdirectories** are organized by audience (e.g., `docs/guides/CSharp_Developer/`). Each has an `info.md` with `audience` and `guidance` frontmatter.
- **Do not break existing wikilinks** when moving or renaming files. Update all references.
- When referencing implementation details, link to the plan: `[[ImplementationPlan]]`.

## Platform Data Directories

```
Windows:  %LOCALAPPDATA%\pvtt\
Linux:    ~/.local/share/pvtt/
macOS:    ~/Library/Application Support/pvtt/
```

Subdirectories: `config.toml`, `models/`, `profiles/<name>/` (config.toml, vocabulary.txt, ngram.binary, adapter/, training-data/).

## CLI Commands Reference

```
pvtt transcribe {live,file}
pvtt train {collect,finetune,evaluate,build-lm}
pvtt model {download,list,info,remove}
pvtt profile {create,list,show,switch,delete,export,import}
pvtt config {show,set,path}
pvtt doctor
```

## Build Phases

Full roadmap in `docs/plans/ImplementationPlan.md`. Summary:

1. **Foundation** — pyproject.toml, config, engine abstraction, `pvtt transcribe file`, `pvtt model download/list`
2. **Streaming + CLI** — audio capture, VAD, live transcription, SRT/VTT/JSON export, `pvtt doctor`
3. **Profiles + Prompt Personalization** — profile CRUD, vocabulary-driven initial_prompt
4. **N-gram LM** — KenLM training, shallow fusion integration
5. **LoRA Fine-Tuning** — data collection, PEFT training, adapter merge, evaluation
6. **Polish & Packaging** — profile import/export, CI matrix, docs, error handling

## Agent Skills

- `.agents/skills/python-performance-optimization/SKILL.md` — profiling, cProfile, memory optimization
- `.agents/skills/python-testing-patterns/SKILL.md` — pytest fixtures, mocking, TDD, parameterization
