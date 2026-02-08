# PrivateVoiceToText — Research Guide & Implementation Plan

## Context

You want a privacy-focused voice-to-text CLI that runs AI entirely on your own hardware — no cloud APIs, no data leaving your machine. The tool should leverage mature open-source frameworks and support personalizing the model to an individual user's voice and writing style over time.

This document is the deliverable — a research guide and implementation plan to be placed in the repo as a reference document. No code will be written in this pass.

**Target hardware**: NVIDIA GPU with 8–12GB VRAM (e.g., RTX 3060/3070/4060/4070). Model and quantization defaults are chosen accordingly.

Contents:
1. **Research guide** — landscape of open-source STT frameworks, personalization techniques, and trade-offs
2. **High-level implementation plan** — architecture, CLI design, and phased build roadmap

---

# Part 1: Research Guide — Open-Source Voice-to-Text Frameworks

## Framework Comparison Matrix

| Framework | License | Maintained | Fine-Tunable | Real-Time | CPU-Only Feasible | Params Range |
|-----------|---------|------------|-------------|-----------|-------------------|-------------|
| **Whisper (OpenAI)** | MIT | Yes (large-v3-turbo, 2024) | Full + LoRA | Via wrappers | Small models only | 39M–1.55B |
| **Faster-Whisper** | MIT | Yes (14k+ stars, active 2026) | Via merge+convert | Yes | Yes (slower) | Same as Whisper |
| **whisper.cpp** | MIT | Very active (v1.8.3, Feb 2026) | Inference only | Yes (streaming) | Yes (quantized) | 31MB–3GB files |
| **Vosk** | Apache 2.0 | Moderate | Limited (vocab only) | Yes (native) | Yes (Raspberry Pi) | 50MB–1GB+ |
| **NVIDIA NeMo** | Apache 2.0 | Very active (NVIDIA-backed) | Full | Yes | Possible but slow | Various |
| **wav2vec 2.0 / HuBERT** | MIT | Stable (HuggingFace) | Excellent (few-shot) | Via wrappers | BASE only | 95M–1B |
| **SpeechBrain** | Apache 2.0 | Yes (v1.0, Jan 2024) | Full | Yes (Conformer) | Some models | Various |
| **WhisperX** | BSD 2-Clause | Yes (v3.7.6, Jan 2026) | Inherits Whisper | Batch-focused | No | Whisper + extras |
| **Distil-Whisper** | MIT | Yes (v3.5, early 2025) | Inherits Whisper | Yes | Small models | 166M–~800M |
| **Kaldi** | Apache 2.0 | Community only | Full (complex) | Yes | Yes | Custom |
| ~~DeepSpeech~~ | MPL 2.0 | **Archived June 2025** | N/A | N/A | N/A | N/A |
| ~~Coqui STT~~ | MPL 2.0 | **Dead** | N/A | N/A | N/A | N/A |

### Detailed Framework Notes

**Whisper (OpenAI)** — The dominant open-source ASR model. Encoder-decoder Transformer trained on 680k hours of multilingual audio. Supports 99+ languages. Fine-tuning well-supported via HuggingFace Transformers and PEFT/LoRA.

**Whisper model sizes** (VRAM shown for Faster-Whisper FP16 / INT8):

| Model | Params | VRAM (FP16) | VRAM (INT8) | Relative Speed | Fits 8–12GB? |
|-------|--------|-------------|-------------|----------------|---------------|
| tiny / tiny.en | 39M | ~1 GB | <1 GB | ~10x | Yes |
| base / base.en | 74M | ~1 GB | <1 GB | ~7x | Yes |
| small / small.en | 244M | ~2 GB | ~1.5 GB | ~4x | Yes |
| medium / medium.en | 769M | ~5 GB | ~3 GB | ~2x | Yes |
| **large-v3-turbo** | **809M** | **~6 GB** | **~3.5 GB** | **~8x** | **Yes (recommended)** |
| large-v3 | 1,550M | ~10 GB | ~5 GB | 1x | INT8 only |

The `large-v3-turbo` is the clear winner for 8–12GB GPUs: near-large-v3 accuracy at 8x the speed and 60% less VRAM.

**Faster-Whisper (SYSTRAN)** — Reimplementation of Whisper using CTranslate2 (C++ inference engine). 2–6x faster than PyTorch Whisper at same accuracy. VRAM drops from 11.3GB to 4.7GB (FP16) for large-v2, and to 3.1GB with INT8 quantization. Serves as the recommended backend for real-time streaming wrappers (Whisper-Streaming, WhisperLive). Active development, 14k+ GitHub stars.

**whisper.cpp** — Pure C/C++ Whisper port using GGML. Zero external dependencies. Supports quantized models (Q4–Q8) that cut memory 50–75% with minimal accuracy loss. Multi-backend GPU: CUDA, Vulkan (AMD/Intel iGPUs — 12x boost in v1.8.3), Metal (Apple Silicon), OpenVINO. Excellent for CPU-only and embedded. Bindings for Python, Go, Java, C#, and more. Very actively maintained (latest release Feb 2026).

**Vosk** — Lightweight, offline-first toolkit built on Kaldi. 50MB portable models, runs on Raspberry Pi. Native real-time streaming API. 20+ languages. Best for ultra-low-resource scenarios but limited fine-tuning depth compared to Whisper-family.

**NVIDIA NeMo** — Enterprise-grade toolkit. Canary model tops HuggingFace Open ASR Leaderboard (5.63% WER). FastConformer architecture. Comprehensive fine-tuning with multi-GPU support. Heavier infrastructure footprint. Best if you have NVIDIA hardware and want state-of-the-art accuracy.

**wav2vec 2.0 / HuBERT (Meta)** — Self-supervised models designed for fine-tuning. HuBERT LARGE achieves 4.7% WER on LibriSpeech with just 10 minutes of labeled data. Excellent for few-shot speaker adaptation. Primarily used as components within larger pipelines (SpeechBrain, NeMo).

**SpeechBrain** — Comprehensive PyTorch speech toolkit (ASR, speaker recognition, enhancement). 100+ pretrained models. First-class support for fine-tuning Whisper, wav2vec 2.0, HuBERT. Good middle ground between raw HuggingFace and NeMo.

**WhisperX** — Pipeline on top of Faster-Whisper adding word-level timestamps (via wav2vec2 forced alignment) and speaker diarization (via pyannote-audio). 70x realtime speed. Best for batch processing with speaker identification needs.

**Distil-Whisper (HuggingFace)** — Knowledge-distilled Whisper. 5.8x faster, 51% fewer params, within 1% WER. `distil-small.en` (166M params) suitable for mobile/edge. Can pair with full Whisper for speculative decoding (2x speedup, mathematically identical output).

---

## Personalization Techniques

### Layer 1: Zero-Training — Prompt Engineering (Immediate)
Whisper's `initial_prompt` parameter accepts up to 224 tokens of context that steers decoding. Provide a spelling guide of personal vocabulary, proper nouns, and domain terms. Use `carry_initial_prompt=True` for consistency across chunks. No GPU, no training — works out of the box.

### Layer 2: N-gram Language Model (Minutes of Setup, No GPU)
Train a KenLM n-gram model on the user's own writing (emails, docs, notes). Apply via shallow fusion during beam search: `score = acoustic_score + alpha * lm_score + beta * seq_length`. Biases transcription toward user's vocabulary and phrasing. KenLM trains in seconds on CPU.

### Layer 3: LoRA Fine-Tuning (Hours of Setup, Requires GPU)
Fine-tune Whisper with Low-Rank Adaptation (LoRA) on the user's voice recordings paired with corrected transcriptions.

| Technique | Data Needed | Params Trained | WER Improvement | VRAM Required |
|-----------|------------|----------------|-----------------|---------------|
| LoRA (rank 16–32) | 1–8 hours audio | 0.1–1% of model | ~10–20% relative | <8GB with INT8 |
| DoRA (Samsung Research) | 5–10 minutes | 139K params | ~20% relative | Similar to LoRA |
| Full fine-tuning | 5–20 hours | 100% of model | 20–40% relative | 24–32GB |

**Recommended**: LoRA with 8-bit quantization via `bitsandbytes`. Trains Whisper-large with <8GB VRAM. Adapter weights are ~60MB. Merge into base model and convert to CTranslate2 format for Faster-Whisper inference.

**Tools**: HuggingFace Transformers + PEFT library. Dedicated tools: `fast-whisper-finetuning` (Vaibhavs10), `whisper-finetune` (vasistalodagala).

### Layer 4: LLM Post-Processing (Optional, Highest Accuracy)
Run transcription output through a small local LLM fine-tuned on user's correction patterns. Fixes domain terms, punctuation style, capitalization preferences. Use Phi-3, Llama-3, or Mistral with LoRA. Entirely local.

### Contextual Biasing (Alternative to Fine-Tuning)
WhisperBiasing and zero-shot context biasing use prefix trees to guide transcription toward target vocabulary without any training. Achieves 43–44% reduction in biased-word error rate while maintaining overall WER.

---

## Privacy Analysis

All listed frameworks run 100% locally. No audio data needs to leave the device. Models download once and run offline. Key privacy properties:

- **whisper.cpp**: Zero dependencies, no network code whatsoever. Strongest privacy guarantee.
- **Faster-Whisper**: Set `local_files_only=True` to prevent any network calls. Models cached on disk.
- **Vosk**: Designed specifically for offline/privacy-sensitive deployment.
- **Training**: All LoRA fine-tuning runs locally via PyTorch. Audio, transcriptions, and adapter weights stay on disk.

---

## Recommendation for 8–12GB VRAM

**Primary stack**: Faster-Whisper for inference + HuggingFace Transformers + PEFT for fine-tuning + KenLM for n-gram personalization.

**Default model**: `large-v3-turbo` — 809M params, ~6GB VRAM with FP16, ~3.5GB with INT8. Fits comfortably in 8–12GB and offers ~8x speed of large-v3 with near-equivalent accuracy. This is the sweet spot for your hardware.

**Fine-tuning model**: `medium` or `large-v3-turbo` with 8-bit quantization + LoRA. Both fit in 8–12GB for training. `medium.en` (769M params) is the safest choice for training headroom; `large-v3-turbo` is feasible but tighter.

**Inference quantization**: INT8 by default (via CTranslate2). Cuts VRAM roughly in half with negligible accuracy loss.

**CPU-only fallback**: whisper.cpp with quantized models (Q5_1) for machines without a GPU. `small.en` at Q5_1 is 182MB and runs real-time on modern CPUs.

---

# Part 2: High-Level Implementation Plan

## Technology Choices

- **Language**: Python 3.10+ (best ecosystem support — Faster-Whisper, HuggingFace, PEFT, sounddevice are all Python-native)
- **CLI Framework**: Typer (type-hint-driven, auto-generated help, Rich integration)
- **Config**: Pydantic v2 models + TOML files
- **Inference**: Faster-Whisper (CTranslate2 backend)
- **Audio Capture**: sounddevice (PortAudio bindings)
- **VAD**: Silero VAD
- **Fine-Tuning**: HuggingFace Transformers + PEFT (LoRA/DoRA)
- **N-gram LM**: KenLM
- **Testing**: pytest
- **Linting**: ruff + mypy

## Project Structure

```
src/pvtt/                          # "Private Voice To Text" package
    __init__.py
    __main__.py                    # python -m pvtt
    cli/                           # Thin CLI layer (delegates to core)
        app.py                     # Root Typer app, global options
        transcribe.py              # pvtt transcribe {live,file}
        train.py                   # pvtt train {collect,finetune,evaluate,build-lm}
        model.py                   # pvtt model {download,list,info,remove}
        profile.py                 # pvtt profile {create,list,show,switch,delete}
        config_cmd.py              # pvtt config {show,set,path}
        formatters.py              # Rich console output helpers
    core/                          # Business logic (no CLI/IO knowledge)
        transcriber.py             # Orchestrates engine + personalization
        streaming.py               # Real-time mic -> VAD -> inference pipeline
        batch.py                   # Batch file transcription
        trainer.py                 # Fine-tuning orchestration
        personalizer.py            # All personalization layers
        model_manager.py           # Download, cache, list, delete models
        profile_manager.py         # Profile CRUD
    engine/                        # Inference backend abstraction
        base.py                    # InferenceEngine Protocol
        faster_whisper.py          # Faster-Whisper implementation
        whisper_cpp.py             # whisper.cpp implementation (future)
        registry.py                # Engine factory
    audio/                         # Audio I/O
        capture.py                 # Mic capture via sounddevice
        file_reader.py             # Audio file decoding
        vad.py                     # Silero VAD wrapper
        preprocessing.py           # Normalization, resampling, chunking
    personalization/               # Personalization pipeline
        prompt_builder.py          # Build Whisper initial_prompt from vocab
        ngram_lm.py                # KenLM training + shallow fusion
        lora_adapter.py            # LoRA adapter merge + CT2 conversion
        training/
            data_collector.py      # Guided audio+transcript collection
            data_pipeline.py       # Preprocessing for HF datasets
            lora_trainer.py        # PEFT LoRA/DoRA training loop
            evaluator.py           # WER/CER computation
    export/                        # Output formatting
        base.py, plain_text.py, srt.py, vtt.py, json_export.py, registry.py
    config/                        # Configuration
        schema.py                  # Pydantic models for all config
        loader.py                  # Layered config: defaults -> global -> profile -> CLI
        paths.py                   # Platform-aware paths (AppData / XDG / ~/Library)
    util/
        hardware.py                # GPU/VRAM detection, compute type selection
        logging.py, types.py
tests/
    unit/                          # Fast, no models needed
    integration/                   # CliRunner tests, tiny model tests
    fixtures/                      # Test audio files
```

## CLI Command Hierarchy

```
pvtt transcribe live     # Real-time mic transcription
pvtt transcribe file     # Batch file transcription

pvtt train collect       # Guided data collection (record + correct)
pvtt train finetune      # Run LoRA fine-tuning
pvtt train evaluate      # Compare base vs. fine-tuned WER
pvtt train build-lm      # Train KenLM n-gram from text corpus

pvtt model download      # Download a Whisper model
pvtt model list          # List cached models
pvtt model info          # Model details (params, VRAM estimate)
pvtt model remove        # Delete a cached model

pvtt profile create      # New user profile
pvtt profile list/show/switch/delete
pvtt profile export/import

pvtt config show/set/path
pvtt doctor              # System check: GPU, deps, models
```

## Key Architecture Decisions

1. **Engine abstraction** (`engine/base.py` Protocol) — Swap between Faster-Whisper and whisper.cpp without touching core logic. Adding a new backend = implement one interface + register.

2. **LoRA merge at training time, not inference time** — Faster-Whisper uses CTranslate2 which doesn't support runtime LoRA loading. Adapters are merged into the base model and converted to CT2 format. Each profile's personalized model is ~1–3GB on disk (with INT8 quantization). This preserves the 2–6x speed advantage of CTranslate2.

3. **Three-thread streaming model** — Main thread (CLI), audio thread (PortAudio C callback -> queue), pipeline thread (queue -> VAD -> inference -> output). Heavy computation is in C/C++ (PortAudio, CTranslate2) so the GIL is not a bottleneck.

4. **Layered config** — Defaults (Pydantic) -> global TOML -> profile TOML -> env vars -> CLI args. All validated via Pydantic v2.

5. **`local_files_only=True` as default** — No network calls during inference. Models must be explicitly downloaded via `pvtt model download`.

6. **Tiered optional dependencies** — Core install is lightweight. `pip install pvtt[train]` adds PyTorch/PEFT for fine-tuning. `pip install pvtt[lm]` adds KenLM. Users only install what they need.

## Data Flow: Real-Time Transcription

```
Mic (sounddevice) -> audio queue -> Silero VAD (speech/silence gating)
  -> speech segment -> Faster-Whisper transcribe(audio, initial_prompt=..., ...)
  -> segments -> KenLM rescoring (if enabled) -> text normalization
  -> Rich live display / file output / clipboard
```

## Data Flow: Fine-Tuning

```
pvtt train collect: Record -> base model transcribes -> user corrects -> save (wav, txt) pairs
pvtt train finetune: Load pairs as HF Dataset -> WhisperProcessor tokenization
  -> Load base Whisper in 8-bit -> Apply LoRA (rank 16, target q_proj/v_proj)
  -> Train with Seq2SeqTrainer -> Save adapter (~60MB)
  -> Merge adapter into base -> Convert to CTranslate2 -> Register with profile
pvtt train evaluate: Transcribe test set with base and adapted -> compare WER
```

## Platform Data Directories

```
Windows:  %LOCALAPPDATA%\pvtt\          (C:\Users\<user>\AppData\Local\pvtt\)
Linux:    ~/.local/share/pvtt/
macOS:    ~/Library/Application Support/pvtt/

<data_root>/
    config.toml                    # Global config
    models/                        # Downloaded model cache
    profiles/
        default/
            config.toml            # Profile overrides
            vocabulary.txt         # Spelling guide
            ngram.binary           # KenLM model
            adapter/               # Merged CT2 model (personalized)
            training-data/         # Audio+transcript pairs
```

## Phased Build Roadmap

### Phase 1: Foundation
Build: pyproject.toml, config system (schema + loader + paths), hardware detection, engine abstraction + Faster-Whisper backend, basic transcriber, plain text exporter, `pvtt transcribe file` command, `pvtt model download/list`.
**Milestone**: `pvtt model download tiny.en && pvtt transcribe file test.wav` prints text.

### Phase 2: Streaming + CLI Polish
Build: audio capture, Silero VAD, streaming pipeline, `pvtt transcribe live`, SRT/VTT/JSON exporters, Rich live display, `pvtt config` commands, `pvtt doctor`.
**Milestone**: `pvtt transcribe live --format srt -o meeting.srt` streams mic to SRT file.

### Phase 3: Profiles + Zero-Training Personalization
Build: profile manager, profile CLI, prompt builder from vocabulary, personalizer orchestration, wire `--profile` through all commands.
**Milestone**: Per-user profiles with vocabulary-driven prompt personalization.

### Phase 4: N-gram Language Model
Build: KenLM wrapper, `pvtt train build-lm`, shallow fusion integration.
**Milestone**: User's writing corpus biases transcription toward their vocabulary.

### Phase 5: LoRA Fine-Tuning Pipeline
Build: data collector (interactive record+correct), data pipeline, LoRA trainer, evaluator, adapter merge+convert, `pvtt train collect/finetune/evaluate`.
**Milestone**: Full loop — collect voice data -> train -> evaluate -> transcribe with personalized model.

### Phase 6: Polish & Packaging
Build: profile export/import, batch optimization, clipboard integration, comprehensive error handling, documentation, CI (Windows/Linux/macOS matrix).

### Future (Phase 7+)
- whisper.cpp engine backend for CPU-only deployment
- Speaker diarization via WhisperX/pyannote-audio
- Speaker verification gate (only transcribe authorized voice)
- Local LLM post-processing for writing style correction
- Plugin system for custom post-processors

## Verification Plan

After each phase, verify with these tests:

- **Phase 1**: `pip install -e .` succeeds; `pvtt model download tiny.en` downloads model; `pvtt transcribe file <wav>` produces correct-ish text; unit tests pass for config, exporters, engine protocol
- **Phase 2**: `pvtt transcribe live` captures mic and prints rolling transcript; Ctrl+C cleanly stops; SRT/VTT/JSON output is well-formed; `pvtt doctor` reports GPU/CPU status correctly
- **Phase 3**: `pvtt profile create test && pvtt profile list` shows profiles; vocabulary file influences transcription output
- **Phase 4**: `pvtt train build-lm --corpus <text files>` creates KenLM binary; transcription with LM enabled produces different (better) results for domain-specific terms
- **Phase 5**: `pvtt train collect` records and saves pairs; `pvtt train finetune` trains without error; `pvtt train evaluate` shows WER comparison; personalized model loads and transcribes
- **All phases**: `pytest tests/unit/` passes; `ruff check src/` clean; `mypy src/pvtt/` clean

---

## Key References

### Core Frameworks
- OpenAI Whisper: https://github.com/openai/whisper (MIT)
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper (MIT)
- whisper.cpp: https://github.com/ggml-org/whisper.cpp (MIT)
- Vosk: https://github.com/alphacep/vosk-api (Apache 2.0)
- NVIDIA NeMo: https://github.com/NVIDIA-NeMo/NeMo (Apache 2.0)
- SpeechBrain: https://github.com/speechbrain/speechbrain (Apache 2.0)
- WhisperX: https://github.com/m-bain/whisperX (BSD 2-Clause)
- Distil-Whisper: https://github.com/huggingface/distil-whisper (MIT)

### Fine-Tuning & Personalization
- HuggingFace PEFT: https://github.com/huggingface/peft
- fast-whisper-finetuning (LoRA, <8GB VRAM): https://github.com/Vaibhavs10/fast-whisper-finetuning
- whisper-finetune (custom datasets): https://github.com/vasistalodagala/whisper-finetune
- HuggingFace Fine-Tune Whisper Guide: https://huggingface.co/blog/fine-tune-whisper
- Samsung DoRA Speaker Personalization: https://research.samsung.com/blog/Speaker-Personalization-for-Automatic-Speech-Recognition-using-Weight-Decomposed-Low-Rank-Adaptation
- OpenAI Whisper Prompting Guide: https://cookbook.openai.com/examples/whisper_prompting_guide
- WhisperBiasing (contextual biasing without fine-tuning): https://github.com/BriansIDP/WhisperBiasing
- KenLM (n-gram language models): https://github.com/kpu/kenlm
- NeMo ASR Language Model Customization: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_language_modeling_and_customization.html

### Python CLI & Tooling
- Typer: https://typer.tiangolo.com/
- Rich: https://github.com/Textualize/rich
- Pydantic v2: https://docs.pydantic.dev/latest/
- sounddevice: https://python-sounddevice.readthedocs.io/
- Silero VAD: https://github.com/snakers4/silero-vad
- platformdirs: https://github.com/platformdirs/platformdirs

### Benchmarks & Comparisons
- Open Source STT Benchmarks 2026: https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- Choosing Whisper Variants: https://modal.com/blog/choosing-whisper-variants
- HuggingFace Open ASR Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
