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
