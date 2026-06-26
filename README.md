<div align="center">

# 🔊 VoxBench

**The open-source benchmark pipeline for comparing Text-to-Speech models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![OpenAI](https://img.shields.io/badge/ASR-OpenAI%20Whisper-412991?logo=openai)](https://platform.openai.com)
[![NISQA](https://img.shields.io/badge/MOS-NISQA-orange)](https://github.com/gabrielmittag/NISQA)

Drop in any TTS model's audio. Get a full objective scorecard in minutes.

</div>

---

## What is VoxBench?

VoxBench is a standardized benchmark pipeline built to evaluate and compare open-source Text-to-Speech (TTS) models side by side.

The TTS ecosystem has exploded — dozens of open-source models, each claiming to be the best. But fair comparison is hard: teams cherry-pick metrics, use different test sets, and report numbers that don't translate across papers. VoxBench fixes that with a single reproducible pipeline covering five complementary metrics across a diverse 10-sample evaluation set.

**Originally built during an internship** to evaluate and rank commercial and open-source TTS engines against each other. Open-sourced so the community can run the same benchmark on any model and contribute results to a shared leaderboard.

---

## Metrics

| Metric | What it measures | Range | Better |
|--------|-----------------|-------|--------|
| **WER** | Word Error Rate — transcription accuracy | 0 → ∞ | Lower |
| **CER** | Character Error Rate — fine-grained transcription accuracy | 0 → ∞ | Lower |
| **Semantic Similarity** | Meaning preservation (reference text ↔ transcript) | 0 → 1 | Higher |
| **Speaker Similarity** | Voice identity match (generated ↔ reference audio) | 0 → 1 | Higher |
| **NISQA MOS** | Predicted Mean Opinion Score for naturalness | 1 → 5 | Higher |

### How each metric works

**WER / CER** — Generated audio is transcribed via OpenAI Whisper. The transcript is compared to the reference text using [`jiwer`](https://github.com/jitsi/jiwer). WER counts word-level substitutions/insertions/deletions; CER counts character-level errors.

**Semantic Similarity** — Both reference text and Whisper transcript are embedded using [`sentence-transformers`](https://www.sbert.net/) (MiniLM-L6-v2). Cosine similarity between the two embeddings measures how well the meaning was preserved, even if the exact words differ.

**Speaker Similarity** — Reference and generated audio are embedded using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) d-vectors (speaker encoder). The dot product of the L2-normalized embeddings gives the cosine similarity between voice identities.

**NISQA MOS** — [NISQA](https://github.com/gabrielmittag/NISQA) predicts naturalness without any reference audio. Uses the `nisqa_tts.tar` model fine-tuned specifically for TTS evaluation. Score of 4.0+ indicates high naturalness.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   evaluate.py (CLI)  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   tts_eval/config   │  ← argparse + .env merge
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  tts_eval/pipeline  │  ← orchestrator + logging
                        └──────┬──────┬───────┘
                               │      │
          ┌────────────────────┤      ├──────────────────────┐
          │                    │      │                       │
   ┌──────▼──────┐   ┌─────────▼──┐  ┌▼───────────┐  ┌──────▼──────┐
   │  asr.py     │   │text_metrics│  │ speaker.py │  │  nisqa.py   │
   │  (Whisper)  │   │(WER/CER/   │  │(Resemblyzer│  │  (NISQA     │
   │  + retry    │   │ Semantic)  │  │  d-vector) │  │   MOS)      │
   └─────────────┘   └────────────┘  └────────────┘  └─────────────┘
          │
   ┌──────▼──────┐
   │  audio.py   │  ← cross-platform WAV validation
   └─────────────┘
          │
   ┌──────▼──────┐
   │  results.py │  ← upsert to leaderboard_results.csv
   └─────────────┘
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- CUDA GPU (optional, but recommended)

### Install

```bash
# Clone with NISQA submodule
git clone --recurse-submodules https://github.com/jeevav62/TTS-leaderboard-metrics.git
cd TTS-leaderboard-metrics

# CPU install
pip install -r requirements.txt

# GPU install (CUDA 12.1)
pip install -r requirements-gpu.txt
```

### Configure

```bash
cp .env.example .env
# Open .env and add your key:
# OPENAI_API_KEY=sk-...
```

### Run

```bash
# Evaluate a single model
python evaluate.py --model your-model-name

# Evaluate all models in outputs/
python evaluate.py --all

# Dry run — validate files without API calls (free)
python evaluate.py --all --dry-run
```

---

## Adding Your TTS Model

### Step 1 — Generate audio

Use the 10 test scripts in `Ten_Scripts.txt`. Generate one `.wav` file per script.

The test set covers diverse speech categories:
- Questions
- Pronunciations
- Paralinguistics
- Emotions
- Syntactic Complexity
- Foreign Words

### Step 2 — Name the files

Files must be named exactly:
```
S01.wav  S02.wav  S03.wav  S04.wav  S05.wav
S06.wav  S07.wav  S08.wav  S09.wav  S10.wav
```

### Step 3 — Place in outputs/

```
outputs/
└── your-model-name/
    ├── S01.wav
    ├── S02.wav
    ├── ...
    └── S10.wav
```

### Step 4 — Evaluate

```bash
python evaluate.py --model your-model-name
```

Results are automatically upserted into `leaderboard_results.csv`.

---

## CLI Reference

```
python evaluate.py [--model NAME | --all] [OPTIONS]

Target (required, mutually exclusive):
  --model NAME              Evaluate one model subdirectory
  --all                     Evaluate all subdirectories in --outputs-dir

Paths:
  --outputs-dir DIR         TTS output root         (default: outputs)
  --ref-audio-dir DIR       Reference audio dir     (default: eval_dataset1/audio)
  --metadata FILE           Metadata CSV            (default: eval_dataset1/metadata.csv)
  --nisqa-weights FILE      NISQA weights .tar      (default: NISQA/weights/nisqa_tts.tar)
  --results FILE            Output CSV              (default: leaderboard_results.csv)
  --log-dir DIR             Log directory           (default: logs)

API:
  --asr-model MODEL         OpenAI ASR model        (default: gpt-4o-mini-transcribe)
  --asr-max-retries N       Retries on rate limit   (default: 3)
  --asr-retry-base-delay S  Backoff base in seconds (default: 5.0)

Device:
  --device auto|cuda|cpu    Compute device          (default: auto)

Utility:
  --dry-run                 Validate files, skip API calls and result writes
```

---

## Dataset Format

`eval_dataset1/metadata.csv`:

```csv
sample_id,text,category,reference_audio_file
S01,"Can you tell me what time it is?",Questions,eval_dataset1/audio/S01.wav
S02,...
```

| Column | Description |
|--------|-------------|
| `sample_id` | File stem — must match `{sample_id}.wav` in audio dirs |
| `text` | Reference transcript (ground truth) |
| `category` | Test category for analysis |

---

## Project Structure

```
TTS-leaderboard-metrics/
├── evaluate.py                   # CLI entry point
├── tts_eval/
│   ├── __init__.py
│   ├── config.py                 # EvalConfig dataclass + argparse
│   ├── audio.py                  # Cross-platform WAV validation
│   ├── pipeline.py               # Main orchestrator + logging
│   ├── results.py                # CSV upsert
│   └── metrics/
│       ├── asr.py                # Whisper transcription + retry
│       ├── text_metrics.py       # WER, CER (fixed), semantic sim
│       ├── speaker.py            # Resemblyzer speaker sim
│       └── nisqa.py              # NISQA MOS batch wrapper
├── NISQA/                        # Git submodule
├── eval_dataset1/
│   ├── audio/                    # Reference: S01.wav – S10.wav
│   └── metadata.csv
├── outputs/                      # Generated TTS audio (per model)
├── logs/                         # Per-run logs (auto-created)
├── Ten_Scripts.txt               # 10 evaluation test scripts
├── leaderboard_results.csv       # Aggregated leaderboard
├── requirements.txt              # CPU dependencies
├── requirements-gpu.txt          # GPU (CUDA 12.1) dependencies
├── .env.example                  # API key template
└── .gitignore
```

---

## Sample Results

| Model | WER ↓ | CER ↓ | Semantic ↑ | Speaker ↑ | MOS ↑ |
|-------|-------|-------|------------|-----------|-------|
| rime-coda | 0.160 | — | 0.968 | 0.663 | 4.575 |
| stepaudio | 0.173 | — | 0.959 | 0.649 | 4.467 |
| chatterbox | 0.214 | — | 0.932 | 0.698 | 4.102 |
| Inflect-Nano-v1 | 0.291 | — | 0.884 | 0.618 | 3.507 |

> CER values are being re-evaluated — an earlier bug caused CER to equal WER for all models.

---

## Requirements

### CPU

```
torch >= 2.0.0
openai >= 1.30.0
librosa >= 0.10.0
soundfile >= 0.12.0
jiwer >= 3.0.0
sentence-transformers >= 2.2.0
Resemblyzer >= 0.1.4
scikit-learn >= 1.2.0
pandas >= 2.0.0
python-dotenv >= 1.0.0
PyYAML >= 6.0
```

### GPU (additional)

```bash
pip install -r requirements-gpu.txt  # adds CUDA-enabled torch
```

---

## Contributing

Contributions welcome — new metrics, bigger datasets, CLI improvements.

```bash
# Fork and clone
git clone https://github.com/your-username/TTS-leaderboard-metrics.git

# Add your model outputs
mkdir outputs/your-model-name
# ... copy S01.wav–S10.wav ...

# Evaluate and verify
python evaluate.py --model your-model-name

# Open a PR with updated leaderboard_results.csv
```

### Adding a new metric

1. Create `tts_eval/metrics/your_metric.py` with a single function
2. Call it in `tts_eval/pipeline.py` inside `evaluate_model()`
3. Add the column to `RESULT_COLUMNS` in `tts_eval/results.py`
4. Update `README.md` metrics table

---

## License

MIT — free to use, modify, and distribute.

---

<div align="center">
Built during internship · Open-sourced for the community
</div>
