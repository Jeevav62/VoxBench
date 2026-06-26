import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvalConfig:
    model_name: str | None
    evaluate_all: bool

    outputs_dir: Path
    ref_audio_dir: Path
    metadata_path: Path
    nisqa_weights: Path
    nisqa_repo: Path
    results_file: Path
    log_dir: Path

    openai_api_key: str
    asr_model: str

    asr_max_retries: int
    asr_retry_base_delay: float

    device: str
    dry_run: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate TTS model(s) against 4 audio quality metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", metavar="NAME",
                        help="Name of model subdirectory under --outputs-dir")
    target.add_argument("--all", action="store_true",
                        help="Evaluate every subdirectory found in --outputs-dir")

    p.add_argument("--outputs-dir", default="outputs",
                   help="Root directory containing per-model output folders")
    p.add_argument("--ref-audio-dir", default="eval_dataset1/audio",
                   help="Directory with reference .wav files")
    p.add_argument("--metadata", default="eval_dataset1/metadata.csv",
                   help="CSV with columns: sample_id, text, category")
    p.add_argument("--nisqa-weights", default="NISQA/weights/nisqa_tts.tar",
                   help="Path to NISQA pretrained weights (.tar)")
    p.add_argument("--nisqa-repo", default="NISQA",
                   help="Path to local NISQA repository")
    p.add_argument("--results", default="leaderboard_results.csv",
                   help="Output CSV path (upserted, not overwritten)")
    p.add_argument("--log-dir", default="logs",
                   help="Directory for per-run log files")

    p.add_argument("--asr-model", default="gpt-4o-mini-transcribe",
                   help="OpenAI ASR model to use for transcription")
    p.add_argument("--asr-max-retries", type=int, default=3,
                   help="Max retries on OpenAI rate limit errors")
    p.add_argument("--asr-retry-base-delay", type=float, default=5.0,
                   help="Base delay (seconds) for exponential backoff on retries")

    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device for ML models")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate files and print plan without running metrics or API calls")

    return p.parse_args()


def load_config() -> EvalConfig:
    """
    Merge priority: CLI args > .env > hardcoded defaults.
    Validates required paths and API key before returning.
    """
    args = _parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        print("       Copy .env.example to .env and fill in your key.", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path(args.outputs_dir).resolve()
    ref_audio_dir = Path(args.ref_audio_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    nisqa_weights = Path(args.nisqa_weights).resolve()
    nisqa_repo = Path(args.nisqa_repo).resolve()
    results_file = Path(args.results).resolve()
    log_dir = Path(args.log_dir).resolve()

    errors = []
    if not outputs_dir.is_dir():
        errors.append(f"outputs-dir not found: {outputs_dir}")
    if not ref_audio_dir.is_dir():
        errors.append(f"ref-audio-dir not found: {ref_audio_dir}")
    if not metadata_path.is_file():
        errors.append(f"metadata not found: {metadata_path}")
    if not nisqa_weights.is_file():
        errors.append(f"nisqa-weights not found: {nisqa_weights}")
    if not nisqa_repo.is_dir():
        errors.append(f"nisqa-repo not found: {nisqa_repo}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    return EvalConfig(
        model_name=args.model,
        evaluate_all=args.all,
        outputs_dir=outputs_dir,
        ref_audio_dir=ref_audio_dir,
        metadata_path=metadata_path,
        nisqa_weights=nisqa_weights,
        nisqa_repo=nisqa_repo,
        results_file=results_file,
        log_dir=log_dir,
        openai_api_key=api_key,
        asr_model=args.asr_model,
        asr_max_retries=args.asr_max_retries,
        asr_retry_base_delay=args.asr_retry_base_delay,
        device=device,
        dry_run=args.dry_run,
    )
