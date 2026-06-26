#!/usr/bin/env python3
"""
TTS Leaderboard Evaluation CLI

Usage:
    python evaluate.py --model chatterbox
    python evaluate.py --all
    python evaluate.py --model chatterbox --results custom_results.csv
    python evaluate.py --all --device cpu
    python evaluate.py --all --dry-run
"""
from tts_eval.config import load_config
from tts_eval.pipeline import run_evaluation

if __name__ == "__main__":
    config = load_config()
    run_evaluation(config)
