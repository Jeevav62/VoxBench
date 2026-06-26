import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from resemblyzer import VoiceEncoder
from sentence_transformers import SentenceTransformer

from tts_eval.audio import discover_models, validate_and_fix_wav
from tts_eval.config import EvalConfig
from tts_eval.metrics.asr import transcribe_with_retry
from tts_eval.metrics.nisqa import run_nisqa_batch
from tts_eval.metrics.speaker import compute_speaker_similarity
from tts_eval.metrics.text_metrics import (
    compute_cer,
    compute_semantic_similarity,
    compute_wer,
)
from tts_eval.results import upsert_result

_MIN_VALID_FRACTION = 0.5  # Require at least 50% of samples to succeed


def setup_logging(log_dir: Path, run_label: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}_{run_label}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)

    return logging.getLogger("tts_eval")


def evaluate_model(
    model_name: str,
    config: EvalConfig,
    openai_client: OpenAI,
    speaker_encoder: VoiceEncoder,
    semantic_model: SentenceTransformer,
    logger: logging.Logger,
) -> dict | None:
    """
    Run the full 4-metric evaluation for one TTS model.
    Returns aggregated results dict, or None if too many samples failed.
    """
    gen_dir = config.outputs_dir / model_name
    if not gen_dir.is_dir():
        logger.error("Model output directory not found: %s", gen_dir)
        return None

    logger.info("=" * 60)
    logger.info("Evaluating model: %s", model_name)
    logger.info("=" * 60)

    # NISQA batch — score all generated audio up front
    nisqa_scores = run_nisqa_batch(
        gen_dir, config.nisqa_weights, config.nisqa_repo, logger
    )

    metadata = pd.read_csv(config.metadata_path)

    wer_scores, cer_scores, sem_scores, spk_scores, mos_scores = [], [], [], [], []
    skipped = 0

    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        reference_text = str(row["text"])
        gen_path = gen_dir / f"{sample_id}.wav"
        ref_path = config.ref_audio_dir / f"{sample_id}.wav"

        if not gen_path.exists():
            logger.warning("[SKIP] Missing generated audio: %s", gen_path.name)
            skipped += 1
            continue
        if not ref_path.exists():
            logger.warning("[SKIP] Missing reference audio: %s", ref_path.name)
            skipped += 1
            continue

        logger.info("Processing %s ...", sample_id)

        if not validate_and_fix_wav(gen_path, logger):
            logger.warning("[SKIP] Unreadable audio: %s", gen_path.name)
            skipped += 1
            continue

        if config.dry_run:
            logger.info("  [DRY RUN] Would transcribe and score %s", sample_id)
            continue

        # ASR
        hypothesis = transcribe_with_retry(
            openai_client, gen_path, config.asr_model,
            config.asr_max_retries, config.asr_retry_base_delay, logger,
        )
        if hypothesis is None:
            logger.warning("[SKIP] ASR failed for %s", sample_id)
            skipped += 1
            continue

        # Text metrics
        w = compute_wer(reference_text, hypothesis)
        c = compute_cer(reference_text, hypothesis)
        sem = compute_semantic_similarity(reference_text, hypothesis, semantic_model)

        # Speaker similarity
        spk = compute_speaker_similarity(gen_path, ref_path, speaker_encoder, logger)
        if spk is None:
            logger.warning("[SKIP] Speaker similarity failed for %s", sample_id)
            skipped += 1
            continue

        # NISQA MOS
        mos = nisqa_scores.get(f"{sample_id}.wav", 0.0)

        wer_scores.append(w)
        cer_scores.append(c)
        sem_scores.append(sem)
        spk_scores.append(spk)
        mos_scores.append(mos)

        logger.info(
            "  WER: %.3f | CER: %.3f | Semantic: %.3f | Speaker: %.3f | MOS: %.3f",
            w, c, sem, spk, mos,
        )

    if config.dry_run:
        logger.info("[DRY RUN] %s — %d samples would be evaluated, %d skipped.",
                    model_name, len(metadata) - skipped, skipped)
        return None

    total = len(metadata)
    valid = len(wer_scores)
    if valid == 0 or valid / total < _MIN_VALID_FRACTION:
        logger.error(
            "Too few valid samples for %s: %d/%d. Skipping result write.",
            model_name, valid, total,
        )
        return None

    result = {
        "Model": model_name,
        "WER": round(float(np.mean(wer_scores)), 4),
        "CER": round(float(np.mean(cer_scores)), 4),
        "Semantic Sim": round(float(np.mean(sem_scores)), 4),
        "Speaker Sim": round(float(np.mean(spk_scores)), 4),
        "NISQA_MOS_avg": round(float(np.mean(mos_scores)), 4),
    }

    logger.info("--- %s RESULTS ---", model_name)
    for k, v in result.items():
        logger.info("  %s: %s", k, v)

    return result


def run_evaluation(config: EvalConfig) -> None:
    run_label = config.model_name if config.model_name else "all"
    logger = setup_logging(config.log_dir, run_label)

    logger.info("TTS Evaluation Pipeline v0.1.0")
    logger.info("Device: %s | Dry run: %s", config.device, config.dry_run)

    if config.evaluate_all:
        models = discover_models(config.outputs_dir)
        if not models:
            logger.error("No model subdirectories found in %s", config.outputs_dir)
            return
        logger.info("Found %d models: %s", len(models), ", ".join(models))
    else:
        models = [config.model_name]

    # Load shared heavy models once
    logger.info("Loading speaker encoder...")
    speaker_encoder = VoiceEncoder()

    logger.info("Loading semantic model...")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    openai_client = OpenAI(api_key=config.openai_api_key)

    all_results = []
    for model_name in models:
        result = evaluate_model(
            model_name, config,
            openai_client, speaker_encoder, semantic_model,
            logger,
        )
        if result is not None:
            upsert_result(result, config.results_file)
            all_results.append(result)
            logger.info("Results saved to %s", config.results_file)

    if len(all_results) > 1:
        logger.info("\n%s", "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        df = pd.DataFrame(all_results).set_index("Model")
        logger.info("\n%s", df.to_string())
