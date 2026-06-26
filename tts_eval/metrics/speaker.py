import logging
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


def compute_speaker_similarity(
    gen_path: Path,
    ref_path: Path,
    encoder: VoiceEncoder,
    logger: logging.Logger,
) -> float | None:
    """
    Cosine similarity between speaker embeddings of generated and reference audio.
    Returns None on error so the caller can skip this sample.
    """
    try:
        gen_wav = preprocess_wav(str(gen_path))
        ref_wav = preprocess_wav(str(ref_path))
        gen_emb = encoder.embed_utterance(gen_wav)
        ref_emb = encoder.embed_utterance(ref_wav)
        return float(np.dot(gen_emb, ref_emb))
    except Exception as e:
        logger.error("Speaker similarity failed for %s: %s", gen_path.name, e)
        return None
