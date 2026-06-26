import re

import jiwer
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

# Bug fix: original used ReduceToListOfListOfWords for both WER and CER,
# making CER identical to WER for every model in the leaderboard.
CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    return float(wer(
        reference, hypothesis,
        reference_transform=WER_TRANSFORM,
        hypothesis_transform=WER_TRANSFORM,
    ))


def compute_cer(reference: str, hypothesis: str) -> float:
    return float(cer(
        reference, hypothesis,
        reference_transform=CER_TRANSFORM,
        hypothesis_transform=CER_TRANSFORM,
    ))


def compute_semantic_similarity(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer,
) -> float:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    ref_emb = model.encode(ref_norm)
    hyp_emb = model.encode(hyp_norm)
    return float(cosine_similarity([ref_emb], [hyp_emb])[0][0])
