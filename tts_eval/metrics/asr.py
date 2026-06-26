import logging
import time
from pathlib import Path

from openai import APIError, OpenAI, RateLimitError


def transcribe_with_retry(
    client: OpenAI,
    audio_path: Path,
    model: str,
    max_retries: int,
    base_delay: float,
    logger: logging.Logger,
) -> str | None:
    """
    Transcribe audio with exponential backoff on rate limit errors.
    Returns None on permanent API error or exhausted retries — caller skips sample.
    """
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(model=model, file=f)
            return result.text.strip()
        except RateLimitError:
            wait = base_delay * (2 ** attempt)
            logger.warning(
                "Rate limit on %s (attempt %d/%d), retrying in %.1fs",
                audio_path.name, attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
        except APIError as e:
            logger.error("API error transcribing %s: %s", audio_path.name, e)
            return None
    logger.error("All %d retries exhausted for %s", max_retries, audio_path.name)
    return None
