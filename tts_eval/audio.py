import logging
from pathlib import Path

import librosa
import soundfile as sf


def validate_and_fix_wav(path: Path, logger: logging.Logger) -> bool:
    """
    Cross-platform WAV validation using soundfile (replaces Unix `file` command).
    Re-encodes to PCM_16 @ 16 kHz if the file is not standard PCM WAV.
    Returns False if the file is unreadable (caller should skip the sample).
    """
    try:
        info = sf.info(str(path))
        if info.format == "WAV" and info.subtype in {"PCM_16", "PCM_24", "FLOAT"}:
            return True
        # Non-PCM WAV (e.g. MP3-in-WAV container) — re-encode
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
        sf.write(str(path), audio, 16000, subtype="PCM_16")
        logger.warning("Re-encoded non-PCM audio: %s", path.name)
        return True
    except sf.SoundFileError:
        # Completely unreadable — try librosa as last resort
        try:
            audio, sr = librosa.load(str(path), sr=16000, mono=True)
            sf.write(str(path), audio, 16000, subtype="PCM_16")
            logger.warning("Recovered unreadable audio via librosa: %s", path.name)
            return True
        except Exception as e:
            logger.error("Cannot read audio file %s: %s", path.name, e)
            return False
    except Exception as e:
        logger.error("Unexpected error validating %s: %s", path.name, e)
        return False


def discover_models(outputs_dir: Path) -> list[str]:
    """Return sorted list of model subdirectory names under outputs_dir."""
    if not outputs_dir.is_dir():
        return []
    return sorted(d.name for d in outputs_dir.iterdir() if d.is_dir())
