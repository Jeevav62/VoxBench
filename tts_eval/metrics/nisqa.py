import logging
import sys
from pathlib import Path


def run_nisqa_batch(
    audio_dir: Path,
    weights_path: Path,
    nisqa_repo: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Run NISQA MOS prediction on all .wav files in audio_dir.
    Returns {filename: mos_score} mapping.

    sys.path management is idempotent — safe to call multiple times.
    tr_bs_val and tr_num_workers are passed explicitly because predict()
    reads them directly; the checkpoint merge in _loadModel() fills the rest.
    """
    repo_str = str(nisqa_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from nisqa.NISQA_model import nisqaModel  # noqa: PLC0415

    nisqa_args = {
        "mode": "predict_dir",
        "pretrained_model": str(weights_path.resolve()),
        "data_dir": str(audio_dir.resolve()),
        "ms_channel": None,
        "ms_max_segments": 15000,
        "output_dir": None,
        "num_workers": 0,
        "bs": 1,
        "tr_num_workers": 0,
        "tr_bs_val": 1,
    }

    try:
        model = nisqaModel(nisqa_args)
        df = model.predict()
        results = {
            Path(row["deg"]).name: float(row["mos_pred"])
            for _, row in df.iterrows()
        }
        logger.info("NISQA scored %d files in %s", len(results), audio_dir.name)
        return results
    except ValueError as e:
        logger.error("NISQA batch failed for %s: %s", audio_dir.name, e)
        return {}
    except Exception as e:
        logger.error("NISQA unexpected error for %s: %s", audio_dir.name, e)
        return {}
