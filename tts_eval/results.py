from pathlib import Path

import pandas as pd

RESULT_COLUMNS = ["Model", "WER", "CER", "Semantic Sim", "Speaker Sim", "NISQA_MOS_avg"]


def load_results(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def upsert_result(result: dict, output_path: Path) -> None:
    """
    Write result row with upsert semantics: replace existing row for the same
    Model name, or append if new. Fixes the original bug of blind CSV append
    that created duplicate rows on re-runs.
    """
    df = load_results(output_path)
    df = df[df["Model"] != result["Model"]]
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(output_path, index=False)
