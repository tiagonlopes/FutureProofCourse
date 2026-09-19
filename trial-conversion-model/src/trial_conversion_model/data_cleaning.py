"""
Cleaning all raw data and preparing it for feature engineering and model training.
"""

from pathlib import Path

import pandas as pd

from trial_conversion_model.fetch_data import RAW_DATA_DIR

INTERIM_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "02_interim"
DATE_COLUMNS = ["snapshot_date", "trial_started_at"]


def clean_trials(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("missing values per column:")
    print(df.isna().sum())
    print("duplicate trials:", df.duplicated(subset="trial_id").sum())

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col])

    return df


def load_and_clean(
    input_path: Path = RAW_DATA_DIR / "trials_raw.csv",
    output_path: Path = INTERIM_DATA_DIR / "trials_clean.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = clean_trials(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    load_and_clean()
