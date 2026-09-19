"""
Feature engineering on cleaned data, preparing it for model training.
"""

from pathlib import Path

import pandas as pd

from trial_conversion_model.data_cleaning import INTERIM_DATA_DIR

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "03_processed"


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["sessions_3d"] = data[["sessions_day1", "sessions_day2", "sessions_day3"]].sum(axis=1)
    data["active_days_3d"] = (data[["sessions_day1", "sessions_day2", "sessions_day3"]] > 0).sum(axis=1)
    data["day1_share"] = data["sessions_day1"] / data["sessions_3d"]
    data["listen_share"] = data["listen_sessions_3d"] / data["sessions_3d"]
    data["avg_session_minutes"] = data["total_minutes_3d"] / data["sessions_3d"]

    # trials with no sessions in the first 3 days divide by zero above; zero
    # engagement is real information, so those become zeros rather than NaN
    print("trials with no sessions in first 3 days:", (data["sessions_3d"] == 0).sum())
    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        data[col] = data[col].fillna(0)

    return data


def load_and_add_features(
    input_path: Path = INTERIM_DATA_DIR / "trials_clean.csv",
    output_path: Path = PROCESSED_DATA_DIR / "trials_processed.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = add_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    load_and_add_features()
