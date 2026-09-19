"""
Read and Fetch data from the database.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "01_raw"
TRIAL_SNAPSHOT_QUERY = "SELECT * FROM ml.trial_snapshot_latest"


def get_engine():
    load_dotenv()
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )
    return create_engine(url)


def fetch_trials_raw(output_path: Path = RAW_DATA_DIR / "trials_raw.csv") -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(TRIAL_SNAPSHOT_QUERY, engine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    fetch_trials_raw()
