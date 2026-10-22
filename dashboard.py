"""Read-only dashboard over what the scheduled jobs publish to the S3 bucket.

Run on a laptop with: uv run streamlit run dashboard.py
Needs S3_BUCKET and AWS credentials with read access (from .env or the shell).
"""

import io
import json
import os
import re
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BUCKET = os.environ["S3_BUCKET"]
DAYS = 30
LOCAL_METRICS = Path("models/metrics.json")

s3 = boto3.client("s3")


def list_keys(prefix: str) -> list[str]:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        keys += [obj["Key"] for obj in page.get("Contents", [])]
    return keys


def key_date(key: str) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", key)
    return pd.Timestamp(match.group(1)) if match else None


@st.cache_data(ttl=300)
def load_predictions() -> dict[pd.Timestamp, pd.DataFrame]:
    """One scored cohort per day, newest last."""
    days = {}
    for key in list_keys("predictions/"):
        day = key_date(key)
        if day is not None and key.endswith(".csv"):
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            days[day] = pd.read_csv(io.BytesIO(body))
    return dict(sorted(days.items()))


@st.cache_data(ttl=300)
def load_drift() -> pd.DataFrame:
    """One verdict per day. Backfilled history is placeholder data, not a check."""
    rows = []
    for key in list_keys("monitoring/"):
        day = key_date(key)
        if day is not None and key.endswith(".json"):
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            result = json.loads(body)
            rows.append(
                {
                    "day": day,
                    "drifted_share": result["drifted_share"],
                    "drift": result["drift"],
                    "backfilled": result.get("source") == "backfill",
                }
            )
    return pd.DataFrame(rows).sort_values("day") if rows else pd.DataFrame()


@st.cache_data(ttl=300)
def load_metrics() -> dict | None:
    if LOCAL_METRICS.exists():
        return json.loads(LOCAL_METRICS.read_text())
    try:
        body = s3.get_object(Bucket=BUCKET, Key="models/metrics.json")["Body"]
        return json.loads(body.read())
    except s3.exceptions.NoSuchKey:
        return None


st.set_page_config(page_title="Trial conversion", layout="wide")
st.title("Trial conversion model")
if st.button("Refresh"):
    st.cache_data.clear()

predictions = load_predictions()
drift = load_drift()
metrics = load_metrics()

# --- Headline: latest drift verdict and model metrics ---
left, mid, right = st.columns(3)
if not drift.empty:
    latest = drift.iloc[-1]
    verdict = "DRIFT" if latest["drift"] else "OK"
    left.metric(
        f"Drift verdict ({latest['day']:%Y-%m-%d})",
        verdict,
        f"{latest['drifted_share']:.0%} of features drifted",
        delta_color="inverse" if latest["drift"] else "off",
    )
else:
    left.metric("Drift verdict", "no data")
if metrics:
    mid.metric("Test AUC", f"{metrics['test_auc']:.3f}")
    right.metric("Train / test rows", f"{metrics['n_train']} / {metrics['n_test']}")
else:
    mid.metric("Test AUC", "no data")

# --- Latest ranked list ---
st.header("Latest ranked list")
if predictions:
    latest_day, latest_df = list(predictions.items())[-1]
    st.caption(
        f"Cohort {latest_day:%Y-%m-%d}, {len(latest_df)} trials. "
        "Lowest conversion probability first: these are most likely to cancel."
    )
    ranked = latest_df.sort_values("conversion_probability")
    st.dataframe(
        ranked,
        width="stretch",
        hide_index=True,
        column_config={
            "conversion_probability": st.column_config.ProgressColumn(
                "conversion_probability", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
else:
    st.info("No predictions in the bucket yet.")

# --- Past month ---
st.header(f"Past {DAYS} days")
cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=DAYS)

if predictions:
    daily = pd.DataFrame(
        [
            {
                "day": day,
                "trials": len(df),
                "mean_probability": df["conversion_probability"].mean(),
                "share_at_risk": (df["conversion_probability"] < 0.3).mean(),
            }
            for day, df in predictions.items()
            if day >= cutoff
        ]
    ).set_index("day")
    c1, c2 = st.columns(2)
    c1.subheader("Mean predicted conversion")
    c1.line_chart(daily["mean_probability"])
    c2.subheader("Share of trials below 30% (at risk)")
    c2.line_chart(daily["share_at_risk"])
    st.subheader("Trials scored per day")
    st.bar_chart(daily["trials"])

if not drift.empty:
    recent = drift[drift["day"] >= cutoff].set_index("day")
    st.subheader("Share of features drifted (alert at 50%)")
    st.line_chart(recent["drifted_share"])
    n_back = int(recent["backfilled"].sum())
    if n_back:
        st.caption(
            f"{n_back} of these days are backfilled placeholders, "
            "not real drift checks."
        )
    st.dataframe(
        recent.assign(verdict=recent["drift"].map({True: "DRIFT", False: "OK"}))[
            ["verdict", "drifted_share", "backfilled"]
        ].sort_index(ascending=False),
        width="stretch",
    )
