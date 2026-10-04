import pandas as pd

from trial_conversion_model.features import add_features


def test_zero_session_trial_gets_zero_shares_not_nan():
    # TODO: write this one yourself, in three steps:
    #   1. Build a one-row DataFrame for a trial with nothing in it: zero
    #      sessions on each of the three days, zero listen sessions, and
    #      zero total minutes.
    #   2. Pass it through add_features.
    #   3. Assert that day1_share, listen_share and avg_session_minutes each
    #      come back as 0 rather than a missing value.
    # raise NotImplementedError
    df = pd.DataFrame(
        {
            "sessions_day1": [0],
            "sessions_day2": [0],
            "sessions_day3": [0],
            "listen_sessions_3d": [0],
            "total_minutes_3d": [0.0],
        }
    )
    result = add_features(df)
    row = result.iloc[0]
    for column in ["day1_share", "listen_share", "avg_session_minutes"]:
        assert not pd.isna(row[column]), f"{column} is NaN"
        assert row[column] == 0
