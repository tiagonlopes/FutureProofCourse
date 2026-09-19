"""
Entry point to pull the latest trial snapshot from the database.

Run with: uv run scripts/fetch.py
"""

from trial_conversion_model.fetch_data import fetch_trials_raw

if __name__ == "__main__":
    fetch_trials_raw()
