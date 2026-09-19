"""
Entry point to run the full pipeline: clean the raw data, engineer features,
and train the model.

Run with: uv run scripts/train.py
"""

from trial_conversion_model.data_cleaning import load_and_clean
from trial_conversion_model.data_engineering import load_and_add_features
from trial_conversion_model.train_model import train_and_save

if __name__ == "__main__":
    load_and_clean()
    load_and_add_features()
    train_and_save()
