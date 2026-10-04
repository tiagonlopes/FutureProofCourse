import sys

from trial_conversion_model.features import build_training_data
from trial_conversion_model.publish_model import publish
from trial_conversion_model.train import train

if __name__ == "__main__":
    build_training_data()
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    print(train(run_name=run_name))
    print(f"published to {publish()}")
