import os

import boto3
from dotenv import load_dotenv

BUCKET = "trial-conversion-artifacts-tiago"
KEY = "models/model.json"
LOCAL_PATH = "models/model.json"


def publish(local_path: str = LOCAL_PATH) -> str:
    """Upload the trained model to S3 and verify the object size matches."""
    load_dotenv()
    s3 = boto3.client("s3")
    s3.upload_file(local_path, BUCKET, KEY)

    head = s3.head_object(Bucket=BUCKET, Key=KEY)
    local_size = os.path.getsize(local_path)
    assert head["ContentLength"] == local_size, (head["ContentLength"], local_size)
    return f"s3://{BUCKET}/{KEY}"


if __name__ == "__main__":
    print(f"uploaded to {publish()}")
