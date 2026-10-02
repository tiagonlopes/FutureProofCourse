import os

import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = "trial-conversion-artifacts-tiago"
KEY = "models/model.json"  # exact key, next week's pipeline reads it by name
LOCAL_PATH = "models/model.json"

s3 = boto3.client("s3")
s3.upload_file(LOCAL_PATH, BUCKET, KEY)
print(f"uploaded to s3://{BUCKET}/{KEY}")

head = s3.head_object(Bucket=BUCKET, Key=KEY)
local_size = os.path.getsize(LOCAL_PATH)
assert head["ContentLength"] == local_size, (head["ContentLength"], local_size)
print(f"verified: {head['ContentLength']} bytes in bucket == {local_size} bytes local")
