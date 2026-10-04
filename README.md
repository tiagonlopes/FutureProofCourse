# trial-conversion-model

Predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14, so the growth team can reach trials that look unlikely to convert while they are still live. The business case and rollout plan are in the model plan document.

## Setup

Install the dependencies and the package:

```
uv sync
```

The training data is not committed to the repository. Copy `.env.example` to `.env` and fill in the values (`.env` is git-ignored, so credentials stay local):

- `DB_*`: the database connection, including the password from the course's Tools & Setup lesson.
- `MLFLOW_TRACKING_URI`: where training runs are logged (defaults to `http://127.0.0.1:5001` if unset).
- `AWS_*`: credentials for the S3 bucket the model is published to.

Then materialize the extract:

```
uv run scripts/fetch_data.py
```

This writes `data/01_raw/trial_snapshot.csv`. Training always runs from that file, never from the live table, so the training data cannot shift between runs.

## Train

```
uv run scripts/train.py
```

This builds the processed training table from the raw extract, trains the model, writes `models/model.json` and `models/metrics.json`, and then publishes the model to S3 (see Publish). Each run is also logged to MLflow (experiment `trial-conversion-model`), so an MLflow server must be reachable at `MLFLOW_TRACKING_URI`. Local MLflow output (`mlartifacts/`, `mlflow.db`) is git-ignored.

## Publish

`models/` is not committed, so the bucket holds the only copy of the model that anything outside your machine can reach. Check that your credentials work, then upload the trained model:

```
uv run src/trial_conversion_model/check_s3.py
uv run src/trial_conversion_model/publish_model.py
```

This uploads `models/model.json` to `s3://trial-conversion-artifacts-tiago/models/model.json` and verifies the object's size matches the local file. The key is fixed: downstream pipelines read the model by that exact name. Run it from the project root, since the local path is relative, and note it overwrites any object already at that key.

## Serve

The model is served over HTTP so the teams that need scores can get them on demand. Run the service locally:

```
uv run uvicorn trial_conversion_model.api.main:app --reload
```

Or build and run it as a container, which is how it ships:

```
docker build -t trial-conversion-model .
docker run -p 8000:8000 trial-conversion-model
```

`POST /predict` takes one trial's first-3-day base aggregates and returns its conversion probability plus a low/medium/high band; `GET /health` reports service status. Interactive docs live at `/docs` while the service runs.

### Checking it works

With the service running, three requests and what each should come back with.

A steady trial, spread across the first three days:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 3, "sessions_day2": 2, "sessions_day3": 2,
  "listen_sessions_3d": 5, "total_minutes_3d": 180,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.8593,"conversion_band":"high"}
```

The same trial's activity crammed into day one:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 9, "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.3385,"conversion_band":"medium"}
```

A request with `sessions_day1` missing, which the contract turns down before any
of your code runs:

```
curl -i -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
HTTP/1.1 422 Unprocessable Entity
```

## Layout

- `src/trial_conversion_model/`: the package. `data.py` acquires the extract from the database and loads the pipeline's inputs; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, and saves the model; `predict.py` scores trials from their base aggregates; `api/` is the FastAPI service (`main.py` builds the app, `routes.py` holds the endpoints, `schemas.py` defines the request and response shapes); `check_s3.py` verifies the S3 credentials and `publish_model.py` uploads the trained model to the bucket.
- `scripts/`: thin entry points that call into the package (`fetch_data.py` materializes the extract, `train.py` builds the training table and trains). Production runs these; the logic stays importable and testable in `src/`.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the immutable extract as pulled from the database (never committed, never modified).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `models/`: trained model artifacts and metrics (not committed).
