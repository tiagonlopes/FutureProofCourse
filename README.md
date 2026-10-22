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
- `S3_BUCKET`: the bucket the model, daily predictions, drift reports and the drift reference are stored in.
- `SLACK_WEBHOOK_URL`: where drift alerts are posted (optional; without it alerts are only printed).

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

`POST /predict` takes one trial's first-3-day base aggregates and returns its conversion probability plus a low/medium/high band (and appends each prediction to `logs/predictions.jsonl` and the terminal log); `GET /health` reports service status, the package version, and the commit the image was built from (`GIT_SHA`). Interactive docs live at `/docs` while the service runs.

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
{"probability":0.8593,"band":"high"}
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
{"probability":0.3385,"band":"medium"}
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

## Scheduled jobs

Two short-lived jobs run daily from a separate image (`Dockerfile.jobs`, published as `trial-conversion-model-jobs`). It carries code and the model, never data; data arrives from S3 at run time. On the server, cron starts one container per job (see the comments in `Dockerfile.jobs` for the exact `docker run` command).

```
uv run scripts/score.py
uv run scripts/check_drift.py [cohort file]
```

- `scripts/score.py` pulls the newest day-3 cohort from the public course bucket (`fpds-trial-cohorts`, no credentials needed), scores every trial with the same `predict_proba` the API uses, writes `data/04_predictions/scored_<date>.csv` (lowest conversion probability first, i.e. most likely to cancel at the top) and uploads it to `s3://$S3_BUCKET/predictions/`. This is the lifecycle team's daily ranked list.
- `scripts/check_drift.py` compares the newest local cohort (or the file you pass) against the training extract, feature by feature, using Evidently. The dataset counts as drifted when at least half of the features have drifted. It writes an HTML report and a JSON verdict to `monitoring/`, uploads both to `s3://$S3_BUCKET/monitoring/`, and posts to Slack when drift is detected. Run it after `score.py`, which pulls the cohort. The reference extract is fetched from `s3://$S3_BUCKET/data/trial_snapshot.csv` on first run if it is not on disk.
- `scripts/backfill.py` is a one-time fill, never scheduled: it writes 30 days of scored cohorts and placeholder drift verdicts to the bucket so the dashboard has history. Backfilled drift entries carry `"source": "backfill"` and are flagged in the dashboard.

## Dashboard

A read-only Streamlit dashboard over what the jobs publish to the bucket:

```
uv run streamlit run dashboard.py
```

It needs `S3_BUCKET` and AWS credentials with read access. It shows the latest drift verdict and the model's test AUC (from `models/metrics.json`, falling back to the bucket), the latest ranked list of trials, and the past 30 days of mean predicted conversion, share of trials below 30% (at risk), trials scored per day, and share of features drifted.

## CI/CD

`.github/workflows/ci.yml` runs on every pull request and push to `main`:

1. **ci**: installs dependencies, lints with `ruff`, pulls the published model from S3, runs `pytest`, and builds the service image.
2. **publish** (main only): builds and pushes both images (API and jobs) to GitHub Container Registry, tagged `latest` and with the commit SHA.
3. **deploy** (main only): over SSH, pulls both images on the EC2 server and restarts the API container. The jobs image is picked up by the next cron run.

Required repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `EC2_HOST`, `EC2_SSH_KEY`.

## Tests

```
uv run pytest
```

`tests/test_api.py` covers `/health`, `/predict` output, and rejection of requests missing a field; `tests/test_features.py` covers feature derivation. The API loads the model at import, so `models/model.json` must exist (pull it from S3 as CI does).

## Layout

- `src/trial_conversion_model/`: the package. `data.py` acquires the extract from the database and loads the pipeline's inputs; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, and saves the model; `predict.py` scores trials from their base aggregates; `api/` is the FastAPI service (`main.py` builds the app, `routes.py` holds the endpoints, `schemas.py` defines the request and response shapes); `check_s3.py` verifies the S3 credentials and `publish_model.py` uploads the trained model to the bucket; `cohorts.py` pulls and locates the daily day-3 cohort files; `monitoring.py` runs the drift check and publishes its reports.
- `scripts/`: thin entry points that call into the package (`fetch_data.py` materializes the extract, `train.py` builds the training table and trains, `score.py` scores the daily cohort, `check_drift.py` runs the drift check, `backfill.py` is the one-time history fill). Production runs these; the logic stays importable and testable in `src/`.
- `dashboard.py`: the Streamlit dashboard (see Dashboard).
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the immutable extract as pulled from the database (never committed, never modified).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `data/01_raw/cohorts/`: daily day-3 cohort files pulled by the scoring job (never committed).
- `data/04_predictions/`: the daily scored, ranked lists written by `score.py` (never committed).
- `monitoring/`: drift reports (HTML and JSON), regenerated by every check (git-ignored).
- `logs/`: `predictions.jsonl`, written by the running service (git-ignored).
- `Dockerfile` / `Dockerfile.jobs`: the API image and the scheduled-jobs image.
- `models/`: trained model artifacts and metrics (not committed).
