# ranking-service

Stage 2 ranking service for the HR resume recommendation system.

The service receives vacancy fields and Stage 1 retrieval candidates from the backend orchestrator, prepares the Stage 2 response contract, and returns candidates in ranked order.

## Current status

This is the initial FastAPI skeleton for the Stage 2 service.

Implemented:

- `GET /health`
- `GET /ready`
- `POST /rank`
- Pydantic request/response schemas
- Dockerfile

Current ranking behavior:

- candidates are ranked deterministically by `embedding_rank`, then by `embedding_score`
- `model_score` is temporarily set to `embedding_score`

Not implemented yet:

- loading CatBoost model
- loading feature schema
- loading normalized CV feature store
- vacancy normalization
- pair feature construction
- CatBoost scoring

## Service role

The final Stage 2 service is responsible for:

1. receiving candidates from Stage 1 retrieval;
2. loading normalized CV features by `cv_id_hash`;
3. normalizing incoming vacancy fields;
4. building CatBoost-compatible pair features;
5. scoring candidates with CatBoost;
6. returning ranked candidates to the backend orchestrator.

The service does not calculate embeddings and does not run Stage 1 retrieval.

## API

### `GET /health`

Returns process liveness.

Response:

```json
{
  "status": "ok"
}
```

### `GET /ready`

Returns service readiness.

Response:

```json
{
  "status": "ready"
}
```

### `POST /rank`

Accepts vacancy fields and Stage 1 candidates.

Request shape:

```json
{
  "job_id": "job-1",
  "vacancy": {
    "vacancy_text": "Python backend developer",
    "profession": "developer",
    "group_profession": "IT",
    "business_category": "software",
    "sfera": "IT",
    "experience": "более 1 года",
    "schedule": "фиксированный",
    "employment_type": "полная занятость",
    "education_level": "не имеет значения"
  },
  "candidates": [
    {
      "cv_id_hash": "cv-1",
      "embedding_score": 0.91,
      "embedding_rank": 1
    },
    {
      "cv_id_hash": "cv-2",
      "embedding_score": 0.84,
      "embedding_rank": 2
    }
  ],
  "result_limit": 500
}
```

Response shape:

```json
{
  "job_id": "job-1",
  "ranked": [
    {
      "cv_id_hash": "cv-1",
      "rank": 1,
      "model_score": 0.91,
      "embedding_score": 0.91,
      "embedding_rank": 1,
      "display": {
        "profession": null,
        "group_profession": null,
        "business_category": null,
        "sfera": null,
        "experience_bucket": null,
        "education": null,
        "federal_district": null,
        "salary_bucketed": null,
        "employment_type": null,
        "schedule": null
      }
    }
  ]
}
```

## Artifacts

The final service is expected to use the following Stage 2 artifacts:

```text
/app/artifacts/model.cbm
/app/artifacts/feature_schema.json
/app/artifacts/cv_store.parquet
```

Artifact policy:

- `model.cbm` is a versioned model artifact and can be committed for MVP/demo usage.
- `feature_schema.json` is a versioned inference contract and should be stored together with the model.
- `cv_store.parquet` contains normalized CV data and should be provided through an external Docker volume.
- raw CV data, embeddings, parquet datasets, and NumPy matrices should not be committed.

## Planned inference flow

```text
Stage2Request
  -> validate request
  -> load candidates from request
  -> load normalized CV features by cv_id_hash
  -> normalize vacancy fields
  -> build pair features
  -> create CatBoost Pool
  -> predict model_score
  -> sort candidates by model_score
  -> return Stage2Response
```

## Local run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run service on port `8002` on Windows:

```bash
set PYTHONPATH=src
python -m uvicorn ranking_service.app:app --host 127.0.0.1 --port 8002 --reload
```

Run service on port `8002` on Linux/macOS:

```bash
export PYTHONPATH=src
python -m uvicorn ranking_service.app:app --host 127.0.0.1 --port 8002 --reload
```

Open healthcheck:

```text
http://127.0.0.1:8002/health
```

Open Swagger UI:

```text
http://127.0.0.1:8002/docs
```

## Docker

Build image:

```bash
docker build -t ranking-service .
```

Run container:

```bash
docker run --rm -p 8002:8000 ranking-service
```

Check health:

```text
http://127.0.0.1:8002/health
```

## Backend integration

The backend orchestrator should call this service through `STAGE2_URL`.

Example:

```text
STAGE2_URL=http://ranking-service:8000/rank
```

The backend sends:

- vacancy fields;
- Stage 1 candidates;
- `embedding_score`;
- `embedding_rank`;
- `result_limit`.

The ranking service returns:

- ranked CV ids;
- final rank;
- `model_score`;
- original Stage 1 score and rank;
- optional display fields for frontend cards.
