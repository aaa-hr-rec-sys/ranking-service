from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from ranking_service.artifacts import (
    RankingArtifacts,
    artifact_status,
    load_artifacts,
)
from ranking_service.config import get_settings
from ranking_service.ranker import RankerError, rank_candidates
from ranking_service.schemas import Stage2Request, Stage2Response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ranking artifacts on application startup."""
    settings = get_settings()

    app.state.settings = settings
    app.state.artifacts = None
    app.state.artifact_error = None

    try:
        app.state.artifacts = load_artifacts(settings)
    except Exception as exc:
        app.state.artifact_error = str(exc)

    yield


app = FastAPI(
    title="Ranking Service",
    description="Stage 2 service for reranking Stage 1 candidates.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_loaded_artifacts(request: Request) -> RankingArtifacts:
    """Return loaded artifacts or raise a readiness error."""
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is not None:
        return artifacts

    settings = getattr(request.app.state, "settings", get_settings())
    artifact_error = getattr(request.app.state, "artifact_error", None)

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "status": "not_ready",
            "error": artifact_error,
            "artifacts": artifact_status(settings),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    settings = getattr(request.app.state, "settings", get_settings())
    artifacts = getattr(request.app.state, "artifacts", None)

    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": getattr(request.app.state, "artifact_error", None),
                "artifacts": artifact_status(settings),
            },
        )

    return {
        "status": "ready",
        "artifacts": artifact_status(settings),
    }


@app.post("/rank", response_model=Stage2Response)
def rank(
    rank_request: Stage2Request,
    request: Request,
) -> Stage2Response:
    artifacts = get_loaded_artifacts(request)

    try:
        return rank_candidates(
            request=rank_request,
            artifacts=artifacts,
        )
    except RankerError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
