from __future__ import annotations

from fastapi import FastAPI

from ranking_service.schemas import (
    CandidateDisplay,
    RankedCandidate,
    Stage2Request,
    Stage2Response,
)

app = FastAPI(
    title="Ranking Service",
    description="Stage 2 service for reranking Stage 1 candidates.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/rank", response_model=Stage2Response)
def rank(request: Stage2Request) -> Stage2Response:
    """Rank Stage 1 candidates using the current scoring implementation."""
    sorted_candidates = sorted(
        request.candidates,
        key=lambda candidate: (
            candidate.embedding_rank,
            -candidate.embedding_score,
            candidate.cv_id_hash,
        ),
    )

    ranked: list[RankedCandidate] = []
    for rank_position, candidate in enumerate(
        sorted_candidates[: request.result_limit],
        start=1,
    ):
        ranked.append(
            RankedCandidate(
                cv_id_hash=candidate.cv_id_hash,
                rank=rank_position,
                model_score=float(candidate.embedding_score),
                embedding_score=float(candidate.embedding_score),
                embedding_rank=int(candidate.embedding_rank),
                display=CandidateDisplay(),
            )
        )

    return Stage2Response(
        job_id=request.job_id,
        ranked=ranked,
    )
