from __future__ import annotations

from pydantic import BaseModel, Field


class VacancyFields(BaseModel):
    vacancy_text: str | None = None
    profession: str | None = None
    group_profession: str | None = None
    business_category: str | None = None
    sfera: str | None = None
    experience: str | None = None
    schedule: str | None = None
    employment_type: str | None = None
    education_level: str | None = None


class Stage1Candidate(BaseModel):
    cv_id_hash: str
    embedding_score: float
    embedding_rank: int


class Stage2Request(BaseModel):
    job_id: str
    vacancy: VacancyFields
    candidates: list[Stage1Candidate]
    result_limit: int = Field(default=500, ge=1, le=500)


class RankedCandidate(BaseModel):
    cv_id_hash: str
    rank: int
    model_score: float


class Stage2Response(BaseModel):
    job_id: str
    ranked: list[RankedCandidate]
