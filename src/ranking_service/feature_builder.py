from __future__ import annotations

import pandas as pd

from ranking_service.ml.features import add_pair_features
from ranking_service.ml.normalization import normalize_vacancies
from ranking_service.schemas import Stage2Request


CANDIDATE_COLUMNS = [
    "vacancy_id_hash",
    "cv_id_hash",
    "embedding_score",
    "embedding_rank",
]


def build_vacancy_frame(request: Stage2Request) -> pd.DataFrame:
    """Build a one-row raw vacancy frame from the Stage 2 request."""
    vacancy = request.vacancy

    return pd.DataFrame(
        [
            {
                "vacancy_id_hash": request.job_id,
                "vacancy_text": vacancy.vacancy_text,
                "profession": vacancy.profession,
                "group_profession": vacancy.group_profession,
                "business_category": vacancy.business_category,
                "sfera": vacancy.sfera,
                "experience": vacancy.experience,
                "schedule": vacancy.schedule,
                "employment_type": vacancy.employment_type,
                "education_level": vacancy.education_level,
            }
        ]
    )


def build_candidates_frame(request: Stage2Request) -> pd.DataFrame:
    """Build candidate pairs from Stage 1 retrieval output."""
    rows = [
        {
            "vacancy_id_hash": request.job_id,
            "cv_id_hash": candidate.cv_id_hash,
            "embedding_score": candidate.embedding_score,
            "embedding_rank": candidate.embedding_rank,
        }
        for candidate in request.candidates
    ]

    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def select_cv_rows(
    cv_store: pd.DataFrame,
    candidate_ids: list[str],
) -> pd.DataFrame:
    """Select normalized CV rows required for the current request."""
    if "cv_id_hash" not in cv_store.columns:
        raise KeyError("cv_store must contain column cv_id_hash")

    candidate_id_set = {str(candidate_id) for candidate_id in candidate_ids}

    cv_rows = cv_store[
        cv_store["cv_id_hash"].astype(str).isin(candidate_id_set)
    ].copy()

    found_ids = set(cv_rows["cv_id_hash"].astype(str))
    missing_ids = sorted(candidate_id_set - found_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"cv_store is missing {len(missing_ids)} candidate ids: {preview}"
        )

    return cv_rows


def build_feature_frame(
    request: Stage2Request,
    cv_store: pd.DataFrame,
) -> pd.DataFrame:
    """Build CatBoost feature frame for one vacancy and Stage 1 candidates."""
    candidates = build_candidates_frame(request)
    if candidates.empty:
        return candidates

    vacancy_raw = build_vacancy_frame(request)
    vacancy_norm = normalize_vacancies(vacancy_raw)

    cv_rows = select_cv_rows(
        cv_store=cv_store,
        candidate_ids=candidates["cv_id_hash"].astype(str).tolist(),
    )

    return add_pair_features(
        candidates=candidates,
        cv_norm=cv_rows,
        vacancies_norm=vacancy_norm,
    )
