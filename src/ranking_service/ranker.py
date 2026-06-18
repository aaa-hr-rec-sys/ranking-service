from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import Pool

from ranking_service.artifacts import RankingArtifacts
from ranking_service.feature_builder import build_feature_frame
from ranking_service.schemas import (
    CandidateDisplay,
    RankedCandidate,
    Stage2Request,
    Stage2Response,
)


class RankerError(RuntimeError):
    """Raised when candidate ranking fails."""


def _optional_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value

    return value


def _first_present(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row.index:
            value = _optional_value(row[column])
            if value is not None:
                return value
    return None


def _optional_float(value: Any) -> float | None:
    value = _optional_value(value)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_display(row: pd.Series) -> CandidateDisplay:
    """Build optional display fields from normalized CV columns."""
    return CandidateDisplay(
        profession=_first_present(
            row,
            ["cv_profession_norm", "profession_norm", "profession"],
        ),
        group_profession=_first_present(
            row,
            ["cv_group_profession_norm", "group_profession_norm", "group_profession"],
        ),
        business_category=_first_present(
            row,
            [
                "cv_business_category_norm",
                "business_category_norm",
                "business_category",
            ],
        ),
        sfera=_first_present(
            row,
            ["cv_sfera_norm", "sfera_norm", "sfera"],
        ),
        experience_bucket=_first_present(
            row,
            ["cv_experience_common", "experience_common", "experience"],
        ),
        education=_first_present(
            row,
            ["cv_education_norm", "education_norm", "education"],
        ),
        federal_district=_first_present(
            row,
            ["cv_federal_district", "federal_district"],
        ),
        salary_bucketed=_optional_float(
            _first_present(
                row,
                ["cv_salary_bucketed", "salary_bucketed", "salary"],
            )
        ),
        employment_type=_first_present(
            row,
            ["cv_employment_type_norm", "employment_type_norm", "employment_type"],
        ),
        schedule=_first_present(
            row,
            ["cv_schedule_norm", "schedule_norm", "schedule"],
        ),
    )


def prepare_model_frame(
    feature_frame: pd.DataFrame,
    artifacts: RankingArtifacts,
) -> pd.DataFrame:
    """Select and type model features for CatBoost inference."""
    feature_columns = artifacts.feature_columns.feature_columns

    missing_columns = [
        column for column in feature_columns if column not in feature_frame.columns
    ]
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        raise RankerError(
            f"Feature frame is missing {len(missing_columns)} model columns: {preview}"
        )

    model_frame = feature_frame[feature_columns].copy()

    for column in artifacts.feature_columns.numeric_feature_columns:
        if column in model_frame.columns:
            model_frame[column] = pd.to_numeric(
                model_frame[column],
                errors="coerce",
            ).fillna(0.0)

    for column in artifacts.feature_columns.categorical_feature_columns:
        if column in model_frame.columns:
            model_frame[column] = model_frame[column].fillna("unknown").astype(str)

    return model_frame


def predict_scores(
    model_frame: pd.DataFrame,
    artifacts: RankingArtifacts,
) -> list[float]:
    """Predict CatBoost ranking scores."""
    feature_columns = artifacts.feature_columns.feature_columns
    categorical_features = [
        feature_columns.index(column)
        for column in artifacts.feature_columns.categorical_feature_columns
        if column in feature_columns
    ]

    pool = Pool(
        data=model_frame,
        cat_features=categorical_features,
    )

    try:
        scores = artifacts.model.predict(pool)
    except Exception as exc:
        raise RankerError("CatBoost prediction failed") from exc

    return [float(score) for score in scores]


def rank_candidates(
    request: Stage2Request,
    artifacts: RankingArtifacts,
) -> Stage2Response:
    """Rank Stage 1 candidates with the loaded CatBoost model."""
    if not request.candidates:
        return Stage2Response(job_id=request.job_id, ranked=[])

    try:
        feature_frame = build_feature_frame(
            request=request,
            cv_store=artifacts.cv_store,
        )
    except Exception as exc:
        raise RankerError("Could not build feature frame") from exc

    if feature_frame.empty:
        return Stage2Response(job_id=request.job_id, ranked=[])

    model_frame = prepare_model_frame(
        feature_frame=feature_frame,
        artifacts=artifacts,
    )

    scores = predict_scores(
        model_frame=model_frame,
        artifacts=artifacts,
    )

    ranked_frame = feature_frame.copy()
    ranked_frame["model_score"] = scores
    ranked_frame["cv_id_hash"] = ranked_frame["cv_id_hash"].astype(str)

    ranked_frame = ranked_frame.sort_values(
        by=["model_score", "embedding_score", "embedding_rank", "cv_id_hash"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )

    ranked: list[RankedCandidate] = []
    for rank_position, (_, row) in enumerate(
        ranked_frame.head(request.result_limit).iterrows(),
        start=1,
    ):
        ranked.append(
            RankedCandidate(
                cv_id_hash=str(row["cv_id_hash"]),
                rank=rank_position,
                model_score=float(row["model_score"]),
                embedding_score=float(row["embedding_score"]),
                embedding_rank=int(row["embedding_rank"]),
                display=build_display(row),
            )
        )

    return Stage2Response(job_id=request.job_id, ranked=ranked)
