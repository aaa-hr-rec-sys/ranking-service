from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ranking_service.config import Settings
from ranking_service.ml.features import get_categorical_feature_columns


class ArtifactLoadError(RuntimeError):
    """Raised when a required ranking artifact cannot be loaded."""


@dataclass(frozen=True)
class RankerManifest:
    """Metadata describing the versioned ranker artifact."""

    path: Path
    payload: dict[str, Any]
    model_path: Path
    feature_columns_path: Path


@dataclass(frozen=True)
class FeatureColumns:
    """Feature columns expected by the CatBoost model."""

    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    feature_columns: list[str]


@dataclass(frozen=True)
class RankingArtifacts:
    """Loaded artifacts required for Stage 2 ranking."""

    model: Any
    feature_columns: FeatureColumns
    cv_store: pd.DataFrame
    manifest: RankerManifest


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"JSON artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(f"Invalid JSON artifact: {path}") from exc


def load_ranker_manifest(path: Path) -> RankerManifest:
    """Load ranker manifest and resolve artifact file paths."""
    payload = load_json(path)

    model_file = payload.get("model_file")
    feature_columns_file = payload.get("feature_columns_file")

    if not model_file:
        raise ArtifactLoadError("ranker manifest must contain model_file")
    if not feature_columns_file:
        raise ArtifactLoadError("ranker manifest must contain feature_columns_file")

    artifact_dir = path.parent

    return RankerManifest(
        path=path,
        payload=payload,
        model_path=artifact_dir / str(model_file),
        feature_columns_path=artifact_dir / str(feature_columns_file),
    )


def load_feature_columns(path: Path) -> FeatureColumns:
    """Load feature columns produced by the training pipeline."""
    payload = load_json(path)

    numeric_columns = list(payload.get("numeric_feature_columns", []))
    if not numeric_columns:
        raise ArtifactLoadError(
            "feature_columns.json must contain numeric_feature_columns"
        )

    categorical_columns = list(
        payload.get("categorical_feature_columns")
        or get_categorical_feature_columns()
    )

    feature_columns = list(
        payload.get("feature_columns")
        or numeric_columns + categorical_columns
    )

    return FeatureColumns(
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=categorical_columns,
        feature_columns=feature_columns,
    )


def load_cv_store(path: Path) -> pd.DataFrame:
    """Load normalized CV feature store."""
    try:
        cv_store = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"CV store is missing: {path}") from exc
    except Exception as exc:
        raise ArtifactLoadError(f"Could not read CV store: {path}") from exc

    if "cv_id_hash" not in cv_store.columns:
        raise ArtifactLoadError("CV store must contain column cv_id_hash")

    if cv_store["cv_id_hash"].isna().any():
        raise ArtifactLoadError("CV store contains empty cv_id_hash values")

    if cv_store["cv_id_hash"].duplicated().any():
        raise ArtifactLoadError("CV store contains duplicate cv_id_hash values")

    cv_store = cv_store.copy()
    cv_store["cv_id_hash"] = cv_store["cv_id_hash"].astype(str)

    return cv_store


def load_model(path: Path) -> Any:
    """Load CatBoost ranking model."""
    try:
        from catboost import CatBoostRanker
    except ModuleNotFoundError as exc:
        raise ArtifactLoadError("catboost is required to load CatBoost model") from exc

    if not path.exists():
        raise ArtifactLoadError(f"Model artifact is missing: {path}")

    model = CatBoostRanker()
    try:
        model.load_model(str(path))
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load CatBoost model: {path}") from exc

    return model


def artifact_status(settings: Settings) -> dict[str, Any]:
    """Return artifact path existence status for diagnostics."""
    status: dict[str, Any] = {
        "ranker_manifest_path": str(settings.ranker_manifest_path),
        "ranker_manifest_exists": settings.ranker_manifest_path.exists(),
        "cv_store_path": str(settings.cv_store_path),
        "cv_store_exists": settings.cv_store_path.exists(),
    }

    if not settings.ranker_manifest_path.exists():
        status["model_path"] = None
        status["model_exists"] = False
        status["feature_columns_path"] = None
        status["feature_columns_exists"] = False
        return status

    try:
        manifest = load_ranker_manifest(settings.ranker_manifest_path)
    except ArtifactLoadError as exc:
        status["manifest_error"] = str(exc)
        return status

    status.update(
        {
            "model_path": str(manifest.model_path),
            "model_exists": manifest.model_path.exists(),
            "feature_columns_path": str(manifest.feature_columns_path),
            "feature_columns_exists": manifest.feature_columns_path.exists(),
        }
    )
    return status


def load_artifacts(settings: Settings) -> RankingArtifacts:
    """Load all ranking artifacts."""
    manifest = load_ranker_manifest(settings.ranker_manifest_path)
    feature_columns = load_feature_columns(manifest.feature_columns_path)
    cv_store = load_cv_store(settings.cv_store_path)
    model = load_model(manifest.model_path)

    return RankingArtifacts(
        model=model,
        feature_columns=feature_columns,
        cv_store=cv_store,
        manifest=manifest,
    )
