from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the ranking service."""

    ranker_manifest_path: Path = Path("/app/artifacts/models/ranker_manifest.json")
    cv_store_path: Path = Path("/app/artifacts/cv/cv_normalized.parquet")

    max_result_limit: int = 500

    model_config = SettingsConfigDict(
        env_prefix="RANKING_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached service settings."""
    return Settings()
