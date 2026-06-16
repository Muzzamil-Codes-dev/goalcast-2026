"""Central configuration. Reads from environment / .env with safe offline defaults."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
KB_DIR = ROOT / "docs" / "knowledge_base"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///data/goalcast.db"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "goalcast-wdl"

    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"

    use_live_api: bool = False
    football_data_api_key: str = ""

    # Modelling knobs
    form_window: int = 5
    test_split_date: str = "2022-01-01"  # train < this, test >=

    def ensure_dirs(self) -> None:
        for d in (DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "features", ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
