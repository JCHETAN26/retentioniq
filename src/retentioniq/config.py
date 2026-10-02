"""Runtime settings, loaded from environment variables / `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RIQ_", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    database_url: str = "postgresql+psycopg://retentioniq:retentioniq@localhost:5432/retentioniq"
    mlflow_tracking_uri: str = "http://localhost:5000"
    s3_bucket: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


settings = Settings()
