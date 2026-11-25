from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Holds all app settings and lets us override them with environment variables or a .env file
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Basic app info and server host/port
    app_name: str = "Model Serving Platform"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Where we save the SQLite database and uploaded model files
    database_url: str = "sqlite:///./data/registry.db"
    artifact_storage_dir: Path = Path("./data/artifacts")

    # Safety limits for uploads, memory, queues, batches, and timeouts
    max_upload_size_mb: int = 100
    max_loaded_models: int = 10
    max_queue_depth_per_deployment: int = 64
    max_batch_size: int = 256
    inference_timeout_seconds: float = 30.0
    worker_startup_timeout_seconds: float = 15.0
    enable_custom_python_models: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        # Convert the upload limit from megabytes to bytes
        return self.max_upload_size_mb * 1024 * 1024
