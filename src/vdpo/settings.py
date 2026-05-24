import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset
    dataset_path: str = "mgolov/Visual-Counterfact"

    # Model
    load_in_4bit: bool = True
    hf_token: str = ""

    # Outputs
    _output_dir_name: str = "outputs"

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / self._output_dir_name


settings = Settings()

if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token
