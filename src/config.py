"""应用配置。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """统一配置来源：yaml + 环境变量。"""

    model_config = SettingsConfigDict(
        env_prefix="IDEAFORGEX_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    config_file: Path = Field(default=Path("config.yaml"))
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_api_key: str = Field(default="")
    llm_model_name: str = Field(default="gpt-4o-mini")
    embedding_base_url: str = Field(default="http://localhost:8000/v1")
    embedding_api_key: str = Field(default="")
    embedding_model_name: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=1536)
    llm_temperature: float = Field(default=1.0)
    llm_json_mode: bool = Field(default=True)
    llm_thinking_enabled: bool = Field(default=False)
    max_retries: int = Field(default=3)
    openalex_api_key: str = Field(default="")
    arxiv_api_url: str = Field(default="https://export.arxiv.org/api/query")
    neo4j_uri: str = Field(default="bolt://localhost:7688")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")
    k_hits: int = Field(default=8)
    max_neighbors: int = Field(default=2)
    max_depth: int = Field(default=1)
    score_decay: float = Field(default=0.5)
    final_k: int = Field(default=15)
    batch_concurrency: int = Field(default=4)
    compact_interval: int = Field(default=10)
    compact_threshold: float = Field(default=0.95)
    short_abstract_threshold: int = Field(default=200)
    log_level: str = Field(default="INFO")
    cypher_version: int = Field(default=5)
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=2048)
    server_root_path: str = Field(default="")
    server_cors_origins: list[str] = Field(default_factory=lambda: ["*"])


def load_config(config_file: str | Path | None = None) -> Config:
    """加载配置，允许显式指定 yaml 文件，LOG_LEVEL 环境变量优先于 yaml。"""

    path = Path(config_file or "config.yaml")
    yaml_data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                yaml_data = loaded

    # LOG_LEVEL 环境变量覆盖 yaml 中的 log_level
    env_log_level = os.environ.get("LOG_LEVEL")
    if env_log_level is not None:
        yaml_data["log_level"] = env_log_level

    return Config(config_file=path, **yaml_data)  # type: ignore[reportArgumentType]
