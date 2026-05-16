"""应用配置。"""

from __future__ import annotations

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
    temperature_llm_a_judge: float = Field(default=0.1)
    llm_json_mode: bool = Field(default=True)
    llm_thinking_enabled: bool = Field(default=False)
    max_retries: int = Field(default=3)
    aminer_api_key: str = Field(default="")
    aminer_base_url: str = Field(
        default="https://datacenter.aminer.cn/gateway/open_platform"
    )
    arxiv_api_url: str = Field(default="https://export.arxiv.org/api/query")
    neo4j_uri: str = Field(default="bolt://localhost:7688")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")
    k_hits: int = Field(default=10)
    max_neighbors: int = Field(default=3)
    max_depth: int = Field(default=2)
    score_decay: float = Field(default=0.7)
    final_k: int = Field(default=20)
    arxiv_short_abstract_threshold: int = Field(default=200)
    log_level: str = Field(default="WARNING")


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
    import os

    env_log_level = os.environ.get("LOG_LEVEL")
    if env_log_level is not None:
        yaml_data["log_level"] = env_log_level

    return Config(config_file=path, **yaml_data)  # type: ignore[reportArgumentType]
