"""AMiner 论文发现与摘要获取。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from src.config import Config


@dataclass
class AMinerClient:
    config: Config

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": self.config.aminer_api_key,
            "X-Platform": "openclaw",
            "Content-Type": "application/json;charset=utf-8",
        }

    @property
    def base_url(self) -> str:
        return self.config.aminer_base_url.rstrip("/")

    def search_papers(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        response = httpx.post(
            f"{self.base_url}/api/paper/qa/search",
            headers=self.headers,
            json={"use_topic": False, "query": query, "size": limit, "sci_flag": True},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def get_abstracts_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        response = httpx.post(
            f"{self.base_url}/api/paper/info",
            headers=self.headers,
            json={"ids": ids},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def get_paper_detail(self, paper_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/api/paper/detail",
            headers=self.headers,
            params={"id": paper_id},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("data", {})
