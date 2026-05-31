"""OpenAlex 论文发现与摘要获取。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import Config

_logger = logging.getLogger("ideaforgex")
_BASE_URL = "https://api.openalex.org"
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """将 OpenAlex 的 abstract_inverted_index 还原为纯文本。"""
    if not inverted_index:
        return ""
    word_positions: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word
    return " ".join(word_positions[i] for i in sorted(word_positions))


def _extract_work_id(full_id: str) -> str:
    """从 OpenAlex 完整 URL 提取 work ID（如 W3167826310）。"""
    return full_id.rsplit("/", 1)[-1] if full_id else ""


@dataclass
class OpenAlexClient:
    config: Config

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{_BASE_URL}{path}"
        params = kwargs.pop("params", {})
        params["api_key"] = self.config.openalex_api_key
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = httpx.request(
                    method, url, params=params, timeout=30.0, **kwargs
                )
                if response.status_code == 429:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_DELAY * (2**attempt)
                        _logger.warning(
                            "OpenAlex 限流 (429)，第 %d/%d 次重试，等待 %.1fs",
                            attempt + 1,
                            _MAX_RETRIES,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < _MAX_RETRIES:
                    last_exc = exc
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    _logger.warning(
                        "OpenAlex 请求失败，第 %d/%d 次重试: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    time.sleep(_RETRY_DELAY * (2**attempt))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def search_papers(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """按标题搜索论文，返回 work 列表。"""
        response = self._request(
            "GET",
            "/works",
            params={"filter": f"title.search:{query}", "per_page": min(limit, 200)},
        )
        return response.json().get("results", [])

    def get_paper_detail(self, paper_id: str) -> dict[str, Any]:
        """获取单篇论文详情。

        paper_id 支持 OpenAlex work ID（如 W3167826310）或完整 URL。
        """
        work_id = _extract_work_id(paper_id)
        response = self._request("GET", f"/works/{work_id}")
        return response.json()
