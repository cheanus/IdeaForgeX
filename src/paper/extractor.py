"""arXiv 与本地 PDF 提取。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.config import Config


@dataclass
class ArxivExtractor:
    config: Config

    def find_arxiv_id(self, paper: dict[str, Any]) -> str | None:
        title = paper.get("title", "")
        response = httpx.get(
            self.config.arxiv_api_url,
            params={"search_query": f"ti:{title}", "max_results": 1},
            timeout=30.0,
        )
        response.raise_for_status()
        match = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", response.text)
        return match.group(1) if match else None

    def fetch_full_text(self, arxiv_id: str) -> str:
        response = httpx.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60.0)
        response.raise_for_status()
        return response.text


def extract_local_pdf_text(pdf_path: str | Path) -> str:
    import fitz  # pymupdf

    document = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()
