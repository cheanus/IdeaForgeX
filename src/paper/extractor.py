"""arXiv 与本地 PDF 提取。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import httpx

from src.config import Config

_ARXIV_PDF_BASE = "https://arxiv.org"


@dataclass
class ArxivExtractor:
    config: Config

    def get_paper_detail(self, arxiv_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"https://arxiv.org/abs/{arxiv_id}",
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.text
        title = re.search(r"Title:</span>(.*?)</h1>", content, re.DOTALL)
        title = title.group(1).strip() if title else arxiv_id
        abstract = re.search(r"Abstract:</span>(.*?)</blockquote>", content, re.DOTALL)
        abstract = abstract.group(1).strip() if abstract else ""
        year = "20" + arxiv_id[:2] if re.match(r"^\d{4}\.\d{4,5}", arxiv_id) else ""
        return {
            "title": title,
            "abstract_slice": abstract[:500],
            "abstract": abstract,
            "year": year,
        }

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
        response = httpx.get(f"{_ARXIV_PDF_BASE}/pdf/{arxiv_id}.pdf", timeout=60.0)
        if response.status_code in {301, 302, 307, 308} and response.headers.get(
            "location"
        ):
            location = response.headers["location"]
            next_url = (
                location
                if location.startswith("http")
                else f"{_ARXIV_PDF_BASE}{location}"
            )
            response = httpx.get(next_url, timeout=60.0)
        response.raise_for_status()

        document = fitz.open(stream=response.content, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in document)  # type: ignore[reportCallIssue,reportArgumentType]
        finally:
            document.close()


def extract_local_pdf_text(pdf_path: str | Path) -> str:
    document = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text() for page in document)  # type: ignore[reportCallIssue,reportArgumentType]
    finally:
        document.close()
