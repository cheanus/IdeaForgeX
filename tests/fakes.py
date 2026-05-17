"""测试共享 Fakes 和常量。"""

from __future__ import annotations

import httpx

from src.config import Config

ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions."
)
TEST_PAPER_ID = "paper-1706.03762"
TEST_ARXIV_ID = "1706.03762"


class FakeAMinerClient:
    """返回固定论文数据的 AMiner 客户端。"""

    def __init__(self, config: Config, *, should_fail: bool = False):
        self.config = config
        self.should_fail = should_fail

    def search_papers(self, query: str, limit: int = 50):
        """模拟语义搜索，返回匹配的论文列表。"""
        return [{"id": TEST_ARXIV_ID, "title": query}]

    def get_paper_detail(self, paper_id: str):
        if self.should_fail:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(
                    500, request=httpx.Request("GET", "https://example.test")
                ),
            )
        return {
            "id": paper_id,
            "title": ATTENTION_TITLE,
            "abstract_slice": "short abstract",
        }


class FakeArxivExtractor:
    """返回固定全文的 arXiv 提取器。"""

    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, arxiv_id: str):
        return {
            "title": ATTENTION_TITLE,
            "abstract": ATTENTION_ABSTRACT[:500],
            "year": "2017",
        }

    def find_arxiv_id(self, paper: dict):
        return TEST_ARXIV_ID

    def fetch_full_text(self, arxiv_id: str):
        return ATTENTION_ABSTRACT
