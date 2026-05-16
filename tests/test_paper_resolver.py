from __future__ import annotations

import httpx
import fitz
import pytest

from src.config import Config
from src.models import InspirationNode, QuestionNode
from src.paper.extractor import ArxivExtractor
from src.paper.resolver import build_practice_summary, resolve_paper_spec
from src.neo4j.schema import create_inspiration, create_question
from tests.fakes import (
    ATTENTION_ABSTRACT,
    ATTENTION_TITLE,
    TEST_ARXIV_ID,
    FakeAMinerClient,
    FakeArxivExtractor,
)


def test_resolve_paper_spec_resolves_arxiv_id_directly(monkeypatch):
    """arXiv ID 格式应直接走 arXiv 路径，不经过 AMiner。"""
    config = Config(arxiv_short_abstract_threshold=200)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)

    result = resolve_paper_spec(config, "1706.03762")

    assert result["paper_id"] == "1706.03762"
    assert result["title"] == ATTENTION_TITLE
    assert result["text"] == ATTENTION_ABSTRACT[:500]


def test_arxiv_title_search_succeeds_skips_aminer(monkeypatch):
    """arXiv 标题搜索优先，命中后不再调用 AMiner。"""
    config = Config(arxiv_short_abstract_threshold=200)

    call_log: list[str] = []

    class SpyAMinerClient:
        def __init__(self, c):
            call_log.append("AMiner.__init__")

        def search_papers(self, query, limit=50):
            call_log.append("AMiner.search_papers")

        def get_paper_detail(self, paper_id):
            call_log.append("AMiner.get_paper_detail")

    monkeypatch.setattr("src.paper.resolver.AMinerClient", SpyAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)

    result = resolve_paper_spec(config, "Attention Is All You Need")

    assert result["title"] == ATTENTION_TITLE
    assert result["paper_id"] == TEST_ARXIV_ID
    assert result["text"] == ATTENTION_ABSTRACT
    assert call_log == []  # AMiner 从未被调用


def test_resolve_paper_spec_falls_back_to_aminer_when_arxiv_fails(monkeypatch):
    """arXiv 标题搜索无结果时降级到 AMiner 语义搜索。"""
    config = Config(arxiv_short_abstract_threshold=200)
    monkeypatch.setattr(
        "src.paper.resolver._try_arxiv_fallback", lambda c, t: None
    )
    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)

    result = resolve_paper_spec(config, "Attention Is All You Need")

    assert result["title"] == ATTENTION_TITLE
    assert result["paper_id"] == TEST_ARXIV_ID
    assert result["text"] == "short abstract"


@pytest.mark.neo4j
def test_build_practice_summary_renders_existing_nodes(neo4j_client):
    """验证 build_practice_summary 读取图中已有节点并生成摘要。"""
    dim = neo4j_client.config.embedding_dim
    insp = InspirationNode(
        id="i1",
        核心描述="desc",
        粒度=1,
        向量=[0.0] * dim,
    )
    q = QuestionNode(
        id="q1",
        核心描述="question",
        向量=[0.0] * dim,
        问题类型="理论缺口",
    )

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        session.execute_write(create_inspiration, insp)
        session.execute_write(create_question, q)

    summary = build_practice_summary(neo4j_client, limit=5)

    assert "i1: desc (粒度 1)" in summary
    assert "q1: question (理论缺口)" in summary


def test_arxiv_extractor_reads_metadata_and_pdf_text(monkeypatch):
    config = Config()
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello Attention")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    def fake_get(url, params=None, timeout=None):
        if url.endswith("api/query"):
            body = (
                "<feed xmlns='http://www.w3.org/2005/Atom'>"
                "<entry><title>Attention Is All You Need</title>"
                "<summary>We propose a new simple network architecture.</summary></entry>"
                "</feed>"
            )
            return httpx.Response(200, text=body, request=httpx.Request("GET", url))
        return httpx.Response(200, content=pdf_bytes, request=httpx.Request("GET", url))

    monkeypatch.setattr("src.paper.extractor.httpx.get", fake_get)

    extractor = ArxivExtractor(config)
    metadata = extractor.get_paper_detail("1706.03762")
    text = extractor.fetch_full_text("1706.03762")

    assert metadata["title"] == "Attention Is All You Need"
    assert "new simple network architecture" in metadata["abstract"]
    assert "Hello Attention" in text


def test_arxiv_extractor_follows_pdf_redirect(monkeypatch):
    config = Config()
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Redirected PDF")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    seen_urls: list[str] = []

    def fake_get(url, params=None, timeout=None, follow_redirects=None):
        seen_urls.append(url)
        if url.endswith(".pdf"):
            return httpx.Response(
                301,
                headers={"location": "/pdf/1706.03762"},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(200, content=pdf_bytes, request=httpx.Request("GET", url))

    monkeypatch.setattr("src.paper.extractor.httpx.get", fake_get)

    extractor = ArxivExtractor(config)
    text = extractor.fetch_full_text("1706.03762")

    assert "Redirected PDF" in text
    assert seen_urls[0].endswith("1706.03762.pdf")
