from __future__ import annotations

import httpx
import fitz
import pytest

from src.config import Config
from src.models import InspirationNode, QuestionNode
from src.paper.extractor import ArxivExtractor
from src.paper.resolver import build_practice_summary, load_paper_record
from src.neo4j.schema import create_inspiration, create_question


ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. "
    "The best performing models also connect the encoder and decoder through an attention mechanism. "
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
    "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. "
    "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. "
    "On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, "
    "a small fraction of the training costs of the best models from the literature. "
    "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
)


class FakeAMinerClient:
    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, paper_id: str):
        raise httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(
                500, request=httpx.Request("GET", "https://example.test")
            ),
        )


class FakeArxivExtractor:
    def __init__(self, config: Config):
        self.config = config

    def fetch_full_text(self, arxiv_id: str):
        return ATTENTION_ABSTRACT


def test_load_paper_record_falls_back_to_arxiv_when_aminer_fails(monkeypatch):
    config = Config(arxiv_short_abstract_threshold=200)
    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)

    result = load_paper_record(config, "1706.03762")

    assert result["title"] == "1706.03762"
    assert result["text"] == ATTENTION_ABSTRACT


@pytest.mark.neo4j
def test_build_practice_summary_renders_existing_nodes(neo4j_client):
    """验证 build_practice_summary 读取图中已有节点并生成摘要。"""
    insp = InspirationNode(
        id="i1",
        核心描述="desc",
        粒度=0,
        向量=[0.0] * neo4j_client.config.embedding_dim,
    )
    q = QuestionNode(
        id="q1",
        核心描述="question",
        向量=[0.0] * neo4j_client.config.embedding_dim,
        问题类型="理论缺口",
    )

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        session.execute_write(create_inspiration, insp)
        session.execute_write(create_question, q)

    summary = build_practice_summary(neo4j_client, limit=5)

    assert "i1: desc (粒度 0)" in summary
    assert "q1: question (理论缺口)" in summary


def test_load_paper_record_falls_back_to_arxiv_when_detail_fails(monkeypatch):
    config = Config(arxiv_short_abstract_threshold=200)

    class FailingAMinerClient:
        def __init__(self, config):
            self.config = config

        def get_paper_detail(self, paper_id: str):
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(
                    500, request=httpx.Request("GET", "https://example.test")
                ),
            )

    class WorkingArxivExtractor:
        def __init__(self, config):
            self.config = config

        def fetch_full_text(self, arxiv_id: str):
            return ATTENTION_ABSTRACT

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FailingAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", WorkingArxivExtractor)

    result = load_paper_record(config, "1706.03762")

    assert result["title"] == "1706.03762"
    assert result["text"] == ATTENTION_ABSTRACT


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
