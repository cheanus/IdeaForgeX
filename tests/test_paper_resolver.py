from __future__ import annotations

import httpx
import fitz

from src.config import Config
from src.paper.extractor import ArxivExtractor
from src.paper.resolver import build_practice_summary, load_paper_record


ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions."
)


class FakeAMinerClient:
    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, paper_id: str):
        raise httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(500, request=httpx.Request("GET", "https://example.test")),
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


def test_build_practice_summary_renders_existing_nodes():
    class FakeRecord:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def data(self):
            return {"n": self._data}

    class FakeResult(list):
        pass

    class FakeSession:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query, **kwargs):
            if "MATCH (n:Inspiration)" in query:
                return FakeResult([FakeRecord({"id": "i1", "core": "desc", "grain": 0})])
            return FakeResult([FakeRecord({"id": "q1", "core": "question", "qtype": "理论缺口"})])

    class FakeDriver:
        def session(self, database):
            return FakeSession()

    class FakeClient:
        def __init__(self):
            self.config = Config(neo4j_database="neo4j-test")
            self.driver = FakeDriver()

    summary = build_practice_summary(FakeClient(), limit=5)

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
                response=httpx.Response(500, request=httpx.Request("GET", "https://example.test")),
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
