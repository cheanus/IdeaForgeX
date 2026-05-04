from __future__ import annotations

from types import SimpleNamespace

from src.agent.training import build_training_graph
from src.config import Config
from src.models import LLMACandidate


ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions."
)


class FakeSession:
    def __init__(self, log: list[tuple[str, tuple, dict]]):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, fn, *args):
        self.log.append((fn.__name__, args, {}))
        tx = SimpleNamespace(run=lambda *run_args, **run_kwargs: self.log.append(("run", run_args, run_kwargs)))
        return fn(tx, *args)

    def run(self, query, **kwargs):
        self.log.append((query, tuple(), kwargs))


class FakeDriver:
    def __init__(self):
        self.log: list[tuple[str, tuple, dict]] = []

    def session(self, database: str):
        self.log.append(("session", (database,), {}))
        return FakeSession(self.log)


class FakeNeo4jClient:
    def __init__(self, config: Config):
        self.config = config
        self.driver = FakeDriver()


class FakeAMinerClient:
    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, paper_id: str):
        return {"id": paper_id, "title": ATTENTION_TITLE, "abstract_slice": "short abstract"}


class FakeArxivExtractor:
    def __init__(self, config: Config):
        self.config = config

    def find_arxiv_id(self, paper: dict):
        return "1706.03762"

    def fetch_full_text(self, arxiv_id: str):
        return ATTENTION_ABSTRACT


def test_training_graph_routes_can_infer_to_record_paper(monkeypatch):
    config = Config(
        neo4j_database="neo4j-test",
        arxiv_short_abstract_threshold=200,
    )
    neo4j_client = FakeNeo4jClient(config)
    captured: dict[str, object] = {}

    monkeypatch.setattr("src.agent.training.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.agent.training.ArxivExtractor", FakeArxivExtractor)

    def fake_call_with_retry(client, messages, max_retries=3, temperature=0.1, parser=None):
        assert parser is not None
        return LLMACandidate(
            can_infer=True,
        )
    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)

    graph = build_training_graph(config, SimpleNamespace(), neo4j_client)
    result = graph.invoke({"paper_id": "paper-1706.03762", "retry_count": 0}, {"configurable": {"thread_id": "paper-1706.03762"}})

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["paper_text"] == ATTENTION_ABSTRACT
    assert result["can_infer"] is True
    assert any(entry[0] == "session" for entry in neo4j_client.driver.log)
