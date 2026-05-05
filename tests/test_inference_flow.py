from __future__ import annotations

from types import SimpleNamespace

from src.agent.common import parse_llm_a_candidate
from src.agent.inference import build_inference_graph
from src.config import Config
from src.models import LLMACandidate
from tests.fakes import (
    ATTENTION_ABSTRACT,
    ATTENTION_TITLE,
    FakeAMinerClient,
    TEST_PAPER_ID,
)


class FakeEmbeddingClient:
    def __init__(self, config: Config):
        self.config = config

    def embed(self, texts: list[str]):
        return [[0.1, 0.2, 0.3]]


def test_inference_graph_generates_llm_a_candidates(monkeypatch):
    config = Config(neo4j_database="neo4j")
    neo4j_client = SimpleNamespace(config=config, driver=SimpleNamespace())
    retrieved_nodes = [{"node": {"id": "insp-1", "type": "Inspiration"}, "score": 0.99}]

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr(
        "src.agent.inference.retrieve_with_traversal",
        lambda client, embedding, cfg: retrieved_nodes,
    )

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        assert parser is parse_llm_a_candidate
        return LLMACandidate(
            can_infer=False,
            inspiration_nodes=[],
            question_nodes=[],
            edges=[],
        )

    monkeypatch.setattr("src.agent.inference.call_with_retry", fake_call_with_retry)

    graph = build_inference_graph(config, FakeEmbeddingClient(config), neo4j_client)  # type: ignore[reportArgumentType]
    result = graph.invoke(
        {"paper_id": TEST_PAPER_ID},
        {"configurable": {"thread_id": TEST_PAPER_ID}},
    )

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["retrieved_nodes"] == retrieved_nodes
    assert result["llm_a"]["can_infer"] is False
