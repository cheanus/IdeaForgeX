from __future__ import annotations

from types import SimpleNamespace

from src.agent.inference import build_inference_graph
from src.config import Config

ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions."
)


class FakePaperClient:
    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, paper_id: str):
        return {
            "id": paper_id,
            "title": ATTENTION_TITLE,
            "abstract_slice": ATTENTION_ABSTRACT,
        }


def test_inference_graph_returns_llm_a_state(monkeypatch):
    config = Config(neo4j_database="neo4j-test")
    neo4j_client = SimpleNamespace(config=config, driver=SimpleNamespace())
    retrieved_nodes = [{"node": {"id": "insp-1", "type": "Inspiration"}, "score": 0.99}]

    monkeypatch.setattr("src.agent.inference.AMinerClient", FakePaperClient)
    monkeypatch.setattr(
        "src.agent.inference.retrieve_with_traversal",
        lambda client, embedding, cfg: retrieved_nodes,
    )

    class FakeInferenceClient:
        def __init__(self, config: Config):
            self.config = config

        def embed(self, texts: list[str]):
            return [[0.1, 0.2, 0.3]]

    graph = build_inference_graph(config, FakeInferenceClient(config), neo4j_client)
    result = graph.invoke(
        {"paper_id": "paper-1706.03762"},
        {"configurable": {"thread_id": "paper-1706.03762"}},
    )

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["retrieved_nodes"] == retrieved_nodes
    assert result["llm_a"]["paper_title"] == ATTENTION_TITLE
    assert result["llm_a"]["retrieved_nodes"] == retrieved_nodes
