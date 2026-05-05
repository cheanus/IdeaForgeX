from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.common import parse_llm_a_candidate
from src.agent.training import build_training_graph
from src.config import Config
from src.models import Edge, InspirationNode, LLMACandidate, QuestionNode, RelationType


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
        return {
            "id": paper_id,
            "title": ATTENTION_TITLE,
            "abstract_slice": "short abstract",
        }


class FakeArxivExtractor:
    def __init__(self, config: Config):
        self.config = config

    def find_arxiv_id(self, paper: dict):
        return "1706.03762"

    def fetch_full_text(self, arxiv_id: str):
        return ATTENTION_ABSTRACT


@pytest.mark.neo4j
def test_training_graph_routes_can_infer_to_record_paper(monkeypatch, neo4j_client):
    """验证 can_infer=True 时走 record_paper 路径并实际写入 Neo4j。"""
    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        assert parser is not None
        return LLMACandidate(can_infer=True)

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)

    graph = build_training_graph(config, SimpleNamespace(), neo4j_client)
    result = graph.invoke(
        {"paper_id": "paper-1706.03762", "retry_count": 0},
        {"configurable": {"thread_id": "paper-1706.03762"}},
    )

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["paper_text"] == ATTENTION_ABSTRACT
    assert result["can_infer"] is True

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        records = session.run(
            "MATCH (r:PaperRecord {id: $id}) RETURN r.title AS title",
            id="paper-1706.03762",
        ).data()
        assert len(records) == 1
        assert records[0]["title"] == ATTENTION_TITLE


@pytest.mark.neo4j
def test_training_graph_commits_candidates_when_cannot_infer(monkeypatch, neo4j_client):
    """验证 can_infer=False 时走 commit_candidates 路径并实际写入节点和关系。"""
    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        return LLMACandidate(
            can_infer=False,
            inspiration_nodes=[
                InspirationNode(id="i1", 核心描述="test insp", 向量=[0.0] * 1536)
            ],
            question_nodes=[
                QuestionNode(id="q1", 核心描述="test q", 向量=[0.0] * 1536)
            ],
            edges=[
                Edge(
                    from_id="i1",
                    to_id="q1",
                    rel_type=RelationType.insp_question,
                    weight=0.8,
                )
            ],
        )

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)

    graph = build_training_graph(config, SimpleNamespace(), neo4j_client)
    result = graph.invoke(
        {"paper_id": "paper-1706.03762", "retry_count": 0},
        {"configurable": {"thread_id": "paper-1706.03762"}},
    )

    assert result["can_infer"] is False

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        insp_count = session.run("MATCH (n:Inspiration) RETURN count(n) AS c").single()[
            "c"
        ]
        q_count = session.run("MATCH (n:Question) RETURN count(n) AS c").single()["c"]
        assert insp_count == 1
        assert q_count == 1

        edge_record = session.run(
            "MATCH (a:Inspiration {id: 'i1'})-[r:INSP_QUESTION]->(b:Question {id: 'q1'}) "
            "RETURN r.weight AS w"
        ).single()
        assert edge_record is not None
        assert edge_record["w"] == 0.8


def test_parse_llm_a_candidate_maps_raw_llm_output():
    payload = {
        "can_infer": False,
        "inspiration_nodes": {
            "I1": {"content": "desc", "granularity": 1, "embedding": [0.1, 0.2]},
        },
        "question_nodes": {
            "Q1": {
                "content": "question",
                "question_type": "方法重构",
                "embedding": [0.3, 0.4],
            },
        },
        "edges": {
            "e1": {
                "source": "I1",
                "target": "Q1",
                "relation": "INSP_QUESTION",
                "weight": 0.8,
            },
        },
    }

    candidate = parse_llm_a_candidate(payload)

    assert candidate.inspiration_nodes[0].核心描述 == "desc"
    assert candidate.inspiration_nodes[0].向量 == [0.1, 0.2]
    assert candidate.question_nodes[0].核心描述 == "question"
    assert candidate.edges[0].from_id == "I1"


def test_parse_llm_a_candidate_accepts_numeric_ids():
    payload = {
        "can_infer": False,
        "inspiration_nodes": [
            {"id": 1, "content": "desc", "embedding": [0.1, 0.2]},
            {"id": 2, "content": "fine desc", "embedding": [0.3, 0.4]},
        ],
        "question_nodes": [
            {"id": 3, "content": "question", "embedding": [0.5, 0.6]},
        ],
        "edges": [
            {"source": 1, "target": 2, "relation": "INSP_REFINES", "weight": 1.0},
            {"source": 1, "target": 3, "relation": "INSP_QUESTION", "weight": 0.8},
        ],
    }

    candidate = parse_llm_a_candidate(payload)

    assert candidate.inspiration_nodes[0].id == "1"
    assert candidate.inspiration_nodes[1].id == "2"
    assert candidate.edges[0].rel_type == RelationType.insp_refines
    assert candidate.edges[1].rel_type == RelationType.insp_question
