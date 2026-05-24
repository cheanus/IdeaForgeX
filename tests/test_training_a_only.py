from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.common import (
    parse_llm_a_candidate,
    parse_query_text,
    validate_and_fix_refinement_edges,
)
from src.agent.training import build_training_graph
from src.config import Config
from src.models import Edge, InspirationNode, LLMACandidate, QuestionNode, RelationType
from tests.fakes import (
    ATTENTION_ABSTRACT,
    ATTENTION_TITLE,
    FakeAMinerClient,
    FakeArxivExtractor,
    TEST_PAPER_ID,
)


class FakeEmbeddingClient(SimpleNamespace):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]


def _zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def _prefix_id(paper_id: str, node_id: str) -> str:
    return f"{paper_id}__{node_id}"


def _mock_paper_not_trained(monkeypatch):
    """模拟论文未训练：is_trained→False（走训练流程），try_reserve→True（commit 时标记成功）。"""
    monkeypatch.setattr(
        "src.agent.training.PaperLibrary.is_trained",
        lambda self, paper_id: False,
    )
    monkeypatch.setattr(
        "src.agent.training.PaperLibrary.try_reserve",
        lambda self, paper_id, title, year="": True,
    )


def _mock_paper_already_trained(monkeypatch):
    """模拟论文已训练：is_trained→True，触发跳过逻辑。"""
    monkeypatch.setattr(
        "src.agent.training.PaperLibrary.is_trained",
        lambda self, paper_id: True,
    )


@pytest.mark.neo4j
def test_training_graph_routes_can_infer_to_commit_candidates(
    monkeypatch, neo4j_client
):
    """验证 can_infer=True（可直接推断）时不创建节点。"""
    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)
    monkeypatch.setattr(
        "src.agent.training.retrieve_with_traversal",
        lambda client, embedding, cfg: [{"node": {"id": "insp-1"}, "score": 0.99}],
    )

    call_count = [0]

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        call_count[0] += 1
        if parser is parse_query_text:
            return {"query_text": "attention mechanism transformer"}
        if parser is parse_llm_a_candidate:
            return LLMACandidate(can_infer=True)
        raise RuntimeError(f"unexpected parser: {parser}")

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)

    monkeypatch.setattr(
        "src.agent.training.PaperLibrary.is_trained",
        lambda self, paper_id: False,
    )

    resolved_id = None

    def track_resolve(self, paper_id, title, year=""):
        nonlocal resolved_id
        resolved_id = paper_id
        return True

    monkeypatch.setattr("src.agent.training.PaperLibrary.try_reserve", track_resolve)

    graph = build_training_graph(config, FakeEmbeddingClient(), neo4j_client)  # type: ignore[reportArgumentType]
    result = graph.invoke(
        {"paper_id": TEST_PAPER_ID, "retry_count": 0},  # type: ignore[reportArgumentType]
        {"configurable": {"thread_id": TEST_PAPER_ID}},
    )

    assert result.get("paper_id") is not None
    assert result["paper_title"] == ATTENTION_TITLE
    assert result["paper_text"] == ATTENTION_ABSTRACT
    assert result["can_infer"] is True
    assert result.get("already_trained") is False
    assert len(result["retrieved_nodes"]) == 1
    assert call_count[0] == 2
    assert resolved_id is not None


@pytest.mark.neo4j
def test_training_graph_commits_candidates_when_cannot_infer(monkeypatch, neo4j_client):
    """验证 can_infer=False（需要提取）时走 commit_candidates 路径并写入节点和关系。"""
    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )
    dim = config.embedding_dim

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)
    monkeypatch.setattr(
        "src.agent.training.retrieve_with_traversal",
        lambda client, embedding, cfg: [],
    )

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        if parser is parse_query_text:
            return {"query_text": "attention mechanism"}
        if parser is parse_llm_a_candidate:
            return LLMACandidate(
                can_infer=False,
                inspiration_nodes=[
                    InspirationNode(
                        id="I1", 核心描述="test insp", 向量=_zero_vector(dim)
                    )
                ],
                question_nodes=[
                    QuestionNode(id="Q1", 核心描述="test q", 向量=_zero_vector(dim))
                ],
                edges=[
                    Edge(
                        from_id="I1",
                        to_id="Q1",
                        rel_type=RelationType.insp_question,
                        weight=0.8,
                    )
                ],
            )
        raise RuntimeError(f"unexpected parser: {parser}")

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)
    _mock_paper_not_trained(monkeypatch)

    graph = build_training_graph(config, FakeEmbeddingClient(), neo4j_client)  # type: ignore[reportArgumentType]
    result = graph.invoke(
        {"paper_id": TEST_PAPER_ID, "retry_count": 0},  # type: ignore[reportArgumentType]
        {"configurable": {"thread_id": TEST_PAPER_ID}},
    )

    assert result["can_infer"] is False
    resolved_id = result.get("paper_id")
    assert resolved_id is not None

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
            f"MATCH (a:Inspiration {{id: '{_prefix_id(resolved_id, 'I1')}'}})"
            f"-[r:INSP_QUESTION]->"
            f"(b:Question {{id: '{_prefix_id(resolved_id, 'Q1')}'}}) "
            "RETURN r.weight AS w"
        ).single()
        assert edge_record is not None
        assert edge_record["w"] == 0.8


@pytest.mark.neo4j
def test_training_graph_commits_node_updates(monkeypatch, neo4j_client):
    """验证 node_updates 能正确 SET 已有节点的属性。"""
    from src.neo4j.schema import create_inspiration

    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )
    dim = config.embedding_dim

    existing = InspirationNode(
        id="insp-existing",
        核心描述="existing method",
        向量=_zero_vector(dim),
        已知实例="old example",
    )
    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        session.execute_write(create_inspiration, existing)

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)
    monkeypatch.setattr(
        "src.agent.training.retrieve_with_traversal",
        lambda client, embedding, cfg: [],
    )

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        if parser is parse_query_text:
            return {"query_text": "attention"}
        if parser is parse_llm_a_candidate:
            return LLMACandidate(
                can_infer=False,
                node_updates=[  # type: ignore[reportArgumentType]
                    {"node_id": "insp-existing", "已知实例": "new example from paper"}
                ],
            )
        raise RuntimeError(f"unexpected parser: {parser}")

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)
    _mock_paper_not_trained(monkeypatch)

    graph = build_training_graph(config, FakeEmbeddingClient(), neo4j_client)  # type: ignore[reportArgumentType]
    result = graph.invoke(
        {"paper_id": TEST_PAPER_ID, "retry_count": 0},  # type: ignore[reportArgumentType]
        {"configurable": {"thread_id": TEST_PAPER_ID}},
    )

    assert result["can_infer"] is False

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        records = session.run(
            "MATCH (n:Inspiration {id: 'insp-existing'}) RETURN n.已知实例 AS ex"
        ).data()
        assert len(records) == 1
        assert records[0]["ex"] == "old example; Attention Is All You Need (2017)"


@pytest.mark.neo4j
def test_training_graph_skips_duplicate(monkeypatch, neo4j_client):
    """验证论文已训练时跳过整个训练流程。"""
    config = Config(
        neo4j_database="neo4j",
        arxiv_short_abstract_threshold=200,
    )

    monkeypatch.setattr("src.paper.resolver.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.paper.resolver.ArxivExtractor", FakeArxivExtractor)
    _mock_paper_already_trained(monkeypatch)

    call_count = [0]

    def fake_call_with_retry(
        client, messages, max_retries=3, temperature=0.1, parser=None
    ):
        call_count[0] += 1
        raise RuntimeError("LLM 不应被调用")

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)

    graph = build_training_graph(config, FakeEmbeddingClient(), neo4j_client)  # type: ignore[reportArgumentType]
    result = graph.invoke(
        {"paper_id": TEST_PAPER_ID, "retry_count": 0},  # type: ignore[reportArgumentType]
        {"configurable": {"thread_id": TEST_PAPER_ID}},
    )

    assert result["already_trained"] is True
    assert result["paper_title"] == ATTENTION_TITLE
    assert "query_text" not in result
    assert "can_infer" not in result
    assert call_count[0] == 0


def test_parse_query_text():
    payload = {"query_text": "attention mechanism for NLP"}
    result = parse_query_text(payload)
    assert result["query_text"] == "attention mechanism for NLP"


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
    assert candidate.inspiration_nodes[0].向量 == []
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


def test_parse_llm_a_candidate_handles_node_updates():
    payload = {
        "can_infer": False,
        "node_updates": [
            {"node_id": "q-existing", "当前现状": "new context"},
        ],
    }

    candidate = parse_llm_a_candidate(payload)

    assert len(candidate.node_updates) == 1
    assert candidate.node_updates[0].node_id == "q-existing"
    assert candidate.node_updates[0].当前现状 == "new context"


def test_parse_llm_a_candidate_preserves_query_text():
    payload = {
        "can_infer": False,
        "query_text": "transformer attention mechanism",
    }

    candidate = parse_llm_a_candidate(payload)

    assert candidate.query_text == "transformer attention mechanism"


class TestValidateAndFixRefinementEdges:
    """validate_and_fix_refinement_edges 边界校验测试。"""

    def _insp(self, nid: str, granularity: int) -> InspirationNode:
        return InspirationNode(id=nid, 核心描述="desc", 粒度=granularity, 向量=[0.0])

    def _edge(self, from_id: str, to_id: str, rel_type: RelationType) -> Edge:
        return Edge(from_id=from_id, to_id=to_id, rel_type=rel_type, weight=1.0)

    def _retrieved(self, items: list[tuple[str, int]]) -> list[dict]:
        return [{"node": {"id": nid, "粒度": g}} for nid, g in items]

    def test_non_refinement_edges_pass_through(self):
        """非 INSP_REFINES 边直接透传。"""
        edges = [self._edge("a", "b", RelationType.insp_question)]
        result = validate_and_fix_refinement_edges(edges, [], [])
        assert len(result) == 1
        assert result[0].rel_type == RelationType.insp_question

    def test_valid_refinement_passes_through(self):
        """粒度 N→N+1 的精化边直接通过。"""
        edges = [self._edge("i1", "i2", RelationType.insp_refines)]
        inspirations = [self._insp("i1", 1), self._insp("i2", 2)]
        result = validate_and_fix_refinement_edges(edges, inspirations, [])
        assert len(result) == 1
        assert result[0].rel_type == RelationType.insp_refines
        assert result[0].from_id == "i1"
        assert result[0].to_id == "i2"

    def test_missing_granularity_passes_through(self):
        """未知粒度的边直接透传。"""
        edges = [self._edge("x", "y", RelationType.insp_refines)]
        result = validate_and_fix_refinement_edges(edges, [], [])
        assert len(result) == 1
        assert result[0].rel_type == RelationType.insp_refines

    def test_granularity_skip_with_retrieved_bridge(self):
        """粒度跳跃但 retrieved 中有中间节点 → 拆分为两条边。"""
        edges = [self._edge("i1", "i3", RelationType.insp_refines)]
        inspirations = [self._insp("i1", 1), self._insp("i3", 3)]
        retrieved = self._retrieved([("bridge", 2)])
        result = validate_and_fix_refinement_edges(edges, inspirations, retrieved)
        assert len(result) == 2
        rel_types = {e.rel_type for e in result}
        assert rel_types == {RelationType.insp_refines}
        ids = {(e.from_id, e.to_id) for e in result}
        assert ids == {("i1", "bridge"), ("bridge", "i3")}

    def test_granularity_skip_with_new_inspiration_bridge(self):
        """粒度跳跃但新的 Inspiration 中有中间节点 → 拆分为两条边。"""
        edges = [self._edge("i1", "i3", RelationType.insp_refines)]
        inspirations = [
            self._insp("i1", 1),
            self._insp("bridge", 2),
            self._insp("i3", 3),
        ]
        result = validate_and_fix_refinement_edges(edges, inspirations, [])
        assert len(result) == 2
        ids = {(e.from_id, e.to_id) for e in result}
        assert ids == {("i1", "bridge"), ("bridge", "i3")}

    def test_granularity_skip_no_bridge_downgraded(self):
        """粒度跳跃且无中间节点 → 降级为 INSP_COMBINES。"""
        edges = [self._edge("i1", "i3", RelationType.insp_refines)]
        inspirations = [self._insp("i1", 1), self._insp("i3", 3)]
        result = validate_and_fix_refinement_edges(edges, inspirations, [])
        assert len(result) == 1
        assert result[0].rel_type == RelationType.insp_combines

    def test_non_increasing_granularity_downgraded(self):
        """非递增粒度 → 降级为 INSP_COMBINES。"""
        edges = [self._edge("i3", "i1", RelationType.insp_refines)]
        inspirations = [self._insp("i3", 3), self._insp("i1", 1)]
        result = validate_and_fix_refinement_edges(edges, inspirations, [])
        assert len(result) == 1
        assert result[0].rel_type == RelationType.insp_combines

    def test_mixed_edges_correctly_handled(self):
        """混合边类型：非精化边 + 合规精化边 + 违规边 各自正确处理。"""
        edges = [
            self._edge("a", "b", RelationType.insp_question),
            self._edge("i1", "i2", RelationType.insp_refines),
            self._edge("i3", "i1", RelationType.insp_refines),
        ]
        inspirations = [
            self._insp("i1", 1),
            self._insp("i2", 2),
            self._insp("i3", 3),
        ]
        result = validate_and_fix_refinement_edges(edges, inspirations, [])
        assert len(result) == 3
        rel_types = [e.rel_type for e in result]
        assert rel_types == [
            RelationType.insp_question,
            RelationType.insp_refines,
            RelationType.insp_combines,
        ]
