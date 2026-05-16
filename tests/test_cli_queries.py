"""CLI queries 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.cli.queries import cmd_inspect, cmd_retrieve
from src.config import Config

FAKE_INSP_NODE = {
    "id": "insp-001",
    "type": "Inspiration",
    "粒度": 1,
    "核心描述": "基于注意力机制的序列建模范式",
    "前提条件": "序列长度适中，有足够训练数据",
    "操作步骤": "1) 计算 self-attention 2) 多头聚合 3) 残差归一化",
    "已知实例": "Transformer (NeurIPS 2017), BERT (NAACL 2019)",
    "向量": [0.1, 0.2],
}
FAKE_QUESTION_NODE = {
    "id": "q-001",
    "type": "Question",
    "核心描述": "长序列注意力计算复杂度过高",
    "问题类型": "工程瓶颈",
    "当前现状": "已有稀疏注意力、线性注意力等方法",
    "未解决部分": "长程依赖与计算效率的平衡仍未彻底解决",
    "向量": [0.3, 0.4],
}
FAKE_CHAIN_INSP = [
    {"id": "insp-g0", "粒度": 0, "核心描述": "序列建模"},
    {"id": "insp-001", "粒度": 1, "核心描述": "基于注意力机制的序列建模范式"},
    {"id": "insp-g2", "粒度": 2, "核心描述": "Multi-Head Scaled Dot-Product Attention"},
]
FAKE_EDGES_BATCH = {
    "insp-001": [
        {
            "type": "INSP_COMBINES",
            "weight": 0.85,
            "target": {
                "id": "insp-c77f",
                "type": "Inspiration",
                "核心描述": "多尺度特征对齐",
                "粒度": 2,
            },
        },
        {
            "type": "INSP_QUESTION",
            "weight": 0.76,
            "target": {
                "id": "q-4a2e",
                "type": "Question",
                "核心描述": "医学图像标注成本高导致少样本过拟合",
                "问题类型": "工程瓶颈",
            },
        },
    ],
}


class FakeLLMClient:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3]]


def _make_fake_neo4j_session(single_result: dict | None):
    """构建一个兼容 context manager 的 mock session。"""

    class FakeRecord:
        def __init__(self, wrapped: dict):
            self._wrapped = wrapped

        def __getitem__(self, key):
            return self._wrapped[key]

        def data(self):
            return self._wrapped

    session = MagicMock()
    if single_result is not None:
        session.run.return_value.single.return_value = FakeRecord({"n": single_result})
    else:
        session.run.return_value.single.return_value = None
    return session


def _make_fake_neo4j_client_ctx(session):
    """构建兼容 'with client.session() as session:' 的 mock client。"""

    def _session_ctx():
        return MagicMock(__enter__=lambda s: session, __exit__=lambda *a: None)

    return _session_ctx


def test_cmd_retrieve_format(monkeypatch):
    """验证 cmd_retrieve 输出格式符合 docs/use_cli.md 定义。"""
    config = Config()

    def fake_retrieve_with_traversal(client, embedding, cfg):
        return [
            {"node": FAKE_INSP_NODE, "score": 0.92},
            {"node": FAKE_QUESTION_NODE, "score": 0.85},
        ]

    monkeypatch.setattr(
        "src.cli.queries.retrieve_with_traversal", fake_retrieve_with_traversal
    )

    def fake_build_chain(node, client):
        if node.get("type") == "Question":
            return []
        return [
            {
                "id": n["id"],
                "granularity": n["粒度"],
                "core_description": n["核心描述"],
                "direction": d,
            }
            for n, d in zip(FAKE_CHAIN_INSP, ["coarser", "self", "finer"])
        ]

    monkeypatch.setattr("src.cli.queries._build_chain", fake_build_chain)
    monkeypatch.setattr(
        "src.cli.queries._get_edges_batch", lambda c, ids: FAKE_EDGES_BATCH
    )

    result = cmd_retrieve(
        config,
        FakeLLMClient(),  # type: ignore[reportArgumentType]
        SimpleNamespace(),  # type: ignore[reportArgumentType]
        "使用注意力机制做序列建模",
    )

    assert result["query"] == "使用注意力机制做序列建模"
    assert "meta" in result
    assert result["meta"]["expansion_hops"] == config.max_depth
    assert isinstance(result["meta"]["runtime_ms"], int)
    assert result["meta"]["runtime_ms"] >= 0

    assert len(result["nodes"]) == 2

    # Inspiration 节点
    insp = result["nodes"][0]
    assert insp["id"] == "insp-001"
    assert insp["type"] == "Inspiration"
    assert insp["score"] == 0.92
    assert insp["granularity"] == 1
    assert insp["core_description"] == "基于注意力机制的序列建模范式"
    assert insp["snippet"] == {
        "前提条件": "序列长度适中，有足够训练数据",
        "操作步骤": "1) 计算 self-attention 2) 多头聚合 3) 残差归一化",
        "已知实例": "Transformer (NeurIPS 2017), BERT (NAACL 2019)",
    }
    assert len(insp["chain"]) == 3
    assert insp["chain"][0]["direction"] == "coarser"
    assert insp["chain"][1]["direction"] == "self"
    assert insp["chain"][2]["direction"] == "finer"

    assert len(insp["edges"]) == 2
    assert insp["edges"][0]["type"] == "灵感组合边"
    assert insp["edges"][0]["target"] == "insp-c77f"
    assert insp["edges"][0]["target_summary"] == "多尺度特征对齐"

    # Question 节点（无精化链）
    q = result["nodes"][1]
    assert q["id"] == "q-001"
    assert q["type"] == "Question"
    assert q["granularity"] is None
    assert q["chain"] == []
    assert q["snippet"] == {
        "问题类型": "工程瓶颈",
        "当前现状": "已有稀疏注意力、线性注意力等方法",
        "未解决部分": "长程依赖与计算效率的平衡仍未彻底解决",
    }


def test_cmd_inspect_inspiration_node(monkeypatch):
    """验证 inspect 对 Inspiration 节点输出全字段。"""
    session = _make_fake_neo4j_session(FAKE_INSP_NODE)
    fake_neo4j = SimpleNamespace(session=_make_fake_neo4j_client_ctx(session))

    monkeypatch.setattr(
        "src.cli.queries._build_chain",
        lambda node, client: [
            {
                "id": n["id"],
                "granularity": n["粒度"],
                "core_description": n["核心描述"],
                "direction": d,
            }
            for n, d in zip(FAKE_CHAIN_INSP, ["coarser", "self", "finer"])
        ],
    )
    monkeypatch.setattr("src.cli.queries._get_edges_for_node", lambda c, nid: [])

    results = cmd_inspect(fake_neo4j, "insp-001")  # type: ignore[reportArgumentType]

    assert len(results) == 1
    node_output = results[0]["node"]
    assert node_output["id"] == "insp-001"
    assert node_output["type"] == "Inspiration"
    assert node_output["granularity"] == 1
    assert node_output["core_description"] == "基于注意力机制的序列建模范式"
    assert node_output["前提条件"] == "序列长度适中，有足够训练数据"
    assert node_output["操作步骤"] == "1) 计算 self-attention 2) 多头聚合 3) 残差归一化"
    assert node_output["已知实例"] == "Transformer (NeurIPS 2017), BERT (NAACL 2019)"
    assert len(results[0]["chain"]) == 3


def test_cmd_inspect_question_node(monkeypatch):
    """验证 inspect 对 Question 节点输出全字段。"""
    session = _make_fake_neo4j_session(FAKE_QUESTION_NODE)
    fake_neo4j = SimpleNamespace(session=_make_fake_neo4j_client_ctx(session))

    monkeypatch.setattr("src.cli.queries._get_edges_for_node", lambda c, nid: [])
    monkeypatch.setattr("src.cli.queries._build_chain", lambda node, client: [])

    results = cmd_inspect(fake_neo4j, "q-001")  # type: ignore[reportArgumentType]

    assert len(results) == 1
    node_output = results[0]["node"]
    assert node_output["id"] == "q-001"
    assert node_output["type"] == "Question"
    assert node_output["core_description"] == "长序列注意力计算复杂度过高"
    assert node_output["问题类型"] == "工程瓶颈"
    assert node_output["当前现状"] == "已有稀疏注意力、线性注意力等方法"
    assert node_output["未解决部分"] == "长程依赖与计算效率的平衡仍未彻底解决"
    assert results[0]["chain"] == []


def test_cmd_inspect_edges_expanded(monkeypatch):
    """验证 inspect 展开边时 target 为 mini-inspect 格式。"""
    session = _make_fake_neo4j_session(FAKE_INSP_NODE)
    fake_neo4j = SimpleNamespace(session=_make_fake_neo4j_client_ctx(session))

    monkeypatch.setattr(
        "src.cli.queries._get_edges_for_node",
        lambda c, nid: FAKE_EDGES_BATCH["insp-001"],
    )
    monkeypatch.setattr(
        "src.cli.queries._build_chain",
        lambda node, client: [
            {
                "id": n["id"],
                "granularity": n["粒度"],
                "core_description": n["核心描述"],
                "direction": d,
            }
            for n, d in zip(FAKE_CHAIN_INSP, ["coarser", "self", "finer"])
        ],
    )

    results = cmd_inspect(fake_neo4j, "insp-001")  # type: ignore[reportArgumentType]

    edges = results[0]["edges"]
    assert len(edges) == 2
    assert edges[0]["type"] == "灵感组合边"
    assert edges[0]["target"]["id"] == "insp-c77f"
    assert edges[0]["target"]["type"] == "Inspiration"
    assert edges[0]["target"]["core_description"] == "多尺度特征对齐"
    assert edges[1]["type"] == "灵感问题边"
    assert edges[1]["target"]["type"] == "Question"
    assert edges[1]["target"]["问题类型"] == "工程瓶颈"


def test_cmd_inspect_missing_node(monkeypatch):
    """验证 inspect 查询不存在的节点时返回 error。"""
    session = _make_fake_neo4j_session(None)
    fake_neo4j = SimpleNamespace(session=_make_fake_neo4j_client_ctx(session))

    monkeypatch.setattr("src.cli.queries._get_edges_for_node", lambda c, nid: [])
    monkeypatch.setattr("src.cli.queries._build_chain", lambda node, client: [])

    results = cmd_inspect(fake_neo4j, "nonexistent")  # type: ignore[reportArgumentType]

    assert len(results) == 1
    assert "error" in results[0]
    assert results[0]["id"] == "nonexistent"


def test_cmd_inspect_comma_separated_ids(monkeypatch):
    """验证 inspect 支持逗号分隔多个 ID。"""
    session_a = _make_fake_neo4j_session(FAKE_INSP_NODE)
    session_b = _make_fake_neo4j_session(FAKE_QUESTION_NODE)
    sessions = iter([session_a, session_b])

    def _session_ctx():
        s = next(sessions)
        return MagicMock(__enter__=lambda _: s, __exit__=lambda *a: None)

    fake_neo4j = SimpleNamespace(session=_session_ctx)

    def fake_build_chain(node, client):
        if node.get("type") == "Inspiration":
            return [
                {
                    "id": n["id"],
                    "granularity": n["粒度"],
                    "core_description": n["核心描述"],
                    "direction": d,
                }
                for n, d in zip(FAKE_CHAIN_INSP, ["coarser", "self", "finer"])
            ]
        return []

    monkeypatch.setattr("src.cli.queries._get_edges_for_node", lambda c, nid: [])
    monkeypatch.setattr("src.cli.queries._build_chain", fake_build_chain)

    results = cmd_inspect(fake_neo4j, "a,b")  # type: ignore[reportArgumentType]

    assert len(results) == 2
    assert results[0]["node"]["id"] == "insp-001"
    assert results[1]["node"]["id"] == "q-001"
