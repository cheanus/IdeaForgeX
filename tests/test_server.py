"""FastAPI 服务器端点测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import Config
from src.server.app import create_app


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def client(config):
    """构建 TestClient，注入 mock 客户端防止 lifespan 真正连数据库。"""

    mock_neo4j = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("src.server.app.create_client", return_value=mock_neo4j),
        patch("src.server.app.ChatClient", return_value=mock_llm),
    ):
        app = create_app(config)
        with TestClient(app) as tc:
            yield tc


class TestHealth:
    def test_root_returns_ok(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "IdeaForgeX"

    def test_docs_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema["paths"]
        assert "/retrieve" in paths
        assert "/inspect/{node_ids}" in paths
        assert "/stats" in paths
        assert "/random" in paths
        assert "/relate/{from_id}/{to_id}" in paths
        # 不暴露写命令
        assert "/train" not in paths
        assert "/batch-train" not in paths
        assert "/bootstrap" not in paths
        assert "/reset" not in paths
        assert "/delete-node" not in paths
        assert "/compact" not in paths


class TestRetrieve:
    def test_retrieve_requires_query(self, client):
        response = client.post("/retrieve", json={})
        assert response.status_code == 422

    def test_retrieve_with_query(self, client):
        with patch("src.server.app.cmd_retrieve") as mock_cmd:
            mock_cmd.return_value = {
                "query": "test query",
                "nodes": [],
                "meta": {"total_hits": 0, "runtime_ms": 10},
            }
            response = client.post("/retrieve", json={"query": "test query"})
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert data["meta"]["total_hits"] == 0
            mock_cmd.assert_called_once()

    def test_retrieve_with_all_params(self, client):
        with patch("src.server.app.cmd_retrieve") as mock_cmd:
            mock_cmd.return_value = {
                "query": "q",
                "nodes": [],
                "meta": {"total_hits": 0, "runtime_ms": 0},
            }
            request_body = {
                "query": "扩散模型医学图像分割",
                "top_k": 10,
                "expand_hops": 2,
                "max_per_node": 3,
                "decay": 0.6,
                "final_limit": 20,
            }
            response = client.post("/retrieve", json=request_body)
            assert response.status_code == 200
            call_kwargs = mock_cmd.call_args.kwargs
            assert call_kwargs["top_k"] == 10
            assert call_kwargs["expand_hops"] == 2
            assert call_kwargs["max_per_node"] == 3
            assert call_kwargs["decay"] == 0.6
            assert call_kwargs["final_limit"] == 20


class TestInspect:
    def test_inspect_returns_nodes(self, client):
        with patch("src.server.app.cmd_inspect") as mock_cmd:
            mock_cmd.return_value = [
                {
                    "node": {
                        "id": "insp-001",
                        "type": "Inspiration",
                        "core_description": "测试灵感",
                    },
                    "chain": [],
                    "edges": [],
                }
            ]
            response = client.get("/inspect/insp-001")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["node"]["id"] == "insp-001"

    def test_inspect_comma_separated(self, client):
        with patch("src.server.app.cmd_inspect") as mock_cmd:
            mock_cmd.return_value = [
                {"node": {"id": "a"}, "chain": [], "edges": []},
                {"node": {"id": "b"}, "chain": [], "edges": []},
            ]
            response = client.get("/inspect/a,b")
            assert response.status_code == 200
            assert len(response.json()) == 2

    def test_inspect_no_expand_edges(self, client):
        with patch("src.server.app.cmd_inspect") as mock_cmd:
            mock_cmd.return_value = [{"node": {"id": "x"}, "chain": [], "edges": []}]
            response = client.get("/inspect/x?expand_edges=false")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["expand_edges"] is False

    def test_inspect_default_expand_edges(self, client):
        with patch("src.server.app.cmd_inspect") as mock_cmd:
            mock_cmd.return_value = [{"node": {"id": "x"}, "chain": [], "edges": []}]
            response = client.get("/inspect/x")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["expand_edges"] is True


class TestStats:
    def test_stats_returns_counts(self, client):
        with patch("src.server.app.cmd_stats") as mock_cmd:
            mock_cmd.return_value = {
                "nodes": {
                    "inspiration_total": 100,
                    "question_total": 50,
                    "paper_total": 30,
                },
                "inspiration_granularity": {},
                "question_types": {},
                "edges": {},
                "paper_years": {},
            }
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["nodes"]["inspiration_total"] == 100


class TestRandom:
    def test_random_default_count(self, client):
        with patch("src.server.app.cmd_random") as mock_cmd:
            mock_cmd.return_value = {
                "mode": "random",
                "nodes": [],
                "meta": {"total_hits": 0, "runtime_ms": 0},
            }
            response = client.get("/random")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["count"] == 5
            assert mock_cmd.call_args.kwargs["query_text"] is None

    def test_random_with_count(self, client):
        with patch("src.server.app.cmd_random") as mock_cmd:
            mock_cmd.return_value = {
                "mode": "random",
                "nodes": [],
                "meta": {"total_hits": 0, "runtime_ms": 0},
            }
            response = client.get("/random?count=10")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["count"] == 10

    def test_random_with_query(self, client):
        with patch("src.server.app.cmd_random") as mock_cmd:
            mock_cmd.return_value = {
                "mode": "random-weighted",
                "nodes": [],
                "meta": {"total_hits": 0, "runtime_ms": 0},
            }
            response = client.get("/random?query=扩散模型")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["query_text"] == "扩散模型"
            assert mock_cmd.call_args.kwargs["count"] == 5


class TestRelate:
    def test_relate_connected(self, client):
        with patch("src.server.app.cmd_relate") as mock_cmd:
            mock_cmd.return_value = {
                "connected": True,
                "hops": 2,
                "nodes": [],
                "edges": [],
                "meta": {"runtime_ms": 5},
            }
            response = client.get("/relate/insp-001/insp-002")
            assert response.status_code == 200
            data = response.json()
            assert data["connected"] is True
            assert data["hops"] == 2

    def test_relate_with_max_len(self, client):
        with patch("src.server.app.cmd_relate") as mock_cmd:
            mock_cmd.return_value = {"connected": False, "reason": "无路径"}
            response = client.get("/relate/a/b?max_len=4")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["max_len"] == 4

    def test_relate_default_max_len(self, client):
        with patch("src.server.app.cmd_relate") as mock_cmd:
            mock_cmd.return_value = {"connected": False, "reason": "无路径"}
            response = client.get("/relate/a/b")
            assert response.status_code == 200
            assert mock_cmd.call_args.kwargs["max_len"] == 6
