from __future__ import annotations

import os

import pytest

from src.config import Config
from src.neo4j.client import Neo4jClient
from src.neo4j.maintenance import clear_graph
from src.neo4j.schema import ensure_schema


def pytest_configure(config):
    config.addinivalue_line("markers", "neo4j: requires Neo4j test container")


@pytest.fixture
def neo4j_client():
    """连接到 neo4j-test 容器的客户端。"""
    config = Config(neo4j_uri="bolt://localhost:7687", neo4j_user="", neo4j_password="")
    client = Neo4jClient(config)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def _reset_neo4j_for_marked_tests(request):
    """标记了 @pytest.mark.neo4j 的测试运行前：清空图库 + 重建 schema。"""
    if "neo4j" not in request.keywords:
        return
    if os.getenv("IDEAFORGEX_SKIP_NEO4J_RESET") == "1":
        return
    config = Config(neo4j_uri="bolt://localhost:7687", neo4j_user="", neo4j_password="")
    client = Neo4jClient(config)
    try:
        clear_graph(client)
        ensure_schema(client)
    finally:
        client.close()
