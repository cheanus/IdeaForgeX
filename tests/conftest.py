from __future__ import annotations

import os

import pytest

from src.config import Config
from src.neo4j.client import Neo4jClient
from src.neo4j.maintenance import clear_graph


def pytest_configure(config):
    config.addinivalue_line("markers", "neo4j: requires Neo4j reset before test")


@pytest.fixture(autouse=True)
def clear_test_neo4j_before_marked_tests(request):
    if "neo4j" not in request.keywords:
        yield
        return
    if os.getenv("IDEAFORGEX_SKIP_NEO4J_RESET") == "1":
        yield
        return

    config = Config(neo4j_uri="bolt://localhost:7687", neo4j_user="", neo4j_password="")
    client = Neo4jClient(config)
    try:
        clear_graph(client)
    finally:
        client.close()
    yield
