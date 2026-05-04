from __future__ import annotations

from types import SimpleNamespace

from src.config import Config
from src.neo4j.client import Neo4jClient
from src.neo4j.maintenance import resolve_target_uri


def test_neo4j_client_uses_no_auth_when_credentials_are_blank(monkeypatch):
    calls: list[tuple[str, object]] = []

    def fake_driver(uri: str, auth=None):
        calls.append((uri, auth))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr("src.neo4j.client.GraphDatabase.driver", fake_driver)

    config = Config(neo4j_uri="bolt://localhost:7687", neo4j_user="", neo4j_password="")
    client = Neo4jClient(config)

    assert calls == [("bolt://localhost:7687", None)]
    client.close()


def test_neo4j_client_uses_no_auth_when_password_is_blank(monkeypatch):
    calls: list[tuple[str, object]] = []

    def fake_driver(uri: str, auth=None):
        calls.append((uri, auth))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr("src.neo4j.client.GraphDatabase.driver", fake_driver)

    config = Config(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="")
    client = Neo4jClient(config)

    assert calls == [("bolt://localhost:7687", None)]
    client.close()


def test_resolve_target_uri_maps_test_and_personal_ports():
    assert resolve_target_uri("test") == "bolt://localhost:7687"
    assert resolve_target_uri("personal") == "bolt://localhost:7688"
