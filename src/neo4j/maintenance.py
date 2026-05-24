"""Neo4j 维护工具。"""

from __future__ import annotations

from src.neo4j.client import Neo4jClient


def clear_graph(client: Neo4jClient) -> None:
    with client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
