"""Neo4j 维护工具。"""

from __future__ import annotations

from src.neo4j.client import Neo4jClient


def resolve_target_uri(target: str) -> str:
    if target == "test":
        return "bolt://localhost:7687"
    if target == "personal":
        return "bolt://localhost:7688"
    raise ValueError(f"未知 Neo4j 目标: {target}")


def clear_graph(client: Neo4jClient) -> None:
    with client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
