"""Neo4j 维护工具。"""

from __future__ import annotations

import logging

from src.neo4j.client import Neo4jClient

_logger = logging.getLogger("ideaforgex")


def clear_graph(client: Neo4jClient) -> None:
    with client.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def delete_node_cascade(client: Neo4jClient, node_id: str) -> dict:
    """删除任意类型节点，级联清理孤立依赖。

    Paper 节点：级联删除仅依赖此论文的实践节点（Inspiration/Question）。
    实践节点：级联删除仅依赖此节点的论文节点。
    """
    with client.session() as session:
        record = session.run(
            "MATCH (n {id: $id}) RETURN labels(n) as labels", id=node_id
        ).single()
        if not record:
            return {"deleted": False, "id": node_id, "error": "节点不存在"}

        labels = record["labels"]

        if "Paper" in labels:
            _cascade_orphaned_practice_nodes(session, node_id)
            result = session.run(
                "MATCH (p:Paper {id: $id}) DETACH DELETE p RETURN count(p) as deleted",
                id=node_id,
            ).single()
        else:
            _cascade_orphaned_paper_nodes(session, node_id)
            result = session.run(
                "MATCH (n {id: $id}) DETACH DELETE n RETURN count(n) as deleted",
                id=node_id,
            ).single()

        return {"deleted": True, "id": node_id}


def _cascade_orphaned_practice_nodes(session, paper_id: str) -> None:
    """删除仅被指定论文关联的实践节点。"""
    session.run(
        """
        MATCH (p:Paper {id: $paper_id})-[r:PAPER_CONTRIBUTES]->(n)
        OPTIONAL MATCH (n)<-[:PAPER_CONTRIBUTES]-(other:Paper)
        WHERE other.id <> $paper_id
        WITH n, count(other) AS other_count
        WHERE other_count = 0
        DETACH DELETE n
        """,
        paper_id=paper_id,
    )


def _cascade_orphaned_paper_nodes(session, node_id: str) -> None:
    """删除仅关联指定实践节点的论文节点。"""
    session.run(
        """
        MATCH (n {id: $node_id})<-[r:PAPER_CONTRIBUTES]-(p:Paper)
        OPTIONAL MATCH (p)-[:PAPER_CONTRIBUTES]->(other)
        WHERE other.id <> $node_id
        WITH p, count(other) AS other_count
        WHERE other_count = 0
        DETACH DELETE p
        """,
        node_id=node_id,
    )
