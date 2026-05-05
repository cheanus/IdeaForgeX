"""Neo4j schema、写入与重置。"""

from __future__ import annotations

from typing import Any

from src.models import Edge, InspirationNode, NodeUpdate, QuestionNode, RelationType
from src.neo4j.client import Neo4jClient


def ensure_schema(client: Neo4jClient) -> None:
    with client.session() as session:
        session.run(
            "CREATE CONSTRAINT insp_id_unique IF NOT EXISTS FOR (n:Inspiration) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT q_id_unique IF NOT EXISTS FOR (n:Question) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            """
            CREATE VECTOR INDEX idx_insp_vector IF NOT EXISTS
            FOR (n:Inspiration) ON (n.向量)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $embedding_dim,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            embedding_dim=client.config.embedding_dim,
        )
        session.run(
            """
            CREATE VECTOR INDEX idx_q_vector IF NOT EXISTS
            FOR (n:Question) ON (n.向量)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $embedding_dim,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            embedding_dim=client.config.embedding_dim,
        )


def _node_to_props(node: InspirationNode | QuestionNode) -> dict[str, Any]:
    return node.model_dump(exclude={"type"})


def create_inspiration(tx, node: InspirationNode) -> None:
    tx.run(
        """
        CREATE (n:Inspiration {
            id: $id,
            粒度: $粒度,
            核心描述: $核心描述,
            向量: $向量,
            前提条件: $前提条件,
            操作步骤: $操作步骤,
            已知实例: $已知实例
        })
        """,
        **_node_to_props(node),
    )


def create_question(tx, node: QuestionNode) -> None:
    tx.run(
        """
        CREATE (n:Question {
            id: $id,
            核心描述: $核心描述,
            向量: $向量,
            问题类型: $问题类型,
            当前现状: $当前现状,
            未解决部分: $未解决部分
        })
        """,
        **_node_to_props(node),
    )


def create_edge(tx, edge: Edge) -> None:
    rel_type = (
        edge.rel_type.value
        if isinstance(edge.rel_type, RelationType)
        else str(edge.rel_type)
    )
    tx.run(
        f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[:{rel_type} {{weight: $weight}}]->(b)
        """,
        from_id=edge.from_id,
        to_id=edge.to_id,
        weight=edge.weight,
    )


def batch_write(
    tx,
    inspirations: list[InspirationNode],
    questions: list[QuestionNode],
    edges: list[Edge],
) -> None:
    for node in inspirations:
        create_inspiration(tx, node)
    for node in questions:
        create_question(tx, node)
    for edge in edges:
        create_edge(tx, edge)


def update_node(tx, update: NodeUpdate) -> None:
    """MERGE 已有节点并 SET 指定字段。"""
    mutable_fields = [
        "粒度",
        "前提条件",
        "操作步骤",
        "已知实例",
        "问题类型",
        "当前现状",
        "未解决部分",
    ]
    set_clauses: list[str] = []
    params: dict[str, Any] = {"node_id": update.node_id}
    for field in mutable_fields:
        value = getattr(update, field, None)
        if value is not None:
            set_clauses.append(f"n.{field} = ${field}")
            params[field] = value
    if not set_clauses:
        return
    tx.run(
        f"MATCH (n {{id: $node_id}}) SET {', '.join(set_clauses)}",
        **params,
    )


def batch_update(tx, updates: list[NodeUpdate]) -> None:
    for upd in updates:
        update_node(tx, upd)


def reset_practice_graph(client: Neo4jClient) -> None:
    with client.session() as session:
        session.run("MATCH (n:Inspiration) DETACH DELETE n")
        session.run("MATCH (n:Question) DETACH DELETE n")
