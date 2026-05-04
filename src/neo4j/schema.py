"""Neo4j schema、写入与重置。"""

from __future__ import annotations

from typing import Any

from src.models import Edge, InspirationNode, NodeType, QuestionNode, RelationType
from src.neo4j.client import Neo4jClient


def ensure_schema(client: Neo4jClient) -> None:
    with client.driver.session(database=client.config.neo4j_database) as session:
        session.run("CREATE CONSTRAINT insp_id_unique IF NOT EXISTS FOR (n:Inspiration) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT q_id_unique IF NOT EXISTS FOR (n:Question) REQUIRE n.id IS UNIQUE")
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
    rel_type = edge.rel_type.value if isinstance(edge.rel_type, RelationType) else str(edge.rel_type)
    tx.run(
        f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[:{rel_type} {{weight: $weight}}]->(b)
        """,
        from_id=edge.from_id,
        to_id=edge.to_id,
        weight=edge.weight,
    )


def batch_write(tx, inspirations: list[InspirationNode], questions: list[QuestionNode], edges: list[Edge]) -> None:
    for node in inspirations:
        create_inspiration(tx, node)
    for node in questions:
        create_question(tx, node)
    for edge in edges:
        create_edge(tx, edge)


def reset_practice_graph(client: Neo4jClient) -> None:
    with client.driver.session(database=client.config.neo4j_database) as session:
        session.run("MATCH (n:Inspiration) DETACH DELETE n")
        session.run("MATCH (n:Question) DETACH DELETE n")

