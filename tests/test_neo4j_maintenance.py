from __future__ import annotations

import pytest

from src.neo4j.maintenance import clear_graph


@pytest.mark.neo4j
def test_neo4j_client_connects_and_queries(neo4j_client):
    """验证无认证连接可正常执行查询。"""
    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        result = session.run("RETURN 1 AS n")
        assert result.single()["n"] == 1


@pytest.mark.neo4j
def test_clear_graph_deletes_all_nodes(neo4j_client):
    """验证 clear_graph 删除图中所有节点。"""
    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        session.run("CREATE (:TestLabel {id: 't1'})")
        session.run("CREATE (:TestLabel {id: 't2'})")

    clear_graph(neo4j_client)

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        result = session.run("MATCH (n) RETURN count(n) AS c")
        assert result.single()["c"] == 0


@pytest.mark.neo4j
def test_ensure_schema_creates_constraints_and_indexes(neo4j_client):
    """验证 ensure_schema 幂等创建约束和向量索引。"""
    from src.neo4j.schema import ensure_schema

    ensure_schema(neo4j_client)

    with neo4j_client.driver.session(
        database=neo4j_client.config.neo4j_database
    ) as session:
        constraints = session.run("SHOW CONSTRAINTS").data()
        constraint_names = {c["name"] for c in constraints}
        assert "insp_id_unique" in constraint_names
        assert "q_id_unique" in constraint_names

        indexes = session.run("SHOW INDEXES").data()
        index_names = {i["name"] for i in indexes}
        assert "idx_insp_vector" in index_names
        assert "idx_q_vector" in index_names
