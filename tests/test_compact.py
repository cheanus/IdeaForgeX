"""compact 压缩算法单元测试。"""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.neo4j.compact import UnionFind

_TEST_VECTOR_DIM = load_config().embedding_dim


class TestUnionFind:
    def test_single_element(self):
        uf = UnionFind({"a"})
        assert uf.find("a") == "a"
        assert uf.get_groups() == []

    def test_union_pair(self):
        uf = UnionFind({"a", "b"})
        uf.union("a", "b")
        groups = uf.get_groups()
        assert len(groups) == 1
        assert groups[0] == {"a", "b"}

    def test_union_chain(self):
        uf = UnionFind({"a", "b", "c"})
        uf.union("a", "b")
        uf.union("b", "c")
        groups = uf.get_groups()
        assert len(groups) == 1
        assert groups[0] == {"a", "b", "c"}

    def test_disjoint_groups(self):
        uf = UnionFind({"a", "b", "c", "d"})
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.get_groups()
        assert len(groups) == 2

    def test_no_merge_singletons(self):
        uf = UnionFind({"a", "b", "c"})
        uf.union("a", "b")
        groups = uf.get_groups()
        # "c" is singleton, should be excluded
        assert len(groups) == 1
        assert groups[0] == {"a", "b"}

    def test_find_path_compression(self):
        uf = UnionFind({"a", "b", "c", "d"})
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        # After path compression, all should have same root
        root = uf.find("a")
        assert uf.find("b") == root
        assert uf.find("c") == root
        assert uf.find("d") == root


@pytest.mark.neo4j
class TestCompactNeo4j:
    """Neo4j 集成测试，需要 neo4j-test 容器运行。"""

    @staticmethod
    def _vec(*values: float) -> list[float]:
        v = list(values)
        if len(v) < _TEST_VECTOR_DIM:
            v.extend([0.0] * (_TEST_VECTOR_DIM - len(v)))
        return v

    def _create_insp_node(
        self,
        neo4j_client,
        node_id: str,
        granularity: int,
        desc: str,
        vector: list[float],
    ) -> None:
        with neo4j_client.driver.session(
            database=neo4j_client.config.neo4j_database
        ) as session:
            session.run(
                """
                CREATE (n:Inspiration {
                    id: $id, 粒度: $granularity, 核心描述: $desc, 向量: $vector,
                    前提条件: '', 操作步骤: ''
                })
                """,
                id=node_id,
                granularity=granularity,
                desc=desc,
                vector=vector,
            )

    def _create_q_node(
        self, neo4j_client, node_id: str, desc: str, vector: list[float]
    ) -> None:
        with neo4j_client.driver.session(
            database=neo4j_client.config.neo4j_database
        ) as session:
            session.run(
                """
                CREATE (n:Question {
                    id: $id, 核心描述: $desc, 向量: $vector,
                    问题类型: '理论缺口', 当前现状: '', 未解决部分: ''
                })
                """,
                id=node_id,
                desc=desc,
                vector=vector,
            )

    def test_compact_no_nodes(self, neo4j_client):
        """空图压缩应正常返回。"""
        from src.neo4j.compact import compact_all

        config = Config()
        result = compact_all(neo4j_client, config)
        assert result["merged_inspirations"]["总量"] == 0
        assert result["merged_questions"]["总量"] == 0

    def test_compact_dry_run_no_modification(self, neo4j_client):
        """dry-run 不修改图。"""
        self._create_insp_node(
            neo4j_client, "insp-1", 1, "test", self._vec(0.1, 0.2, 0.3)
        )
        from src.neo4j.compact import compact_dry_run

        config = Config()
        result = compact_dry_run(neo4j_client, config)
        assert result["dry_run"] is True

        # 验证节点未被删除
        with neo4j_client.session() as session:
            count = session.run(
                "MATCH (n:Inspiration {id: 'insp-1'}) RETURN count(n) AS c"
            ).single()["c"]
        assert count == 1
