"""知识图谱压缩 — 合并语义相近节点。"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient

_logger = logging.getLogger("ideaforgex")


class UnionFind:
    """并查集，用于将相似节点对聚合为合并组。"""

    def __init__(self, elements: set[str]) -> None:
        self._parent = {e: e for e in elements}
        self._rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self._rank[px] < self._rank[py]:
            self._parent[px] = py
        elif self._rank[px] > self._rank[py]:
            self._parent[py] = px
        else:
            self._parent[py] = px
            self._rank[px] += 1

    def get_groups(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for e in self._parent:
            root = self.find(e)
            if root not in groups:
                groups[root] = set()
            groups[root].add(e)
        return [s for s in groups.values() if len(s) > 1]


def _fetch_inspiration_nodes(
    client: Neo4jClient, granularity: int
) -> list[dict[str, Any]]:
    """获取指定粒度的 Inspiration 节点列表。"""
    with client.session() as session:
        result = session.run(
            "MATCH (n:Inspiration) WHERE n.粒度 = $granularity RETURN n.id AS id, n.向量 AS vector",
            granularity=granularity,
        )
        return [{"id": r["id"], "vector": r["vector"]} for r in result]


def _find_similar_pairs(
    client: Neo4jClient,
    nodes: list[dict[str, Any]],
    index_name: str,
    threshold: float,
    topk: int,
    granularity: int | None = None,
) -> list[tuple[str, str, float]]:
    """对节点列表通过 HNSW 向量索引查找相似对。

    返回 [(id_a, id_b, score), ...]，score >= threshold。
    使用 `id_a < id_b` 避免重复对。
    """
    from src.neo4j.schema import ensure_schema

    ensure_schema(client)

    all_ids = {n["id"] for n in nodes}
    pairs: dict[tuple[str, str], float] = {}

    with client.session() as session:
        for node in nodes:
            params: dict[str, Any] = {
                "index": index_name,
                "k": topk,
                "vector": node["vector"],
                "threshold": threshold,
                "my_id": node["id"],
            }
            if granularity is not None:
                query = """
                CALL db.index.vector.queryNodes($index, $k, $vector)
                YIELD node AS neighbor, score
                WHERE neighbor.id IN $all_ids
                  AND neighbor.id < $my_id
                  AND neighbor.粒度 = $granularity
                  AND score >= $threshold
                RETURN neighbor.id AS other_id, score
                """
                params["granularity"] = granularity
                params["all_ids"] = list(all_ids)
            else:
                query = """
                CALL db.index.vector.queryNodes($index, $k, $vector)
                YIELD node AS neighbor, score
                WHERE neighbor.id IN $all_ids
                  AND neighbor.id < $my_id
                  AND score >= $threshold
                RETURN neighbor.id AS other_id, score
                """
                params["all_ids"] = list(all_ids)

            records = session.run(query, **params)
            for record in records:
                key = (node["id"], record["other_id"])
                pairs[key] = record["score"]

    return [(a, b, s) for (a, b), s in pairs.items()]
