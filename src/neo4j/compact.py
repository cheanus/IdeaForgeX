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

    label = "Inspiration" if "insp" in index_name else "Question"

    with client.session() as session:
        for node in nodes:
            params: dict[str, Any] = {
                "k": topk,
                "vector": node["vector"],
                "threshold": threshold,
                "my_id": node["id"],
            }
            if granularity is not None:
                query = f"""
                MATCH (neighbor:{label})
                SEARCH neighbor IN (
                    VECTOR INDEX {index_name}
                    FOR $vector
                    LIMIT $k
                ) SCORE AS score
                WHERE neighbor.id IN $all_ids
                  AND neighbor.id < $my_id
                  AND neighbor.粒度 = $granularity
                  AND score >= $threshold
                RETURN neighbor.id AS other_id, score
                """
                params["granularity"] = granularity
                params["all_ids"] = list(all_ids)
            else:
                query = f"""
                MATCH (neighbor:{label})
                SEARCH neighbor IN (
                    VECTOR INDEX {index_name}
                    FOR $vector
                    LIMIT $k
                ) SCORE AS score
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


def _get_insp_refines_parent(client: Neo4jClient, node_id: str) -> str | None:
    """获取 Inspiration 节点的 INSP_REFINES 上游父节点 ID。"""
    with client.session() as session:
        record = session.run(
            """
            MATCH (parent:Inspiration)-[:INSP_REFINES]->(child:Inspiration {id: $id})
            RETURN parent.id AS parent_id
            """,
            id=node_id,
        ).single()
        return record["parent_id"] if record else None


def _pick_survivor(client: Neo4jClient, node_ids: set[str]) -> str:
    """在合并组中选出存活节点。优先级：子节点多 > 贡献论文多 > 核心描述长 > ID 小。"""
    node_list = list(node_ids)
    if len(node_list) == 1:
        return node_list[0]

    with client.session() as session:
        result = session.run(
            """
            UNWIND $ids AS nid
            MATCH (n {id: nid})
            OPTIONAL MATCH (n)-[:INSP_REFINES]->(child:Inspiration)
            OPTIONAL MATCH (paper:Paper)-[:PAPER_CONTRIBUTES]->(n)
            RETURN n.id AS id,
                   count(DISTINCT child) AS child_count,
                   count(DISTINCT paper) AS paper_count,
                   size(COALESCE(n.核心描述, '')) AS desc_len
            ORDER BY child_count DESC, paper_count DESC, desc_len DESC, id ASC
            LIMIT 1
            """,
            ids=node_list,
        )
        record = result.single()
        if record:
            return record["id"]
        return node_list[0]


def _merge_node_group(
    client: Neo4jClient, survivor_id: str, victim_ids: list[str]
) -> None:
    """将一个合并组中的冗余节点合并到存活节点。

    操作：转移所有边 → 合并可变属性 → 删除冗余节点。
    允许产生重复边，由后续 deduplicate_edges 清理。
    """
    if not victim_ids:
        return

    with client.driver.session(database=client.config.neo4j_database) as session:

        def _merge_tx(tx) -> None:
            for victim_id in victim_ids:
                # 1. 获取被合并节点的所有边
                edges = tx.run(
                    """
                    MATCH (victim {id: $victim_id})-[r]-(neighbor)
                    WHERE type(r) IN ['INSP_REFINES','INSP_COMBINES','INSP_QUESTION','PAPER_CONTRIBUTES','QUESTION_COMBINES']
                    RETURN type(r) AS rel_type,
                           r.weight AS weight,
                           startNode(r).id AS from_id,
                           endNode(r).id AS to_id,
                           neighbor.id AS neighbor_id
                    """,
                    victim_id=victim_id,
                ).data()

                # 2. 转移边到 survivor
                REL_TYPES = {
                    "INSP_REFINES",
                    "INSP_COMBINES",
                    "INSP_QUESTION",
                    "PAPER_CONTRIBUTES",
                    "QUESTION_COMBINES",
                }
                for edge in edges:
                    if edge["neighbor_id"] == survivor_id:
                        continue  # 跳过自环
                    rel_type = edge["rel_type"]
                    if rel_type not in REL_TYPES:
                        continue
                    weight = edge["weight"]
                    if edge["from_id"] == victim_id:
                        tx.run(
                            f"""
                            MATCH (s {{id: $survivor_id}})
                            MATCH (n {{id: $neighbor_id}})
                            CREATE (s)-[:`{rel_type}` {{weight: $weight}}]->(n)
                            """,
                            survivor_id=survivor_id,
                            neighbor_id=edge["neighbor_id"],
                            weight=weight,
                        )
                    else:
                        tx.run(
                            f"""
                            MATCH (n {{id: $neighbor_id}})
                            MATCH (s {{id: $survivor_id}})
                            CREATE (n)-[:`{rel_type}` {{weight: $weight}}]->(s)
                            """,
                            neighbor_id=edge["neighbor_id"],
                            survivor_id=survivor_id,
                            weight=weight,
                        )

                # 3. 合并可变属性（保留更长的值）
                tx.run(
                    """
                    MATCH (victim {id: $victim_id})
                    MATCH (survivor {id: $survivor_id})
                    SET survivor.前提条件 = CASE
                            WHEN size(COALESCE(survivor.前提条件, '')) >= size(COALESCE(victim.前提条件, ''))
                            THEN COALESCE(survivor.前提条件, victim.前提条件)
                            ELSE victim.前提条件 END,
                        survivor.操作步骤 = CASE
                            WHEN size(COALESCE(survivor.操作步骤, '')) >= size(COALESCE(victim.操作步骤, ''))
                            THEN COALESCE(survivor.操作步骤, victim.操作步骤)
                            ELSE victim.操作步骤 END,
                        survivor.当前现状 = CASE
                            WHEN size(COALESCE(survivor.当前现状, '')) >= size(COALESCE(victim.当前现状, ''))
                            THEN COALESCE(survivor.当前现状, victim.当前现状)
                            ELSE victim.当前现状 END,
                        survivor.未解决部分 = CASE
                            WHEN size(COALESCE(survivor.未解决部分, '')) >= size(COALESCE(victim.未解决部分, ''))
                            THEN COALESCE(survivor.未解决部分, victim.未解决部分)
                            ELSE victim.未解决部分 END
                    """,
                    victim_id=victim_id,
                    survivor_id=survivor_id,
                )

                # 4. 删除冗余节点
                tx.run(
                    "MATCH (n {id: $victim_id}) DETACH DELETE n",
                    victim_id=victim_id,
                )

        session.execute_write(_merge_tx)


def compact_inspirations(
    client: Neo4jClient, threshold: float, topk: int
) -> dict[str, int]:
    """自顶向下（粒 1→2→3）合并语义相近的 Inspiration 节点。

    粒度 >1 的合并需满足：上游 INSP_REFINES 父节点也在同一合并组。
    返回 {"总量": N, "粒1": n1, "粒2": n2, "粒3": n3}。
    """
    report: dict[str, int] = {"总量": 0, "粒1": 0, "粒2": 0, "粒3": 0}
    parent_groups: dict[int, dict[str, int]] = {}

    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue

        pairs = _find_similar_pairs(
            client, nodes, "idx_insp_vector", threshold, topk, granularity=granularity
        )
        if not pairs:
            # 即使没有合并对，仍需记录本层映射供下层使用
            parent_groups[granularity] = {n["id"]: i for i, n in enumerate(nodes)}
            continue

        # 链约束过滤（粒度 > 1）
        if granularity > 1:
            prev_groups = parent_groups.get(granularity - 1, {})

            def _same_parent_group(a_id: str, b_id: str) -> bool:
                pa = _get_insp_refines_parent(client, a_id)
                pb = _get_insp_refines_parent(client, b_id)
                if pa is None and pb is None:
                    return True
                if pa is None or pb is None:
                    return False
                ga = prev_groups.get(pa)
                gb = prev_groups.get(pb)
                return ga is not None and gb is not None and ga == gb

            pairs = [(a, b, s) for (a, b, s) in pairs if _same_parent_group(a, b)]

        # 并查集构建合并组
        all_ids = {n["id"] for n in nodes}
        uf = UnionFind(all_ids)
        for a_id, b_id, _ in pairs:
            uf.union(a_id, b_id)

        groups = uf.get_groups()

        # 记录本层的合并组映射（供下一层链约束使用）
        parent_groups[granularity] = {}
        for gid, member_set in enumerate(groups):
            for node_id in member_set:
                parent_groups[granularity][node_id] = gid

        # 执行合并
        for member_set in groups:
            survivor = _pick_survivor(client, member_set)
            victims = [nid for nid in member_set if nid != survivor]
            _merge_node_group(client, survivor, victims)
            count = len(victims)
            report["粒" + str(granularity)] += count
            report["总量"] += count

    return report


def compact_questions(
    client: Neo4jClient, threshold: float, topk: int
) -> dict[str, int]:
    """合并语义相近的 Question 节点（无链约束）。"""
    with client.session() as session:
        result = session.run("MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector")
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]

    if not nodes:
        return {"总量": 0}

    pairs = _find_similar_pairs(client, nodes, "idx_q_vector", threshold, topk)
    if not pairs:
        return {"总量": 0}

    all_ids = {n["id"] for n in nodes}
    uf = UnionFind(all_ids)
    for a_id, b_id, _ in pairs:
        uf.union(a_id, b_id)

    groups = uf.get_groups()
    total_merged = 0
    for member_set in groups:
        survivor = _pick_survivor(client, member_set)
        victims = [nid for nid in member_set if nid != survivor]
        _merge_node_group(client, survivor, victims)
        total_merged += len(victims)

    return {"总量": total_merged}


def _deduplicate_edges(client: Neo4jClient) -> int:
    """清除合并后产生的重复边（同节点对、同类型）。"""
    with client.driver.session(database=client.config.neo4j_database) as session:
        result = session.run(
            """
            MATCH (a)-[r]->(b)
            MATCH (a)-[r2]->(b)
            WHERE type(r) = type(r2) AND elementId(r) < elementId(r2)
            WITH DISTINCT r2
            DELETE r2
            RETURN count(r2) AS total_removed
            """
        )
        record = result.single()
        return record["total_removed"] or 0 if record else 0


def compact_all(client: Neo4jClient, config: Config) -> dict[str, Any]:
    """完整压缩入口：Inspiration → Question → 边去重。"""
    _logger.info("开始压缩 Inspiration 节点 …")
    insp_report = compact_inspirations(
        client, config.compact_threshold, config.compact_topk
    )
    _logger.info(
        "Inspiration 压缩完成，共合并 %d 个节点（粒1=%d, 粒2=%d, 粒3=%d）",
        insp_report["总量"],
        insp_report.get("粒1", 0),
        insp_report.get("粒2", 0),
        insp_report.get("粒3", 0),
    )

    _logger.info("开始压缩 Question 节点 …")
    q_report = compact_questions(client, config.compact_threshold, config.compact_topk)
    _logger.info("Question 压缩完成，共合并 %d 个节点", q_report["总量"])

    _logger.info("开始去重重复边 …")
    removed_edges = _deduplicate_edges(client)
    _logger.info("边去重完成，共删除 %d 条重复边", removed_edges)

    return {
        "merged_inspirations": insp_report,
        "merged_questions": q_report,
        "removed_duplicate_edges": removed_edges,
        "dry_run": False,
    }


def compact_dry_run(client: Neo4jClient, config: Config) -> dict[str, Any]:
    """不实际执行，仅报告将会合并的节点数量和组数。"""
    threshold = config.compact_threshold
    topk = config.compact_topk
    insp_groups = 0
    insp_merged = 0
    q_groups = 0
    q_merged = 0

    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue
        pairs = _find_similar_pairs(
            client, nodes, "idx_insp_vector", threshold, topk, granularity=granularity
        )
        if not pairs:
            continue
        all_ids = {n["id"] for n in nodes}
        uf = UnionFind(all_ids)
        for a_id, b_id, _ in pairs:
            uf.union(a_id, b_id)
        groups = uf.get_groups()
        insp_groups += len(groups)
        insp_merged += sum(len(g) - 1 for g in groups)

    with client.session() as session:
        result = session.run("MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector")
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]
    if nodes:
        pairs = _find_similar_pairs(client, nodes, "idx_q_vector", threshold, topk)
        if pairs:
            all_ids = {n["id"] for n in nodes}
            uf = UnionFind(all_ids)
            for a_id, b_id, _ in pairs:
                uf.union(a_id, b_id)
            groups = uf.get_groups()
            q_groups = len(groups)
            q_merged = sum(len(g) - 1 for g in groups)

    return {
        "merged_inspirations": {"总量": insp_merged, "组数": insp_groups},
        "merged_questions": {"总量": q_merged, "组数": q_groups},
        "dry_run": True,
    }
