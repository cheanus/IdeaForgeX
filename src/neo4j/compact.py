"""知识图谱压缩 — 合并语义相近节点。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

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


def _compute_similar_pairs(
    nodes: list[dict[str, Any]], threshold: float
) -> list[tuple[str, str, float]]:
    """用 numpy 批量计算节点间的余弦相似度，返回 score >= threshold 的对。

    返回 [(id_a, id_b, score), ...]，id_a < id_b。
    """
    if len(nodes) < 2:
        return []

    ids = [n["id"] for n in nodes]
    vectors = np.array([n["vector"] for n in nodes], dtype=np.float64)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    normalized = vectors / norms
    similarity = normalized @ normalized.T

    rows, cols = np.triu_indices(len(nodes), k=1)
    triu_scores = similarity[rows, cols]

    mask = triu_scores >= threshold
    rows, cols, scores = rows[mask], cols[mask], triu_scores[mask]

    return [(ids[r], ids[c], float(s)) for r, c, s in zip(rows, cols, scores)]


def _load_insp_refines_map(client: Neo4jClient) -> dict[str, str]:
    """一次性加载所有 INSP_REFINES 关系，返回 {child_id: parent_id}。"""
    refines_map: dict[str, str] = {}
    with client.session() as session:
        result = session.run(
            "MATCH (parent:Inspiration)-[:INSP_REFINES]->(child:Inspiration) "
            "RETURN child.id AS child_id, parent.id AS parent_id"
        )
        for record in result:
            refines_map[record["child_id"]] = record["parent_id"]
    return refines_map


def _pick_all_survivors(client: Neo4jClient, groups: list[list[str]]) -> dict[int, str]:
    """批量选出所有合并组的存活节点。

    每组一条 CALL 子查询，优先级：子节点多 > 贡献论文多 > 核心描述长 > ID 小。
    返回 {group_index: survivor_id}。
    """
    if not groups:
        return {}

    survivors: dict[int, str] = {}
    with client.session() as session:
        for i, group in enumerate(groups):
            if len(group) == 1:
                survivors[i] = group[0]
                continue
            result = session.run(
                """
                UNWIND $ids AS nid
                MATCH (n {id: nid})
                OPTIONAL MATCH (n)-[:INSP_REFINES]->(child:Inspiration)
                OPTIONAL MATCH (paper:Paper)-[:PAPER_CONTRIBUTES]->(n)
                WITH n.id AS id,
                     count(DISTINCT child) AS child_count,
                     count(DISTINCT paper) AS paper_count,
                     size(COALESCE(n.核心描述, '')) AS desc_len
                ORDER BY child_count DESC, paper_count DESC, desc_len DESC, id ASC
                LIMIT 1
                RETURN id AS survivor_id
                """,
                ids=group,
            )
            record = result.single()
            survivors[i] = record["survivor_id"] if record else group[0]
    return survivors


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

                REL_TYPES = {
                    "INSP_REFINES",
                    "INSP_COMBINES",
                    "INSP_QUESTION",
                    "PAPER_CONTRIBUTES",
                    "QUESTION_COMBINES",
                }
                for edge in edges:
                    if edge["neighbor_id"] == survivor_id:
                        continue
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

                tx.run(
                    "MATCH (n {id: $victim_id}) DETACH DELETE n",
                    victim_id=victim_id,
                )

        session.execute_write(_merge_tx)


def _build_groups_from_pairs(
    nodes: list[dict[str, Any]], pairs: list[tuple[str, str, float]]
) -> list[list[str]]:
    """从节点列表和相似对构建合并组。返回 [member_ids, ...]。"""
    if not pairs:
        return []
    all_ids = {n["id"] for n in nodes}
    uf = UnionFind(all_ids)
    for a_id, b_id, _ in pairs:
        uf.union(a_id, b_id)
    return [list(s) for s in uf.get_groups()]


def compact_inspirations(client: Neo4jClient, threshold: float) -> dict[str, int]:
    """自顶向下（粒 1→2→3）合并语义相近的 Inspiration 节点。

    粒度 >1 的合并需满足：上游 INSP_REFINES 父节点也在同一合并组。
    返回 {"总量": N, "粒1": n1, "粒2": n2, "粒3": n3}。
    """
    report: dict[str, int] = {"总量": 0, "粒1": 0, "粒2": 0, "粒3": 0}
    parent_groups: dict[int, dict[str, int]] = {}

    # 预加载所有 INSP_REFINES 关系
    refines_map = _load_insp_refines_map(client)

    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue

        pairs = _compute_similar_pairs(nodes, threshold)
        if not pairs:
            parent_groups[granularity] = {n["id"]: i for i, n in enumerate(nodes)}
            continue

        # 链约束过滤（粒度 > 1）
        if granularity > 1:
            prev_groups = parent_groups.get(granularity - 1, {})

            def _same_parent_group(a_id: str, b_id: str) -> bool:
                pa = refines_map.get(a_id)
                pb = refines_map.get(b_id)
                if pa is None and pb is None:
                    return True
                if pa is None or pb is None:
                    return False
                ga = prev_groups.get(pa)
                gb = prev_groups.get(pb)
                return ga is not None and gb is not None and ga == gb

            pairs = [(a, b, s) for (a, b, s) in pairs if _same_parent_group(a, b)]

        groups = _build_groups_from_pairs(nodes, pairs)

        # 记录本层的合并组映射（供下一层链约束使用）
        parent_groups[granularity] = {}
        for gid, member_list in enumerate(groups):
            for node_id in member_list:
                parent_groups[granularity][node_id] = gid

        # 批量选出存活节点并执行合并
        survivors = _pick_all_survivors(client, groups)
        for gid, member_list in enumerate(groups):
            survivor = survivors.get(gid, member_list[0])
            victims = [nid for nid in member_list if nid != survivor]
            _merge_node_group(client, survivor, victims)
            count = len(victims)
            report["粒" + str(granularity)] += count
            report["总量"] += count

    return report


def compact_questions(client: Neo4jClient, threshold: float) -> dict[str, int]:
    """合并语义相近的 Question 节点（无链约束）。"""
    with client.session() as session:
        result = session.run("MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector")
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]

    if not nodes:
        return {"总量": 0}

    pairs = _compute_similar_pairs(nodes, threshold)
    if not pairs:
        return {"总量": 0}

    groups = _build_groups_from_pairs(nodes, pairs)
    survivors = _pick_all_survivors(client, groups)
    total_merged = 0
    for gid, member_list in enumerate(groups):
        survivor = survivors.get(gid, member_list[0])
        victims = [nid for nid in member_list if nid != survivor]
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
    insp_report = compact_inspirations(client, config.compact_threshold)
    _logger.info(
        "Inspiration 压缩完成，共合并 %d 个节点（粒1=%d, 粒2=%d, 粒3=%d）",
        insp_report["总量"],
        insp_report.get("粒1", 0),
        insp_report.get("粒2", 0),
        insp_report.get("粒3", 0),
    )

    _logger.info("开始压缩 Question 节点 …")
    q_report = compact_questions(client, config.compact_threshold)
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

    report_item = {"total": 0, "groups": 0}

    insp_item = {"total": 0, "groups": 0}
    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue
        pairs = _compute_similar_pairs(nodes, threshold)
        groups = _build_groups_from_pairs(nodes, pairs)
        insp_item["groups"] += len(groups)
        insp_item["total"] += sum(len(g) - 1 for g in groups)

    q_item = {"total": 0, "groups": 0}
    with client.session() as session:
        result = session.run("MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector")
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]
    if nodes:
        pairs = _compute_similar_pairs(nodes, threshold)
        groups = _build_groups_from_pairs(nodes, pairs)
        q_item["groups"] = len(groups)
        q_item["total"] = sum(len(g) - 1 for g in groups)

    return {
        "merged_inspirations": {
            "总量": insp_item["total"],
            "组数": insp_item["groups"],
        },
        "merged_questions": {"总量": q_item["total"], "组数": q_item["groups"]},
        "dry_run": True,
    }
