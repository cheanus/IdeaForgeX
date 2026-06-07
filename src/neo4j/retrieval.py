"""Neo4j 检索与遍历。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient


@dataclass
class RetrievalHit:
    node: dict[str, Any]
    score: float


def _vector_query_cypher(index_name: str, cypher_version: int = 5) -> str:
    if "insp" in index_name:
        props = ".id, .粒度, .核心描述, .前提条件, .操作步骤"
        label = "Inspiration"
    else:
        props = ".id, .核心描述, .问题类型, .当前现状, .未解决部分"
        label = "Question"

    if cypher_version >= 25:
        return f"""
            MATCH (node:{label})
            SEARCH node IN (
                VECTOR INDEX {index_name}
                FOR $embedding
                LIMIT $k
            ) SCORE AS score
            RETURN node {{{props}}}, labels(node)[0] AS node_type, score
            ORDER BY score DESC
        """
    return f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
        YIELD node, score
        RETURN node {{{props}}}, labels(node)[0] AS node_type, score
        ORDER BY score DESC
    """


def vector_search(
    client: Neo4jClient,
    index_name: str,
    query_embedding: list[float],
    k: int,
    cypher_version: int = 5,
) -> list[RetrievalHit]:
    with client.read_session() as session:
        result = session.run(
            _vector_query_cypher(index_name, cypher_version),  # type: ignore[reportArgumentType]
            k=k,
            embedding=query_embedding,
        )
        return [
            RetrievalHit(
                node=dict(record["node"]) | {"type": record.get("node_type", "")},
                score=float(record["score"]),
            )
            for record in result
        ]


def expand_refinement_chain(client: Neo4jClient, hit_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (hit:Inspiration {id: $hit_id})
    CALL (hit) {
        MATCH (hit)-[:INSP_REFINES*0..10]->(finer:Inspiration) RETURN finer
        UNION
        MATCH (coarser:Inspiration)-[:INSP_REFINES*0..10]->(hit) RETURN coarser AS finer
    }
    RETURN DISTINCT finer, labels(finer)[0] AS node_type
    """
    with client.read_session() as session:
        result = session.run(query, hit_id=hit_id)
        return [
            dict(record["finer"]) | {"type": record.get("node_type", "")}
            for record in result
        ]


def batch_expand_refinement_chain(
    client: Neo4jClient, hit_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """批量展开精化链，返回 {hit_id: [chain_nodes]}。一次网络往返处理所有节点。"""
    if not hit_ids:
        return {}

    query = """
    UNWIND $hit_ids AS hit_id
    MATCH (hit:Inspiration {id: hit_id})
    CALL (hit) {
        MATCH (hit)-[:INSP_REFINES*0..10]->(finer:Inspiration) RETURN finer, hit.id AS src_id
        UNION
        MATCH (coarser:Inspiration)-[:INSP_REFINES*0..10]->(hit) RETURN coarser AS finer, hit.id AS src_id
    }
    RETURN DISTINCT finer, labels(finer)[0] AS node_type, src_id
    """
    with client.read_session() as session:
        result = session.run(query, hit_ids=hit_ids)
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in result:
            src_id = record["src_id"]
            groups.setdefault(src_id, []).append(
                dict(record["finer"]) | {"type": record.get("node_type", "")}
            )
        return groups


def hop_expand(
    client: Neo4jClient, node_id: str, max_neighbors: int
) -> list[dict[str, Any]]:
    query = """
    MATCH (n {id: $node_id})-[r]-(m)
    WHERE type(r) IN ['INSP_COMBINES','INSP_QUESTION','QUESTION_COMBINES']
    RETURN m, labels(m)[0] AS node_type, r.weight AS weight
    ORDER BY weight DESC
    LIMIT $max_neighbors
    """
    with client.read_session() as session:
        result = session.run(query, node_id=node_id, max_neighbors=max_neighbors)
        return [
            {
                "node": dict(record["m"]) | {"type": record.get("node_type", "")},
                "weight": float(record["weight"]),
            }
            for record in result
        ]


def retrieve_with_traversal(
    client: Neo4jClient, query_embedding: list[float], config: Config
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    cv = config.cypher_version
    vectors = vector_search(
        client, "idx_insp_vector", query_embedding, config.k_hits, cypher_version=cv
    )
    vectors += vector_search(
        client, "idx_q_vector", query_embedding, config.k_hits, cypher_version=cv
    )
    candidates: dict[str, dict[str, Any]] = {}

    insp_hit_ids: list[str] = []
    insp_hit_scores: dict[str, float] = {}
    for hit in vectors:
        node = hit.node
        node_id = node["id"]
        candidates[node_id] = {"node": node, "score": hit.score, "source": "vector"}
        if node.get("type") == "Inspiration" or "粒度" in node:
            insp_hit_ids.append(node_id)
            insp_hit_scores[node_id] = hit.score

    if insp_hit_ids:
        chains = batch_expand_refinement_chain(client, insp_hit_ids)
        for hit_id, finer_nodes in chains.items():
            score = insp_hit_scores[hit_id]
            for finer in finer_nodes:
                fid = finer["id"]
                if fid not in candidates:
                    candidates[fid] = {
                        "node": finer,
                        "score": score,
                        "source": "chain",
                    }

    frontier = list(candidates.values())
    for _depth in range(config.max_depth):
        next_frontier: list[dict[str, Any]] = []
        hop_label = f"{_depth + 1}-hop"
        for item in frontier:
            node = item["node"]
            source_score = float(item["score"])
            for neighbor in hop_expand(client, node["id"], config.max_neighbors):
                node_data = neighbor["node"]
                score = source_score * config.score_decay
                existing = candidates.get(node_data["id"])
                if existing is None or score > existing["score"]:
                    candidates[node_data["id"]] = {
                        "node": node_data,
                        "score": score,
                        "source": hop_label,
                    }
                    next_frontier.append(
                        {"node": node_data, "score": score, "source": hop_label}
                    )
        frontier = next_frontier

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    return ranked[: config.final_k]


def random_nodes(
    client: Neo4jClient,
    count: int,
    query_embedding: list[float] | None = None,
    pool_size: int = 50,
    cypher_version: int = 5,
) -> list[dict[str, Any]]:
    """从图中随机取 N 个节点。

    有 query_embedding 时先向量检索 pool_size 个候选再从中随机抽样；
    无则全图均匀随机。
    """
    if query_embedding is None:
        query = (
            "MATCH (n) WHERE labels(n)[0] IN ['Inspiration', 'Question'] "
            "RETURN n, labels(n)[0] AS type ORDER BY rand() LIMIT $count"
        )
        with client.read_session() as session:
            result = session.run(query, count=count)
            nodes = []
            for record in result:
                raw = dict(record["n"])
                nodes.append(
                    {
                        "id": raw["id"],
                        "type": record["type"],
                        "core_description": raw.get("核心描述", ""),
                        "granularity": raw.get("粒度"),
                        "source": "random",
                    }
                )
            return nodes

    pool: dict[str, dict[str, Any]] = {}
    for index_name in ("idx_insp_vector", "idx_q_vector"):
        for hit in vector_search(
            client,
            index_name,
            query_embedding,
            pool_size,
            cypher_version=cypher_version,
        ):
            node = hit.node
            if node["id"] not in pool:
                pool[node["id"]] = {
                    "id": node["id"],
                    "type": node.get("type", ""),
                    "core_description": node.get("核心描述", ""),
                    "granularity": node.get("粒度"),
                    "source": "random-weighted",
                }

    sample_size = min(count, len(pool))
    sampled = random.sample(list(pool.values()), sample_size)
    return sampled


def find_shortest_path(
    client: Neo4jClient,
    from_id: str,
    to_id: str,
    max_len: int = 6,
) -> dict[str, Any]:
    """查找两节点间最短路径。

    单条最短路径，包含路径上的节点摘要和边信息。
    无路径时返回 {"connected": false}，自指时返回 hops=0。
    """
    if from_id == to_id:
        with client.read_session() as session:
            record = session.run(
                "MATCH (n {id: $node_id}) RETURN n, labels(n)[0] AS type",
                node_id=from_id,
            ).single()
            if not record:
                return {"connected": False, "reason": "源节点不存在"}
            raw = dict(record["n"])
            return {
                "connected": True,
                "hops": 0,
                "node": {
                    "id": raw["id"],
                    "type": record["type"],
                    "core_description": raw.get("核心描述", ""),
                    "granularity": raw.get("粒度"),
                },
            }

    query = f"""
    MATCH path = shortestPath((a {{id: $from_id}})-[*..{max_len}]-(b {{id: $to_id}}))
    RETURN [node IN nodes(path) | {{id: node.id, type: labels(node)[0],
             core_description: node.核心描述, granularity: node.粒度}}] AS nodes,
           [rel IN relationships(path) | {{type: type(rel), weight: rel.weight}}] AS edges,
           length(path) AS hops
    """
    with client.read_session() as session:
        record = session.run(query, from_id=from_id, to_id=to_id).single()  # type: ignore[reportArgumentType]

    if record is None:
        return {"connected": False, "reason": f"在 max_len={max_len} 内未找到路径"}

    return {
        "connected": True,
        "hops": int(record["hops"]),
        "nodes": [dict(n) for n in (record["nodes"] or [])],
        "edges": [dict(e) for e in (record["edges"] or [])],
    }
