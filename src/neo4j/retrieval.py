"""Neo4j 检索与遍历。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient


@dataclass
class RetrievalHit:
    node: dict[str, Any]
    score: float


def _vector_query_cypher(index_name: str, k: int) -> str:
    return """
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
    """


def vector_search(
    client: Neo4jClient, index_name: str, query_embedding: list[float], k: int
) -> list[RetrievalHit]:
    with client.session() as session:
        result = session.run(
            _vector_query_cypher(index_name, k),
            index=index_name,
            k=k,
            embedding=query_embedding,
        )
        return [
            RetrievalHit(node=record["node"].data(), score=float(record["score"]))
            for record in result
        ]


def expand_refinement_chain(client: Neo4jClient, hit_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (hit:Inspiration {id: $hit_id})
    CALL {
        MATCH (hit)-[:INSP_REFINES*0..10]->(finer:Inspiration) RETURN finer
        UNION
        MATCH (coarser:Inspiration)-[:INSP_REFINES*0..10]->(hit) RETURN coarser AS finer
    }
    RETURN DISTINCT finer
    """
    with client.session() as session:
        result = session.run(query, hit_id=hit_id)
        return [record["finer"].data() for record in result]


def hop_expand(
    client: Neo4jClient, node_id: str, max_neighbors: int
) -> list[dict[str, Any]]:
    query = """
    MATCH (n {id: $node_id})-[r:INSP_COMBINES|INSP_QUESTION|QUESTION_COMBINES]-(m)
    RETURN m, r.weight AS weight
    ORDER BY weight DESC
    LIMIT $max_neighbors
    """
    with client.session() as session:
        result = session.run(query, node_id=node_id, max_neighbors=max_neighbors)
        return [
            {"node": record["m"].data(), "weight": float(record["weight"])}
            for record in result
        ]


def retrieve_with_traversal(
    client: Neo4jClient, query_embedding: list[float], config: Config
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    vectors = vector_search(client, "idx_insp_vector", query_embedding, config.k_hits)
    vectors += vector_search(client, "idx_q_vector", query_embedding, config.k_hits)
    candidates: dict[str, dict[str, Any]] = {}

    for hit in vectors:
        node = hit.node
        node_id = node["id"]
        candidates[node_id] = {"node": node, "score": hit.score}
        if node.get("type") == "Inspiration" or "粒度" in node:
            for finer in expand_refinement_chain(client, node_id):
                candidates[finer["id"]] = {"node": finer, "score": hit.score}

    frontier = list(candidates.values())
    for _depth in range(config.max_depth):
        next_frontier: list[dict[str, Any]] = []
        for item in frontier:
            node = item["node"]
            source_score = float(item["score"])
            for neighbor in hop_expand(client, node["id"], config.max_neighbors):
                node_data = neighbor["node"]
                score = source_score * config.score_decay
                existing = candidates.get(node_data["id"])
                if existing is None or score > existing["score"]:
                    candidates[node_data["id"]] = {"node": node_data, "score": score}
                    next_frontier.append({"node": node_data, "score": score})
        frontier = next_frontier

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    return ranked[: config.final_k]
