"""retrieve / inspect 命令实现。"""

from __future__ import annotations

import time
from typing import Any

from src.config import Config
from src.llm.client import ChatClient
from src.neo4j.client import Neo4jClient
from src.neo4j.retrieval import expand_refinement_chain, retrieve_with_traversal

EDGE_TYPE_LABELS: dict[str, str] = {
    "INSP_COMBINES": "灵感组合边",
    "INSP_QUESTION": "灵感问题边",
    "QUESTION_COMBINES": "问题组合边",
}
QUESTION_FIELDS: list[str] = ["问题类型", "当前现状", "未解决部分"]
INSPIRATION_FIELDS: list[str] = ["前提条件", "操作步骤", "已知实例"]


def _build_snippet(node: dict[str, Any]) -> dict[str, str]:
    type_label = node.get("type", "")
    if type_label == "Question":
        fields = QUESTION_FIELDS
    else:
        fields = INSPIRATION_FIELDS
    return {f: str(node.get(f, "")) for f in fields if node.get(f)}


def _build_chain(node: dict[str, Any], client: Neo4jClient) -> list[dict[str, Any]]:
    if node.get("type") == "Question" or "粒度" not in node:
        return []
    node_id = node["id"]
    all_nodes = expand_refinement_chain(client, node_id)
    current_granularity = node.get("粒度", 1)
    chain: list[dict[str, Any]] = []
    for n in all_nodes:
        g = n.get("粒度", current_granularity)
        if g < current_granularity:
            direction = "coarser"
        elif g > current_granularity:
            direction = "finer"
        else:
            direction = "self"
        chain.append(
            {
                "id": n["id"],
                "granularity": g,
                "core_description": n.get("核心描述", ""),
                "direction": direction,
            }
        )
    chain.sort(key=lambda c: c["granularity"])
    return chain


def _get_edges_batch(
    client: Neo4jClient, node_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    if not node_ids:
        return {}
    query = """
    MATCH (n)-[r]-(m)
    WHERE n.id IN $node_ids AND type(r) IN ['INSP_COMBINES','INSP_QUESTION','QUESTION_COMBINES']
    RETURN n.id AS source_id, type(r) AS rel_type, r.weight AS weight, m AS target
    ORDER BY source_id, weight DESC
    """
    with client.session() as session:
        result = session.run(query, node_ids=node_ids)
        edges_by_source: dict[str, list[dict[str, Any]]] = {nid: [] for nid in node_ids}
        for record in result:
            edges_by_source[record["source_id"]].append(
                {
                    "type": record["rel_type"],
                    "weight": float(record["weight"]),
                    "target": dict(record["target"]),
                }
            )
        return edges_by_source


def _get_edges_for_node(client: Neo4jClient, node_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (n {id: $node_id})-[r]-(m)
    WHERE type(r) IN ['INSP_COMBINES','INSP_QUESTION','QUESTION_COMBINES']
    RETURN type(r) AS rel_type, r.weight AS weight, m AS target
    ORDER BY weight DESC
    """
    with client.session() as session:
        result = session.run(query, node_id=node_id)
        return [
            {
                "type": record["rel_type"],
                "weight": float(record["weight"]),
                "target": dict(record["target"]),
            }
            for record in result
        ]


def cmd_retrieve(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    query_text: str,
    top_k: int | None = None,
    expand_hops: int | None = None,
    max_per_node: int | None = None,
    decay: float | None = None,
    final_limit: int | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    embeddings = llm_client.embed([query_text])

    k_hits = top_k or config.k_hits
    max_depth = expand_hops or config.max_depth
    max_neighbors = max_per_node or config.max_neighbors
    score_decay = decay or config.score_decay
    final_k = final_limit or config.final_k

    candidates = retrieve_with_traversal(neo4j_client, embeddings[0], config)

    total_hits = len(candidates)
    ranked = candidates[:final_k]

    node_ids = [item["node"]["id"] for item in ranked]
    edges_map = _get_edges_batch(neo4j_client, node_ids)

    nodes: list[dict[str, Any]] = []
    for item in ranked:
        raw_node = item["node"]
        node_id = raw_node["id"]
        chain = _build_chain(raw_node, neo4j_client)
        raw_edges = edges_map.get(node_id, [])
        edges_out: list[dict[str, Any]] = []
        for e in raw_edges:
            tgt = e["target"]
            edges_out.append(
                {
                    "type": EDGE_TYPE_LABELS.get(e["type"], e["type"]),
                    "target": tgt["id"],
                    "weight": e["weight"],
                    "target_summary": tgt.get("核心描述", ""),
                }
            )
        nodes.append(
            {
                "id": node_id,
                "type": raw_node.get("type", ""),
                "score": item["score"],
                "granularity": raw_node.get("粒度"),
                "core_description": raw_node.get("核心描述", ""),
                "snippet": _build_snippet(raw_node),
                "chain": chain,
                "edges": edges_out,
            }
        )

    runtime_ms = int((time.monotonic() - t0) * 1000)

    return {
        "query": query_text,
        "nodes": nodes,
        "meta": {
            "total_hits": total_hits,
            "expansion_hops": max_depth,
            "decay_factor": score_decay,
            "runtime_ms": runtime_ms,
        },
    }


def cmd_inspect(
    neo4j_client: Neo4jClient,
    node_ids_str: str,
    expand_edges: bool = True,
) -> list[dict[str, Any]]:
    node_ids = [nid.strip() for nid in node_ids_str.split(",") if nid.strip()]
    results: list[dict[str, Any]] = []

    for node_id in node_ids:
        with neo4j_client.session() as session:
            record = session.run(
                "MATCH (n {id: $node_id}) RETURN n", node_id=node_id
            ).single()
            if not record:
                results.append({"id": node_id, "error": "节点不存在"})
                continue
            raw_node = dict(record["n"])

        type_label = raw_node.get("type", "")
        node: dict[str, Any] = {
            "id": raw_node["id"],
            "type": type_label,
            "core_description": raw_node.get("核心描述", ""),
        }
        if type_label == "Inspiration":
            node["granularity"] = raw_node.get("粒度")
            for f in INSPIRATION_FIELDS:
                node[f] = raw_node.get(f, "")
        elif type_label == "Question":
            for f in QUESTION_FIELDS:
                node[f] = raw_node.get(f, "")

        chain = _build_chain(raw_node, neo4j_client)

        edges_out: list[dict[str, Any]] = []
        if expand_edges:
            for e in _get_edges_for_node(neo4j_client, node_id):
                tgt = e["target"]
                tgt_type = tgt.get("type", "")
                tgt_summary: dict[str, Any] = {
                    "id": tgt["id"],
                    "type": tgt_type,
                    "core_description": tgt.get("核心描述", ""),
                }
                if tgt_type == "Inspiration":
                    tgt_summary["granularity"] = tgt.get("粒度")
                elif tgt_type == "Question" and tgt.get("问题类型"):
                    tgt_summary["问题类型"] = tgt["问题类型"]
                edges_out.append(
                    {
                        "type": EDGE_TYPE_LABELS.get(e["type"], e["type"]),
                        "target": tgt_summary,
                        "weight": e["weight"],
                    }
                )

        results.append(
            {
                "node": node,
                "chain": chain,
                "edges": edges_out,
            }
        )

    return results
