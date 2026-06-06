"""retrieve / inspect / random / relate / delete-node 命令实现。"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from typing import Any

from tqdm import tqdm

from src.config import Config
from src.llm.client import ChatClient
from src.neo4j.client import Neo4jClient
from src.neo4j.maintenance import delete_node_cascade as _delete_node_cascade
from src.neo4j.retrieval import (
    expand_refinement_chain,
    find_shortest_path,
    random_nodes,
    retrieve_with_traversal,
)

EDGE_TYPE_LABELS: dict[str, str] = {
    "INSP_COMBINES": "灵感组合边",
    "INSP_QUESTION": "灵感问题边",
    "QUESTION_COMBINES": "问题组合边",
    "PAPER_CONTRIBUTES": "论文贡献边",
}
QUESTION_FIELDS: list[str] = ["问题类型", "当前现状", "未解决部分"]
INSPIRATION_FIELDS: list[str] = ["前提条件", "操作步骤"]

_logger = logging.getLogger("ideaforgex")


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


def _get_edges_for_node(client: Neo4jClient, node_id: str) -> list[dict[str, Any]]:
    query = """
    MATCH (n {id: $node_id})-[r]-(m)
    WHERE type(r) IN ['INSP_COMBINES','INSP_QUESTION','QUESTION_COMBINES','PAPER_CONTRIBUTES']
    RETURN type(r) AS rel_type, r.weight AS weight, m AS target, labels(m)[0] AS target_type
    ORDER BY weight DESC
    """
    with client.session() as session:
        result = session.run(query, node_id=node_id)
        return [
            {
                "type": record["rel_type"],
                "weight": float(record["weight"]),
                "target": dict(record["target"])
                | {"type": record.get("target_type", "")},
            }
            for record in result
        ]


def _get_papers_for_node(client: Neo4jClient, node_id: str) -> list[dict[str, Any]]:
    """查找与指定节点关联的论文节点。"""
    query = """
    MATCH (p:Paper)-[:PAPER_CONTRIBUTES]->(n {id: $node_id})
    RETURN p.id AS id, p.title AS title, p.year AS year
    """
    with client.session() as session:
        records = session.run(query, node_id=node_id)
        return [
            {"id": record["id"], "title": record["title"], "year": record["year"]}
            for record in records
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

    overrides = {}
    if top_k is not None:
        overrides["k_hits"] = top_k
    if expand_hops is not None:
        overrides["max_depth"] = expand_hops
    if max_per_node is not None:
        overrides["max_neighbors"] = max_per_node
    if decay is not None:
        overrides["score_decay"] = decay
    if final_limit is not None:
        overrides["final_k"] = final_limit

    effective_config = config.model_copy(update=overrides) if overrides else config
    candidates = retrieve_with_traversal(neo4j_client, embeddings[0], effective_config)

    total_hits = len(candidates)
    _logger.info("图检索完成，返回 %d 条", total_hits)
    ranked = candidates[: effective_config.final_k]

    nodes: list[dict[str, Any]] = []
    for item in ranked:
        raw_node = item["node"]
        nodes.append(
            {
                "id": raw_node["id"],
                "type": raw_node.get("type", ""),
                "score": item["score"],
                "source": item.get("source", "unknown"),
                "granularity": raw_node.get("粒度"),
                "core_description": raw_node.get("核心描述", ""),
            }
        )

    runtime_ms = int((time.monotonic() - t0) * 1000)

    return {
        "query": query_text,
        "nodes": nodes,
        "meta": {
            "total_hits": total_hits,
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
                "MATCH (n {id: $node_id}) RETURN n, labels(n)[0] AS node_type",
                node_id=node_id,
            ).single()
            if not record:
                results.append({"id": node_id, "error": "节点不存在"})
                continue
            raw_node = dict(record["n"]) | {"type": record.get("node_type", "")}

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
            node["papers"] = _get_papers_for_node(neo4j_client, node_id)
        elif type_label == "Question":
            for f in QUESTION_FIELDS:
                node[f] = raw_node.get(f, "")
            node["papers"] = _get_papers_for_node(neo4j_client, node_id)
        elif type_label == "Paper":
            node["title"] = raw_node.get("title", "")
            node["year"] = raw_node.get("year", "")
            node["abstract"] = raw_node.get("abstract", "")
            node["trained_at"] = raw_node.get("trained_at", "")

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
                elif tgt_type == "Paper":
                    tgt_summary["title"] = tgt.get("title", "")
                    tgt_summary["year"] = tgt.get("year", "")
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


def cmd_random(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    count: int = 5,
    query_text: str | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()

    query_embedding: list[float] | None = None
    if query_text:
        query_embedding = llm_client.embed([query_text])[0]

    nodes = random_nodes(neo4j_client, count, query_embedding)

    runtime_ms = int((time.monotonic() - t0) * 1000)

    return {
        "mode": "random-weighted" if query_text else "random",
        "nodes": nodes,
        "meta": {
            "total_hits": len(nodes),
            "runtime_ms": runtime_ms,
        },
    }


def cmd_relate(
    neo4j_client: Neo4jClient,
    from_id: str,
    to_id: str,
    max_len: int = 6,
) -> dict[str, Any]:
    t0 = time.monotonic()
    result = find_shortest_path(neo4j_client, from_id, to_id, max_len)
    result["meta"] = {"runtime_ms": int((time.monotonic() - t0) * 1000)}
    return result


def cmd_delete_node(
    neo4j_client: Neo4jClient,
    node_id: str,
) -> dict[str, Any]:
    return _delete_node_cascade(neo4j_client, node_id)


def cmd_compact(
    config: Config,
    neo4j_client: Neo4jClient,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行全图压缩，dry_run 为 True 时仅报告不执行。"""
    from src.neo4j.compact import compact_all, compact_dry_run

    if dry_run:
        return compact_dry_run(neo4j_client, config)
    return compact_all(neo4j_client, config)


def cmd_stats(neo4j_client: Neo4jClient) -> dict[str, Any]:
    """返回知识图谱统计概览：节点数、粒度分布、问题类型、边分布、论文年份。"""
    with neo4j_client.session() as session:
        insp_total = session.run(  # type: ignore[arg-type]
            "MATCH (n:Inspiration) RETURN count(n) AS c"
        ).single()["c"]
        q_total = session.run(  # type: ignore[arg-type]
            "MATCH (n:Question) RETURN count(n) AS c"
        ).single()["c"]
        paper_total = session.run(  # type: ignore[arg-type]
            "MATCH (n:Paper) RETURN count(n) AS c"
        ).single()["c"]

        granularity: dict[str, int] = {}
        for g in [1, 2, 3]:
            granularity[f"粒{g}"] = session.run(  # type: ignore[arg-type]
                "MATCH (n:Inspiration) WHERE n.粒度 = $g RETURN count(n) AS c",
                g=g,
            ).single()["c"]

        question_types: dict[str, int] = {}
        result = session.run(  # type: ignore[arg-type]
            "MATCH (n:Question) RETURN n.问题类型 AS qt, count(n) AS c ORDER BY c DESC"
        )
        for record in result:
            question_types[record["qt"]] = record["c"]

        edge_counts: dict[str, int] = {}
        result = session.run(  # type: ignore[arg-type]
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC"
        )
        for record in result:
            edge_counts[record["t"]] = record["c"]

        paper_years: dict[str, int] = {}
        result = session.run(  # type: ignore[arg-type]
            "MATCH (n:Paper) WHERE n.year IS NOT NULL RETURN n.year AS year, count(n) AS c ORDER BY year"
        )
        for record in result:
            paper_years[record["year"]] = record["c"]

    return {
        "nodes": {
            "inspiration_total": insp_total,
            "question_total": q_total,
            "paper_total": paper_total,
        },
        "inspiration_granularity": granularity,
        "question_types": question_types,
        "edges": edge_counts,
        "paper_years": paper_years,
    }


def _train_one_paper(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    paper: str,
) -> dict[str, Any]:
    """训练单篇论文（字符串查询），返回 run_training 原始结果。"""
    from src.agent.training import build_training_graph, run_training

    graph = build_training_graph(config, llm_client, neo4j_client)
    return run_training(graph, paper)


def _train_one_record(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    record: dict[str, Any],
) -> dict[str, Any]:
    """训练单篇论文（预加载 JSONL 记录），返回 run_training 原始结果。"""
    from src.agent.training import build_training_graph, run_training_with_record
    from src.paper.resolver import paper_from_record

    pr = paper_from_record(record)
    paper_id = pr["paper_id"]
    graph = build_training_graph(config, llm_client, neo4j_client)
    return run_training_with_record(graph, paper_id, record)


def cmd_batch_train(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    papers: list[str],
    jsonl_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """并行训练多篇论文，定期 compact。

    Args:
        papers: 论文标识列表（走 resolve_paper_spec 解析）
        jsonl_records: JSONL 预加载记录列表（跳过 API 解析）
    """
    from src.neo4j.compact import compact_all

    total = len(papers) + (len(jsonl_records) if jsonl_records else 0)
    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    lock = threading.Lock()

    interval = config.compact_interval

    tasks: list[tuple[str, dict[str, Any] | None]] = []
    tasks.extend((p, None) for p in papers)
    if jsonl_records:
        for rec in jsonl_records:
            tasks.append((rec.get("id", ""), rec))

    if interval <= 0:
        interval = total

    trained_since_compact = 0
    concurrency = max(config.batch_concurrency, 1)

    with (
        tqdm(total=total, desc="训练进度", unit="篇") as pbar,
        concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        future_map: dict[concurrent.futures.Future[Any], str] = {}

        def _submit(
            item: tuple[str, dict[str, Any] | None],
        ) -> concurrent.futures.Future[Any]:
            key, rec = item
            if rec is not None:
                future = executor.submit(
                    _train_one_record, config, llm_client, neo4j_client, rec
                )
            else:
                future = executor.submit(
                    _train_one_paper, config, llm_client, neo4j_client, key
                )
            future_map[future] = key
            return future

        def _process_result(future: concurrent.futures.Future[Any]) -> None:
            nonlocal trained_since_compact
            key = future_map[future]
            try:
                result = future.result()
                if result.get("already_trained"):
                    with lock:
                        skipped.append(key)
                    pbar.set_postfix_str(f"= {key[:40]}")
                else:
                    with lock:
                        succeeded.append(key)
                        trained_since_compact += 1
                    pbar.set_postfix_str(f"✓ {key[:40]}")
            except Exception as e:
                _logger.error("论文 %s 训练失败: %s", key, e)
                with lock:
                    failed.append({"论文": key, "错误": str(e)})
                pbar.set_postfix_str(f"✗ {key[:40]}")
            pbar.update(1)

        pending_queue = list(tasks)
        running: set[concurrent.futures.Future[Any]] = set()

        for _ in range(min(concurrency, len(pending_queue))):
            running.add(_submit(pending_queue.pop(0)))

        while running:
            done, running = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                _process_result(future)

            while len(running) < concurrency and pending_queue:
                running.add(_submit(pending_queue.pop(0)))

            if trained_since_compact >= interval:
                pbar.set_description("训练进度（等待进行中训练完成）")
                if running:
                    drained, running = concurrent.futures.wait(running)
                    for future in drained:
                        _process_result(future)

                _logger.info(
                    "已完成 %d 篇新训练，开始压缩知识图谱 …",
                    trained_since_compact,
                )
                pbar.set_description("训练进度（压缩中）")
                compact_report = compact_all(neo4j_client, config)
                print(json.dumps(compact_report, ensure_ascii=False, indent=2))
                pbar.set_description("训练进度")
                trained_since_compact = 0

                while len(running) < concurrency and pending_queue:
                    running.add(_submit(pending_queue.pop(0)))

    _logger.info("全部训练完成，执行最终压缩 …")
    final_report = compact_all(neo4j_client, config)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))

    result = {
        "总论文数": total,
        "成功数": len(succeeded),
        "跳过数": len(skipped),
        "失败数": len(failed),
        "成功列表": succeeded,
        "跳过列表": skipped,
        "失败列表": failed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
