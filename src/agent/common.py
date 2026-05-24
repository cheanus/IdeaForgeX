"""训练与推理共享工具。"""

from __future__ import annotations

import logging
from typing import Any

from src.models import (
    GRANULARITY_MAX,
    GRANULARITY_MIN,
    Edge,
    LLMACandidate,
    InspirationNode,
    NodeUpdate,
    QuestionNode,
    RelationType,
)

_logger = logging.getLogger("ideaforgex")


def parse_query_text(payload: dict[str, Any]) -> dict[str, str]:
    """解析 LLM 生成的检索查询文本。"""
    return {"query_text": payload.get("query_text", "")}


def _coerce_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, str):
                records.append({"id": item})
        return records
    if isinstance(value, dict):
        records = []
        for key, item in value.items():
            if isinstance(item, dict):
                record = dict(item)
                record.setdefault("id", str(key))
                records.append(record)
            else:
                records.append({"id": str(key), "核心描述": str(item)})
        return records
    return []


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def _coerce_granularity(value: Any) -> int:
    """将 LLM 返回的粒度值强制转为 1-3 范围内的整数。"""
    if value is None:
        return GRANULARITY_MIN
    if isinstance(value, bool):
        return GRANULARITY_MIN
    if isinstance(value, int):
        return max(GRANULARITY_MIN, min(GRANULARITY_MAX, value))
    if isinstance(value, float):
        return max(GRANULARITY_MIN, min(GRANULARITY_MAX, int(value)))
    if isinstance(value, str):
        try:
            return max(GRANULARITY_MIN, min(GRANULARITY_MAX, int(value)))
        except (ValueError, TypeError):
            _logger.warning(f"粒度值 '{value}' 无法转为整数，回退为 {GRANULARITY_MIN}")
            return GRANULARITY_MIN
    return GRANULARITY_MIN


def parse_llm_a_candidate(payload: dict[str, Any]) -> LLMACandidate:
    inspiration_nodes = []
    for item in _coerce_records(payload.get("inspiration_nodes", [])):
        granularity_input = (
            item.get("粒度")
            if item.get("粒度") is not None
            else item.get("granularity", 0)
        )
        inspiration_nodes.append(
            {
                "id": str(item.get("id", "")),
                "核心描述": item.get("核心描述") or item.get("content", ""),
                "向量": [],
                "粒度": _coerce_granularity(granularity_input),
                "前提条件": item.get("前提条件", ""),
                "操作步骤": item.get("操作步骤", ""),
                "已知实例": "",
            }
        )

    question_nodes = []
    for item in _coerce_records(payload.get("question_nodes", [])):
        question_nodes.append(
            {
                "id": str(item.get("id", "")),
                "核心描述": item.get("核心描述") or item.get("content", ""),
                "向量": [],
                "问题类型": item.get("问题类型")
                or item.get("question_type", "理论缺口"),
                "当前现状": item.get("当前现状", ""),
                "未解决部分": item.get("未解决部分", ""),
            }
        )

    edges = []
    for item in _coerce_list_of_dicts(payload.get("edges", [])):
        from_id = item.get("from_id") or item.get("source", "")
        to_id = item.get("to_id") or item.get("target", "")
        edges.append(
            {
                "from_id": str(from_id),
                "to_id": str(to_id),
                "rel_type": item.get("rel_type")
                or item.get("relation", "INSP_QUESTION"),
                "weight": item.get("weight", 0.0),
            }
        )

    node_updates = []
    for item in _coerce_list_of_dicts(payload.get("node_updates", [])):
        upd: dict[str, Any] = {
            "node_id": str(item.get("node_id", "")),
        }
        for field in NodeUpdate.model_fields:
            if field == "node_id":
                continue
            if field in item and item[field] is not None:
                upd[field] = item[field]
        node_updates.append(upd)

    return LLMACandidate.model_validate(
        {
            "can_infer": payload.get("can_infer", False),
            "query_text": payload.get("query_text", ""),
            "inspiration_nodes": inspiration_nodes,
            "question_nodes": question_nodes,
            "edges": edges,
            "node_updates": node_updates,
        }
    )


def validate_and_fix_refinement_edges(
    edges: list[Edge],
    new_inspirations: list[InspirationNode],
    retrieved_nodes: list[dict[str, Any]],
) -> list[Edge]:
    """校验 INSP_REFINES 边的粒度递进约束，违规时自动桥接或降级。

    规则：INSP_REFINES 必须从粒度 N 指向 N+1。违反时：
    - 若 retrieved_nodes 中存在中间粒度节点 → 拆分为两条合规边
    - 否则 → 将 rel_type 降级为 INSP_COMBINES，记录警告日志
    """
    gran_map: dict[str, int] = {}
    for n in new_inspirations:
        gran_map[n.id] = n.粒度
    for item in retrieved_nodes:
        node = item.get("node", {})
        if isinstance(node, dict) and "id" in node and "粒度" in node:
            gran_map[node["id"]] = node["粒度"]

    fixed_edges: list[Edge] = []
    for edge in edges:
        if edge.rel_type != RelationType.insp_refines:
            fixed_edges.append(edge)
            continue

        from_g = gran_map.get(edge.from_id)
        to_g = gran_map.get(edge.to_id)
        if from_g is None or to_g is None:
            fixed_edges.append(edge)
            continue

        if to_g == from_g + 1:
            fixed_edges.append(edge)
            continue

        _logger.warning(
            f"INSP_REFINES 粒度跳跃 {edge.from_id}(g={from_g}) -> {edge.to_id}(g={to_g})，"
            f"期望 {from_g} -> {from_g + 1}"
        )

        if to_g > from_g + 1:
            bridge_id: str | None = None
            for gap in range(from_g + 1, to_g):
                for nid, ng in gran_map.items():
                    if ng == gap and nid != edge.from_id and nid != edge.to_id:
                        bridge_id = nid
                        break
                if bridge_id:
                    break

            if bridge_id:
                _logger.warning(
                    f"  找到中间节点 {bridge_id}(g={gran_map[bridge_id]})，"
                    f"拆分为 {edge.from_id} -> {bridge_id} -> {edge.to_id}"
                )
                fixed_edges.append(
                    Edge(
                        from_id=edge.from_id,
                        to_id=bridge_id,
                        rel_type=RelationType.insp_refines,
                        weight=edge.weight,
                    )
                )
                fixed_edges.append(
                    Edge(
                        from_id=bridge_id,
                        to_id=edge.to_id,
                        rel_type=RelationType.insp_refines,
                        weight=edge.weight,
                    )
                )
            else:
                _logger.warning("  未找到中间粒度节点，降级为 INSP_COMBINES")
                fixed_edges.append(
                    Edge(
                        from_id=edge.from_id,
                        to_id=edge.to_id,
                        rel_type=RelationType.insp_combines,
                        weight=edge.weight,
                    )
                )
        else:
            _logger.warning("  粒度非递增，降级为 INSP_COMBINES")
            fixed_edges.append(
                Edge(
                    from_id=edge.from_id,
                    to_id=edge.to_id,
                    rel_type=RelationType.insp_combines,
                    weight=edge.weight,
                )
            )

    return fixed_edges
