"""训练与推理共享工具。"""

from __future__ import annotations

from typing import Any

from src.models import Edge, LLMACandidate, InspirationNode, NodeUpdate, QuestionNode


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


def parse_llm_a_candidate(payload: dict[str, Any]) -> LLMACandidate:
    inspiration_nodes = []
    for item in _coerce_records(payload.get("inspiration_nodes", [])):
        inspiration_nodes.append(
            {
                "id": str(item.get("id", "")),
                "核心描述": item.get("核心描述") or item.get("content", ""),
                "向量": item.get("向量") or item.get("embedding", []),
                "粒度": (
                    item.get("粒度")
                    if item.get("粒度") is not None
                    else item.get("granularity", 0)
                ),
                "前提条件": item.get("前提条件", ""),
                "操作步骤": item.get("操作步骤", ""),
                "已知实例": item.get("已知实例", ""),
            }
        )

    question_nodes = []
    for item in _coerce_records(payload.get("question_nodes", [])):
        question_nodes.append(
            {
                "id": str(item.get("id", "")),
                "核心描述": item.get("核心描述") or item.get("content", ""),
                "向量": item.get("向量") or item.get("embedding", []),
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
