"""LLM prompt 构造。"""

from __future__ import annotations

from typing import Any

_JSON_NODE_FORMAT = "每个元素：id, 核心描述, 向量(留空[]即可，系统自动生成), 粒度(可选, 整数 1-3: 1=抽象范式, 2=通用方法, 3=技术实现), 前提条件(可选), 操作步骤(可选), 已知实例(可选)"
_JSON_QUESTION_FORMAT = "每个元素：id, 核心描述, 向量(留空[]即可，系统自动生成), 问题类型(可选), 当前现状(可选), 未解决部分(可选)"
_JSON_EDGE_FORMAT = (
    "rel_type 必须为：INSP_REFINES, INSP_COMBINES, INSP_QUESTION, QUESTION_COMBINES。"
    "INSP_REFINES 必须从粒度 N 指向 N+1（N 为 1 或 2），不可跳跃"
)


def _format_retrieved_nodes(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return ""
    lines: list[str] = []
    for item in nodes:
        n = item.get("node", {})
        if not n:
            continue
        score = item.get("score", 0)
        node_id = n.get("id", "")
        desc = n.get("核心描述") or n.get("core", "")
        grain_or_type = (
            f"粒度 {n.get('粒度')}" if "粒度" in n else n.get("问题类型", "")
        )
        lines.append(f"  {node_id} (score={score:.2f}) [{grain_or_type}] {desc}")
    return "\n".join(lines)


def build_query_generation_messages(
    paper_text: str,
) -> list[dict[str, str]]:
    """让 LLM 提炼论文的检索查询文本。"""
    system = (
        "你是一位论文检索专家。根据论文内容，提炼 1-2 句语义检索查询文本，"
        "用于在实践库中查找相关的方法灵感（Inspiration）和研究问题（Question）。"
        "查询应聚焦论文的核心方法思想或待解决的开放问题，去除具体模型名和数据集名。"
        '只输出一个 JSON：{"query_text": "..."}。'
    )
    user = f"论文内容：\n{paper_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_llm_a_judge_messages(
    paper_text: str,
    practice_summary: str,
    retrieved_nodes: list[dict],
) -> list[dict[str, str]]:
    """训练阶段：LLM A 判断论文是否可直接推断，或提取新知识 + 更新已有节点。"""
    retrieved_text = _format_retrieved_nodes(retrieved_nodes)
    if retrieved_text:
        retrieved_text = (
            f"检索到的相关实践库节点（按语义相似度排序）：\n{retrieved_text}"
        )

    system = (
        "你是论文创新点分析器。请判断该论文是否已经可以直接记录到论文库；"
        "如果不能，请从论文中提取新的灵感节点、问题节点、关系边，"
        "并根据检索到的相关节点决定是否需要更新已有节点的属性。\n\n"
        "输出格式（一个 JSON）：\n"
        "  - can_infer (bool)：论文是否可直接推断（无需提取新知识）\n"
        f"  - inspiration_nodes (数组)：新发现的灵感节点\n    {_JSON_NODE_FORMAT}\n"
        f"  - question_nodes (数组)：新发现的问题节点\n    {_JSON_QUESTION_FORMAT}\n"
        "  - edges (数组)：节点间的关系边\n"
        "    每个元素：from_id, to_id, rel_type, weight\n"
        f"    {_JSON_EDGE_FORMAT}\n"
        "  - node_updates (数组)：需要更新属性的已有节点\n"
        "    每个元素：node_id（必填）, 以及需要修改的字段（粒度/前提条件/操作步骤/已知实例/问题类型/当前现状/未解决部分，仅填变化的字段）\n"
        "    不可更新：id, 核心描述, 向量\n\n"
        "规则：\n"
        "  - 若论文提出的是已有节点的新信息（如新的适用场景、新的已知实例），用 node_updates 而非新建节点\n"
        "  - 若论文提出了与检索结果完全不同的新概念，创建新节点\n"
        "  - ID 不能与检索结果中的节点 ID 重复\n"
        '  - 如果无法生成有效 JSON，返回 {"error": "原因"}'
    )
    user = (
        f"论文内容：\n{paper_text}\n\n"
        f"{retrieved_text}\n\n"
        f"实践库概要：\n{practice_summary}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



