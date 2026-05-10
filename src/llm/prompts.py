"""LLM prompt 构造。"""

from __future__ import annotations

from typing import Any

PARADIGMS = [
    "二分关联：连接两个不相关的参照系",
    "问题重构：换一种方式表述问题",
    "类比推理：深层因果结构迁移",
    "约束操控：探索/组合/转换三类创造",
    "否定翻转：否定核心假设",
    "抽象阶梯：泛化/特化/类比三个方向",
    "相邻可能：在可触及边界上创新",
    "双面思维：同时持有矛盾，超越对立",
]

_JSON_NODE_FORMAT = "每个元素：id, 核心描述, 向量, 粒度(可选, 整数 0-5: 0=抽象范式, 1=通用方法, 2=算法策略, 3=技术实现, 4=工程细节, 5=具体实例), 前提条件(可选), 操作步骤(可选), 已知实例(可选)"
_JSON_QUESTION_FORMAT = (
    "每个元素：id, 核心描述, 向量, 问题类型(可选), 当前现状(可选), 未解决部分(可选)"
)
_JSON_EDGE_FORMAT = (
    "rel_type 必须为：INSP_REFINES, INSP_COMBINES, INSP_QUESTION, QUESTION_COMBINES"
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


def build_inference_messages(
    paper_text: str,
    retrieved_nodes: list[dict],
) -> list[dict[str, str]]:
    """推理阶段：LLM A 基于检索结果生成创新点候选。"""
    retrieved_text = _format_retrieved_nodes(retrieved_nodes)
    if retrieved_text:
        retrieved_text = f"检索到的相关实践库节点：\n{retrieved_text}"

    system = (
        "你是论文创新点生成器。基于目标论文和实践库中检索到的相关方法灵感和研究问题，"
        "提出潜在的高价值 AI 研究创新点。\n\n"
        "请运用范式库中的认知框架，结合检索到的现有方法/问题，"
        "生成新的灵感节点、问题节点和关系边。\n\n"
        "输出格式（一个 JSON）：\n"
        "  - can_infer (bool)：固定为 false\n"
        f"  - inspiration_nodes (数组)：生成的灵感节点（代表新方法/技术）\n    {_JSON_NODE_FORMAT}\n"
        f"  - question_nodes (数组)：生成的问题节点（代表未解决的缺口）\n    {_JSON_QUESTION_FORMAT}\n"
        "  - edges (数组)：节点间的关系边\n"
        "    每个元素：from_id, to_id, rel_type, weight\n"
        f"    {_JSON_EDGE_FORMAT}\n\n"
        "规则：\n"
        "  - ID 可以引用检索结果中已有节点的 ID（表示与已有知识关联）\n"
        "  - 新生成的节点 ID 不能与检索结果中的节点 ID 重复\n"
        '  - 如果无法生成有效 JSON，返回 {"error": "原因"}'
    )
    user = (
        f"目标论文：\n{paper_text}\n\n"
        f"范式库：\n{chr(10).join(PARADIGMS)}\n\n"
        f"{retrieved_text}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
