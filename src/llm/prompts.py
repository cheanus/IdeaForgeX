"""LLM prompt 构造。"""

from __future__ import annotations

from typing import Any

_JSON_NODE_FORMAT = "每个元素：id, 核心描述, 粒度(可选, 整数 1-3: 1=抽象范式, 2=通用方法, 3=技术实现), 前提条件(可选), 操作步骤(可选)"
_JSON_QUESTION_FORMAT = (
    "每个元素：id, 核心描述, 问题类型(可选), 当前现状(可选), 未解决部分(可选)"
)
_JSON_EDGE_FORMAT = (
    "rel_type 必须为：INSP_REFINES, INSP_COMBINES, INSP_QUESTION, QUESTION_COMBINES。"
)


def _format_retrieved_nodes(nodes: list[dict[str, Any]], detailed: bool = False) -> str:
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
        line = f"  {node_id} (score={score:.2f}) [{grain_or_type}] {desc}"
        if detailed:
            extras: list[str] = []
            for field in ("前提条件", "操作步骤", "当前现状", "未解决部分"):
                val = n.get(field, "")
                if val:
                    extras.append(f"{field}={val}")
            if extras:
                line += " | " + " | ".join(extras)
        lines.append(line)
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
    paper_title: str = "",
    paper_year: str = "",
) -> list[dict[str, str]]:
    """训练阶段：LLM A 判断论文是否可直接推断，或提取新知识 + 更新已有节点。

    注意 can_infer 语义（容易误解）：
      True  = 论文可直接推断（无需提取新知识）→ 应用层应跳过节点提交
      False = 需要提取新节点 → 应用层应将 inspiration_nodes/question_nodes/edges 写入 Neo4j
    """
    retrieved_text = _format_retrieved_nodes(retrieved_nodes, detailed=True)
    if retrieved_text:
        retrieved_text = (
            f"检索到的相关实践库节点（按语义相似度排序）：\n{retrieved_text}"
        )

    system = (
        "你是论文创新点分析器。请判断该论文是否已经可以直接记录到论文库；"
        "如果不能，请从论文中提取灵感节点、问题节点和关系边，并根据检索结果决定是否更新已有节点。\n\n"
        "输出要求（硬性约束，必须满足）：\n"
        "  1. 灵感节点（Inspiration）：论文的核心方法必须覆盖 3 个粒度层级 —\n"
        "     粒度 1（抽象范式）：上升到方法论层面的高度抽象，如「自监督预训练→下游微调范式」「生成先验注入判别模型」\n"
        "     粒度 2（通用方法）：具体的技术方法组合，如「对比学习+masked auto-encoding 联合预训练」\n"
        "     粒度 3（技术实现）：可操作的技术细节，如「EMA 动量更新 target encoder + stop-grad」\n"
        "     * 若检索到的已有节点已覆盖某些粒度，通过 INSP_REFINES 边串联它们，只新建缺失的粒度节点\n"
        "     * 若已有节点核心概念匹配但前提条件/操作步骤等字段需要补充当前论文中的新信息，用 node_updates 补充而非新建\n"
        "     * 一个论文可有多组独立的方法概念，但每组至少覆盖 3 个粒度\n"
        "  2. 问题节点（Question）：至少生成 1 个该论文试图解决或未完全解决的研究问题 —\n"
        "     问题类型须从以下选择：理论缺口 / 工程瓶颈 / 评估缺失 / 跨领域空白\n"
        "     当前现状写「已有方法解决了什么」，未解决部分写「还缺什么/为什么没解决」\n"
        "     * 若检索到的 Question 节点核心概念恰好匹配，可直接在 edges 中引用，不必新建\n"
        "     * 若匹配节点的当前现状/未解决部分需要补充本论文的视角，通过 node_updates 追加\n"
        "  3. 关系边：每个新节点至少与一个已有或新建节点建立关系 —\n"
        "     INSP_REFINES：串联同组方法概念的不同粒度节点（N→N+1，严格递进，不可跳跃）\n"
        "     INSP_COMBINES：不同方法概念可组合产生新方向\n"
        "     INSP_QUESTION：方法节点驱动某问题的解决\n"
        "     QUESTION_COMBINES：两个问题的交集定义新的研究缺口\n"
        "    权重 0.0~1.0，表示关联强度\n\n"
        "输出格式（一个 JSON）：\n"
        "  - can_infer (bool)：论文是否可直接推断（无需提取新知识）\n"
        f"  - inspiration_nodes (数组)：新发现的灵感节点\n    {_JSON_NODE_FORMAT}\n"
        f"  - question_nodes (数组)：新发现的问题节点\n    {_JSON_QUESTION_FORMAT}\n"
        "  - edges (数组)：节点间的关系边\n"
        "    每个元素：from_id, to_id, rel_type, weight\n"
        f"    {_JSON_EDGE_FORMAT}\n"
        "  - node_updates (数组)：需要更新属性的已有节点\n"
        "    每个元素：node_id（必填）, 以及需要修改的字段（粒度/前提条件/操作步骤/问题类型/当前现状/未解决部分，仅填变化的字段）\n"
        "    不可更新：id, 核心描述\n\n"
        "规则：\n"
        "  - 优先复用检索结果中的已有节点：能通过边串联就不新建，能通过 node_updates 补充就不新建\n"
        "  - 若论文提出了与检索结果完全不同的新概念，创建新节点\n"
        "  - 新节点 ID 不能与检索结果中的节点 ID 重复\n"
        '  - 如果无法生成有效 JSON，返回 {"error": "原因"}'
    )
    user = (
        f"论文元信息：{paper_title} ({paper_year})\n\n"
        f"论文内容：\n{paper_text}\n\n"
        f"{retrieved_text}\n\n"
        f"实践库概要：\n{practice_summary}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
