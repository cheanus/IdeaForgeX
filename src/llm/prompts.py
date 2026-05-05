"""LLM prompt 构造。"""

from __future__ import annotations


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


def build_llm_a_judge_messages(
    paper_text: str, practice_summary: str
) -> list[dict[str, str]]:
    system = (
        "你是论文创新点分析器。请判断论文是否已经足以直接记录到论文库；"
        '如果无法生成有效 JSON，返回 {"error": "原因"}。'
        "只输出一个 JSON，其中包含 can_infer、inspiration_nodes、question_nodes 和 edges。"
        "inspiration_nodes 与 question_nodes 必须是数组，每个元素必须包含 id、核心描述、向量。"
        "edges 必须是数组，每个元素必须包含 from_id、to_id、rel_type、weight。"
        "rel_type 必须是以下枚举值之一（不含引号）："
        "INSP_REFINES（一个灵感细化为另一个灵感）、"
        "INSP_COMBINES（两个灵感组合）、"
        "INSP_QUESTION（灵感引出问题）、"
        "QUESTION_COMBINES（两个问题组合）。"
    )
    user = f"论文内容：\n{paper_text}\n\n范式库：\n{chr(10).join(PARADIGMS)}\n\n实践库概要：\n{practice_summary}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
