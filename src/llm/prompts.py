"""LLM prompt 构造。"""

from __future__ import annotations

from src.models import Edge, InnovationIdea, InspirationNode, QuestionNode


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


def build_llm_a_judge_messages(paper_text: str, practice_summary: str) -> list[dict[str, str]]:
    system = (
        "你是论文创新点分析器。请判断论文是否已经足以直接记录到论文库；"
        "如果不能，输出可执行的 JSON，包含候选 Inspiration、Question 和边。"
    )
    user = f"论文内容：\n{paper_text}\n\n范式库：\n{chr(10).join(PARADIGMS)}\n\n实践库概要：\n{practice_summary}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_llm_b_generate_messages(paper_text: str, retrieved_context: str) -> list[dict[str, str]]:
    system = "你是创新点生成器。请给出候选创新点列表，输出 JSON。"
    user = f"论文内容：\n{paper_text}\n\n检索上下文：\n{retrieved_context}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_llm_c_eval_messages(paper_title: str, paper_text: str, idea: InnovationIdea | dict) -> list[dict[str, str]]:
    system = "你是创新点评估器。请从可行性、新颖性、价值三维度给出结构化 JSON。"
    user = f"论文标题：{paper_title}\n\n论文内容：\n{paper_text}\n\n候选创新点：\n{idea}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def node_to_text(node: InspirationNode | QuestionNode) -> str:
    return node.model_dump_json()


def edges_to_text(edges: list[Edge]) -> str:
    return "\n".join(edge.model_dump_json() for edge in edges)
