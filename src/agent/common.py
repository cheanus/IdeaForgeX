"""训练与推理共享工具。"""

from __future__ import annotations

from typing import Any

from src.models import Edge, LLMACandidate, InspirationNode, QuestionNode


def parse_llm_a_candidate(payload: dict[str, Any]) -> LLMACandidate:
    return LLMACandidate.model_validate(payload)


def dump_nodes(inspirations: list[InspirationNode], questions: list[QuestionNode], edges: list[Edge]) -> dict[str, Any]:
    return {
        "inspirations": [node.model_dump() for node in inspirations],
        "questions": [node.model_dump() for node in questions],
        "edges": [edge.model_dump() for edge in edges],
    }
