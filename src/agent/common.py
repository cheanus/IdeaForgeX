"""训练与推理共享工具。"""

from __future__ import annotations

from typing import Any

from src.models import Edge, FailureRecord, LLMACandidate, LLMBCandidate, LLMCEvaluation, InspirationNode, InnovationIdea, QuestionNode


def parse_llm_a_candidate(payload: dict[str, Any]) -> LLMACandidate:
    return LLMACandidate.model_validate(payload)


def parse_llm_b_candidate(payload: dict[str, Any]) -> LLMBCandidate:
    if "ideas" not in payload and "innovation_ideas" in payload:
        payload = {"ideas": payload["innovation_ideas"]}
    return LLMBCandidate.model_validate(payload)


def parse_llm_c_evaluation(payload: dict[str, Any]) -> LLMCEvaluation:
    if "best_idea" in payload and isinstance(payload["best_idea"], dict):
        payload["best_idea"] = InnovationIdea.model_validate(payload["best_idea"])
    return LLMCEvaluation.model_validate(payload)


def build_failure_record(
    paper_id: str,
    paper_title: str,
    failure_reason: str,
    llm_c_eval: LLMCEvaluation | dict[str, Any],
    candidate_idea_snapshot: InnovationIdea | dict[str, Any],
) -> FailureRecord:
    return FailureRecord(
        paper_id=paper_id,
        paper_title=paper_title,
        failure_reason=failure_reason,
        llm_c_eval=llm_c_eval if isinstance(llm_c_eval, dict) else llm_c_eval.model_dump(),
        candidate_idea_snapshot=(candidate_idea_snapshot if isinstance(candidate_idea_snapshot, dict) else candidate_idea_snapshot.model_dump()),
    )


def dump_nodes(inspirations: list[InspirationNode], questions: list[QuestionNode], edges: list[Edge]) -> dict[str, Any]:
    return {
        "inspirations": [node.model_dump() for node in inspirations],
        "questions": [node.model_dump() for node in questions],
        "edges": [edge.model_dump() for edge in edges],
    }
