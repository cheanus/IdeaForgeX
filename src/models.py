"""核心数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    inspiration = "Inspiration"
    question = "Question"


class RelationType(str, Enum):
    insp_combines = "INSP_COMBINES"
    insp_question = "INSP_QUESTION"
    question_combines = "QUESTION_COMBINES"
    insp_refines = "INSP_REFINES"


class BaseGraphNode(BaseModel):
    id: str
    核心描述: str
    向量: list[float]

    @model_validator(mode="after")
    def _strip_id(self):
        self.id = self.id.strip()
        return self


class InspirationNode(BaseGraphNode):
    type: Literal[NodeType.inspiration.value] = NodeType.inspiration.value
    粒度: int = 0
    前提条件: str = ""
    操作步骤: str = ""
    已知实例: str = ""


class QuestionNode(BaseGraphNode):
    type: Literal[NodeType.question.value] = NodeType.question.value
    问题类型: str = "理论缺口"
    当前现状: str = ""
    未解决部分: str = ""


class Edge(BaseModel):
    from_id: str
    to_id: str
    rel_type: RelationType
    weight: float = Field(ge=0.0, le=1.0)


class InnovationIdea(BaseModel):
    title: str
    description: str
    feasibility_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    value_score: float = Field(ge=0.0, le=1.0)
    reflection: str = ""


class LLMACandidate(BaseModel):
    can_infer: bool = False
    inspiration_nodes: list[InspirationNode] = Field(default_factory=list)
    question_nodes: list[QuestionNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class LLMBCandidate(BaseModel):
    ideas: list[InnovationIdea] = Field(default_factory=list)


class LLMCEvaluation(BaseModel):
    passed: bool = False
    best_idea: InnovationIdea | None = None
    comment: str = ""
    failure_reason: str = ""


class FailureRecord(BaseModel):
    paper_id: str
    paper_title: str
    failure_reason: str
    llm_c_eval: dict[str, Any]
    candidate_idea_snapshot: dict[str, Any]


def generate_inspiration_id() -> str:
    return f"insp-{uuid4()}"


def generate_question_id() -> str:
    return f"q-{uuid4()}"
