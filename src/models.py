"""核心数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# 粒度常量：1=抽象范式, 2=通用方法, 3=技术实现
Granularity = Literal[1, 2, 3]
GRANULARITY_LABELS: dict[int, str] = {1: "抽象范式", 2: "通用方法", 3: "技术实现"}
GRANULARITY_MIN: Literal[1] = 1
GRANULARITY_MAX: Literal[3] = 3


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
    type: str = NodeType.inspiration.value
    粒度: Granularity = 1
    前提条件: str = ""
    操作步骤: str = ""
    已知实例: str = ""


class QuestionNode(BaseGraphNode):
    type: str = NodeType.question.value
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


class NodeUpdate(BaseModel):
    node_id: str
    粒度: Granularity | None = None
    前提条件: str | None = None
    操作步骤: str | None = None
    已知实例: str | None = None
    问题类型: str | None = None
    当前现状: str | None = None
    未解决部分: str | None = None


class LLMACandidate(BaseModel):
    can_infer: bool = False
    query_text: str = ""
    inspiration_nodes: list[InspirationNode] = Field(default_factory=list)
    question_nodes: list[QuestionNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    node_updates: list[NodeUpdate] = Field(default_factory=list)
