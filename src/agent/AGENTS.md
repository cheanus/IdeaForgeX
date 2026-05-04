# src/agent — LangGraph 工作流约定

## 状态定义

每个工作流定义一个 `TypedDict` State：

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class TrainingState(TypedDict):
    paper_text: str
    paper_id: str
    can_infer: bool
    inspirations: list  # InspirationNode
    questions: list  # QuestionNode
    edges: list  # Edge
    messages: Annotated[list, add_messages]
```

## StateGraph 模式

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(TrainingState)

# 节点：同步函数，接收 state 返回 dict（部分更新）
def node_extract(state: TrainingState) -> dict:
    text = extract_pdf(state["paper_id"])
    return {"paper_text": text}

builder.add_node("extract", node_extract)
builder.add_node("llm_a_judge", node_llm_a)
builder.add_conditional_edges("llm_a_judge", route_after_a, {
    "record_paper": "record_paper",
    "commit_candidates": "commit_candidates"
})
```

## 节点编写规则

- 每个节点是纯函数：`(state) -> dict`，返回需要更新的字段。
- 不要在节点内直接读写文件——通过 `paper/extractor.py` 或 `neo4j/` 模块。
- LLM 调用通过 `llm/client.py` 的 `ChatClient`，不直接调 `openai`。
- 节点函数签名中 state 参数使用完整的 State TypedDict 类型注解。

## RetryPolicy

```python
from langgraph.types import RetryPolicy

builder.add_node("llm_a_judge", node_llm_a, retry_policy=RetryPolicy(max_attempts=3))
```

仅 LLM 调用节点需要 RetryPolicy。确定性节点（如 `extract`、`neo4j_write`）不需要。

## Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": paper_id}}

# 执行
for event in graph.stream(initial_state, config):
    ...
```

用 `thread_id = paper_id` 保证每篇论文独立 checkpoint。

## 文件组织

- `training.py`：训练 StateGraph 定义 + 节点函数
- `inference.py`：推理 StateGraph 定义 + 节点函数
- 共享的辅助函数放在 `src/agent/common.py`

## 测试

- 对每个节点函数单独测试：mock LLM/Neo4j，验证 state 变换正确。
- 对条件边测试：给定典型 state，验证路由到正确下游。
