# src/agent — LangGraph 工作流约定

## 状态定义

每个工作流定义一个 `TypedDict` State：

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class TrainingState(TypedDict):
    paper_id: str
    paper_title: str
    paper_year: str
    paper_text: str
    query_text: str          # LLM 生成的检索查询
    retrieved_nodes: list    # 图检索结果 [{node, score}]
    can_infer: bool
    already_trained: bool    # 去重标记
    llm_a: dict
    inspirations: list       # InspirationNode
    questions: list          # QuestionNode
    edges: list              # Edge
    node_updates: list       # NodeUpdate
    messages: Annotated[list, add_messages]
```

## StateGraph 模式

### 训练 (7 节点)

```
load_paper → check_duplicate → {skip | generate_query → retrieve
  → llm_a_judge → commit_candidates} → END
```

```python
builder = StateGraph(TrainingState)

builder.add_node("load_paper", load_paper)
builder.add_node("check_duplicate", check_duplicate)     # SQLite try_reserve
builder.add_node("generate_query", generate_query)       # LLM 调用 1
builder.add_node("retrieve", retrieve)                    # 无 LLM
builder.add_node("llm_a_judge", node_llm_a)               # LLM 调用 2
builder.add_node("commit_candidates", commit_candidates)
builder.add_node("skip", skip_training)

builder.add_edge(START, "load_paper")
builder.add_edge("load_paper", "check_duplicate")
builder.add_conditional_edges("check_duplicate", route_after_dup, {
    "skip": "skip",
    "generate_query": "generate_query",
})
builder.add_edge("skip", END)
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "llm_a_judge")
builder.add_conditional_edges("llm_a_judge", route_after_llm_a, {
    "commit_candidates": "commit_candidates",
})
builder.add_edge("commit_candidates", END)
```

### check_duplicate 去重

利用 `Paper` 节点的 uniqueness constraint 实现原子预留，支持并发训练：

```python
def check_duplicate(state):
    with neo4j_client.session() as session:
        try:
            session.run("CREATE (p:Paper {id: ...})", id=paper_node_id)
            return {"already_trained": False}
        except ConstraintError:
            return {"already_trained": True}
```

`CREATE` 只能成功一次——第二个并发进程触发 `ConstraintError`，直接跳过。`commit_candidates` 中再 `MATCH + SET` 更新占位 Paper 为真实数据。

### commit_candidates 写入顺序

1. 更新 Paper 节点（MATCH + SET，占位符已在 check_duplicate 创建）
2. 若 `node_updates`：batch_update + 创建 PAPER_CONTRIBUTES 边
3. 创建新节点（Inspiration/Question）+ 边 + PAPER_CONTRIBUTES 边
4. LLM 生成的节点 ID 加 paper_id 前缀防并发冲突

## 节点编写规则

- 每个节点是纯函数：`(state) -> dict`，返回需要更新的字段。
- 不要在节点内直接读写文件——通过 `paper/` 或 `neo4j/` 模块。
- LLM 调用通过 `llm/client.py` 的 `ChatClient`，不直接调 `openai`。
- 节点函数签名中 state 参数使用完整的 State TypedDict 类型注解。

## RetryPolicy

仅 LLM 调用节点需要，确定性节点不需要。通过 `langgraph.types.RetryPolicy(max_attempts=3)` 配置。

## Checkpointer

用 `MemorySaver`，`thread_id = paper_id` 保证每篇论文独立 checkpoint。

## 文件组织

- `training.py`：训练 StateGraph 定义 + 节点函数
- 共享的辅助函数放在 `src/agent/common.py`

## 测试

- 对每个节点函数单独测试：mock LLM/Neo4j，验证 state 变换正确。
- 对条件边测试：给定典型 state，验证路由到正确下游。
