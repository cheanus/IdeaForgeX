# src/neo4j — 图数据库操作约定

## 连接管理

`Neo4jClient` 在 `client.py` 中封装 `GraphDatabase.driver`。其他模块通过该 client 操作 Neo4j，不直接实例化 driver。

## 事务模式

- 写入：`session.execute_write(unit_of_work, *args)`
- 读取：`session.execute_read(unit_of_work, *args)`

## 向量索引管理

`schema.py` 在首次连接时幂等创建约束和索引（Neo4j 5.18+ `IF NOT EXISTS` 语法）：

```python
def ensure_schema(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT insp_id_unique IF NOT EXISTS ...")
        session.run("CREATE CONSTRAINT q_id_unique IF NOT EXISTS ...")
        session.run("""
            CREATE VECTOR INDEX idx_insp_vector IF NOT EXISTS
            FOR (n:Inspiration) ON (n.向量)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $embedding_dim,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        session.run("CREATE VECTOR INDEX idx_q_vector IF NOT EXISTS ...")
```

## 向量查询

使用 `SEARCH` 子句（`db.index.vector.queryNodes` 已弃用）：

```python
def vector_search(tx, index_name: str, query_embedding: list[float], k: int):
    # SEARCH 子句不支持将索引名作为参数，按索引名分支构造查询语句
    cypher = (
        """
        MATCH (node:Inspiration)
        SEARCH node IN (
            VECTOR INDEX idx_insp_vector
            FOR $embedding
            LIMIT $k
        ) SCORE AS score
        RETURN node, score
        ORDER BY score DESC
        """
        if index_name == "idx_insp_vector"
        else
        """
        MATCH (node:Question)
        SEARCH node IN (
            VECTOR INDEX idx_q_vector
            FOR $embedding
            LIMIT $k
        ) SCORE AS score
        RETURN node, score
        ORDER BY score DESC
        """
    )
    result = tx.run(cypher, k=k, embedding=query_embedding)
    return [{"node": r["node"], "score": r["score"]} for r in result]
```

## 实现备注

- **CALL 子查询作用域**：使用带作用域的子查询 `CALL (hit) { ... }`，避免 Neo4j 将无作用域的 `CALL { ... }` 标记为已弃用。
- **关系类型过滤**：采用通用关系模式 `-[r]-`，再通过 `WHERE type(r) IN [...]` 过滤。避免数据库尚无某种关系类型时收到 `relationship type does not exist` 警告。

## 遍历查询

按 `docs/superpowers/specs/2025-05-30-system-design.md` §5 的 5 阶段算法实现。关键约束：

- 精化链展开用 `UNION` 双向查询，不限深度。
- 1-hop/2-hop 扩展用 `ORDER BY r.weight DESC LIMIT $M` 取 top。
- 分数衰减在 Python 侧计算，不走 Cypher。

## 节点更新

LLM 可通过 `node_updates` 增量修改已有节点属性：

```python
def update_node(tx, update: NodeUpdate) -> None:
    mutable_fields = ["粒度", "前提条件", "操作步骤",
                      "问题类型", "当前现状", "未解决部分"]
    set_clauses = []
    params = {"node_id": update.node_id}
    for field in mutable_fields:
        value = getattr(update, field, None)
        if value is not None:
            set_clauses.append(f"n.{field} = ${field}")
            params[field] = value
    if not set_clauses:
        return
    tx.run(f"MATCH (n {{id: $node_id}}) SET {', '.join(set_clauses)}", **params)
```

不可更新 `id`、`核心描述`。`已知实例` 由代码自动注入。

在 `commit_candidates` 中，节点更新必须在节点新增之前执行。

## 已知实例追加

`已知实例` 不由 LLM 产出，由代码在 `commit_candidates` 中自动注入：

```python
def append_known_instance(tx, node_ids: list[str], entry: str) -> None:
    for node_id in node_ids:
        tx.run(
            """
            MATCH (n {id: $node_id})
            WHERE n.已知实例 IS NULL OR NOT n.已知实例 CONTAINS $entry
            SET n.已知实例 = coalesce(n.已知实例 + '; ', '') + $entry
            """,
            node_id=node_id,
            entry=entry,
        )
```

## 边创建

无向边在 Neo4j 中存为单条有向边，遍历用 `()-[]-()`。`rel_type` 由 `models.py` 的枚举常量提供，不手写字符串。大量节点/边的写入使用 `execute_write` + 单个事务内循环，不要每条边开新事务。

## 文件职责

| 文件 | 职责 |
|---|---|---|---|
| `client.py` | `Neo4jClient` 连接管理 |
| `schema.py` | 约束/索引创建（含 Paper uniqueness）+ 节点/边写入 + 节点更新 |
| `retrieval.py` | 向量搜索 + 5 阶段遍历 + 去重截断 |
| `compact.py` | 图压缩：numpy 批量余弦相似度 → 链约束过滤 → 并查集合并组 → 边转移+属性合并+节点删除 |
| `maintenance.py` | `clear_graph` 全量清图 + `delete_node_cascade` 级联删除 |
