# src/neo4j — 图数据库操作约定

## 连接管理

```python
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
```

在 `client.py` 中封装。其他模块通过该 client 操作 Neo4j，不直接实例化 `GraphDatabase.driver`。

## 事务模式

所有写入使用**写入事务**：

```python
def create_inspiration(tx, node: InspirationNode) -> None:
    tx.run("""
        CREATE (n:Inspiration {
            id: $id, 粒度: $granularity,
            核心描述: $description, 向量: $embedding,
            前提条件: $precondition, 操作步骤: $steps,
            已知实例: $examples
        })
    """, id=node.id, granularity=node.粒度, ...)

with client.driver.session() as session:
    session.execute_write(create_inspiration, node)
```

读取使用**读取事务**：

```python
def get_node_by_id(tx, node_id: str):
    result = tx.run("MATCH (n {id: $id}) RETURN n", id=node_id)
    return result.single()
```

## 向量索引管理

`schema.py` 在首次连接时幂等创建约束和索引：

```python
def ensure_schema(driver):
    """幂等：已存在则跳过"""
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

> `IF NOT EXISTS` 是 Neo4j 5.18+ 语法。如果 Neo4j 版本低于此，改用 `SHOW INDEXES` + 条件创建。

## 向量查询

```python
def vector_search(tx, index_name: str, query_embedding: list[float], k: int):
    result = tx.run("""
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
    """, index=index_name, k=k, embedding=query_embedding)
    return [{"node": r["node"], "score": r["score"]} for r in result]
```

## 遍历查询

按 `doc/architecture.md` §7 的 5 阶段算法实现。关键约束：

- 精化链展开用 `UNION` 双向查询，不限深度。
- 1-hop/2-hop 扩展用 `ORDER BY r.weight DESC LIMIT $M` 取 top-3。
- 分数衰减在 Python 侧计算，不走 Cypher。

## 节点更新

LLM 可通过 `node_updates` 增量修改已有节点属性。不可更新 `id`、`核心描述`、`向量`。

```python
def update_node(tx, update: NodeUpdate) -> None:
    mutable_fields = ["粒度", "前提条件", "操作步骤", "已知实例",
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

def batch_update(tx, updates: list[NodeUpdate]) -> None:
    for upd in updates:
        update_node(tx, upd)
```

在 `commit_candidates` 中，`batch_update` 必须在 `batch_write` 之前执行（先更新后新增）。

## 边创建

无向边在 Neo4j 中存为单条有向边，方向不重要（遍历用 `()-[]-()` ）。

```python
def create_edge(tx, from_id: str, to_id: str, rel_type: str, weight: float):
    tx.run(f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[:{rel_type} {{weight: $weight}}]->(b)
    """, from_id=from_id, to_id=to_id, weight=weight)
```

> `rel_type` 由 `models.py` 的枚举常量提供，不手写字符串。

## 批量操作

大量节点/边的写入使用 `execute_write` + 单个事务内循环：

```python
def batch_create(tx, nodes: list[InspirationNode], edges: list[Edge]):
    for node in nodes:
        create_inspiration(tx, node)  # 同上
    for edge in edges:
        create_edge(tx, edge.from_id, edge.to_id, edge.rel_type, edge.weight)
```

不要每条边开一个新事务——会严重拖慢性能。

## 文件职责

| 文件 | 职责 |
|---|---|
| `client.py` | `Neo4jClient` 连接管理 |
| `schema.py` | 约束/索引创建 + 节点/边写入 + 节点更新 |
| `retrieval.py` | 向量搜索 + 5 阶段遍历 + 去重截断 |
| `maintenance.py` | `clear_graph` 全量清图 + `resolve_target_uri` 端口映射 |
