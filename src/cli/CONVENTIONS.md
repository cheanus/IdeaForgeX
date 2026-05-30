# src/cli — CLI 查询命令约定

## 概述

CLI 暴露知识图谱的查询层。生成创新点、多轮交互、论文发现等均由外部 agent 框架调用 CLI 命令自行编排。

四个查询命令：

| 命令 | 语义 | 适用场景 |
|---|---|---|---|
| `retrieve` | 广度优先向量+图检索 | agent 拿到候选列表展示给用户 |
| `inspect` | 深度优先钻取 | agent 在用户选中某节点后深入理解上下文 |
| `random` | 随机探索 | agent 做头脑风暴时引入意外发现 |
| `relate` | 路径查询 | agent 想知道两个节点之间有何联系 |
| `delete-node` | 删除节点 + 级联清理 | 维护：删除论文/灵感/问题 |

## 输出格式

所有输出为 JSON，通过 `print()` 输出到 stdout，供外部 agent 解析。

### 边类型标签

| RelationType | CLI 输出标签 |
|---|---|
| `INSP_COMBINES` | 灵感组合边 |
| `INSP_QUESTION` | 灵感问题边 |
| `QUESTION_COMBINES` | 问题组合边 |

精化边（`INSP_REFINES`）通过 `chain` 字段内联展示，不出现于 `edges` 列表中。

### retrieve 输出

瘦身格式，仅含节点识别信息和检索质量元数据。agent 根据 `core_description` 和 `source` 判断哪些节点值得深入：

- `id` / `type` / `score` / `source` / `granularity` / `core_description` — 节点核心字段
- `source` — 来源标记：`vector`(向量命中)、`chain`(精化链展开)、`1-hop`/`2-hop`(图扩展)
- `meta.total_hits` — 检索命中总数

### inspect 输出

全字段展开：节点所有属性完整输出；`chain` 内联精化链（按粒度升序，`direction` 标注 self/coarser/finer）；边的 `target` 展开为 mini-inspect。

### random 输出

与 `retrieve` 瘦身格式一致，`source` 标记为 `random` 或 `random-weighted`，顶层多一个 `mode` 字段。

### relate 输出

`connected` 布尔值 + 路径节点序列（id / type / core_description）+ 边序列 + `hops`。无路径时返回 `reason`。

## 文件职责

`queries.py` 包含 `cmd_retrieve` / `cmd_inspect` / `cmd_random` / `cmd_relate` 函数实现 + 输出格式化。

## 实现约定

- 格式化逻辑与数据获取分离。数据获取复用 `src/neo4j/retrieval.py`。
- 节点 type 由 Neo4j 标签确定（`labels(n)[0]`），不依赖属性字段。
- 精化链按粒度升序排列，`direction` 通过比较粒度确定。
- `random` 命令通过 `random_nodes()` 处理纯随机和主题加权两种模式。

## 重试与错误

- CLI 命令本身不内置重试，由外部 agent 决定重试策略。
- 节点不存在时 `inspect` 返回 `{"id": "...", "error": "节点不存在"}`。
