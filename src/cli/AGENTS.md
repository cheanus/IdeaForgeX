# src/cli — CLI 查询命令约定

## 概述

CLI 暴露知识图谱的查询层。生成创新点、多轮交互、论文发现等均由外部 agent 框架调用 CLI 命令自行编排。

两个核心命令：

| 命令 | 语义 | 适用场景 |
|---|---|---|
| `retrieve` | 广度优先扫射 | agent 拿到候选列表展示给用户 |
| `inspect` | 深度优先钻取 | agent 在用户选中某节点后深入理解上下文 |

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

检索结果包含折叠视图，减少 agent 首次解析的 token 量：

- `core_description`：节点核心描述，agent 第一眼看节点靠它
- `snippet`：折叠次要字段，供 agent 判断方向价值
- `chain`：内联精化链，`direction` 标注 self/coarser/finer
- `edges.target_summary`：边目标仅给摘要字符串，agent 感兴趣再 `inspect`
- `meta.total_hits`：检索命中总数，agent 据此判断检索质量

### inspect 输出

节点/边全字段展开：

- 节点所有属性完整输出
- 边的 `target` 展开为 mini-inspect（id / type / core_description / 关键字段）

## 文件职责

| 文件 | 职责 |
|---|---|
| `queries.py` | `cmd_retrieve` / `cmd_inspect` 函数实现 + 输出格式化 |

## 实现约定

- 格式化逻辑与数据获取分离。数据获取复用 `src/neo4j/retrieval.py`。
- 边查询使用批量 Cypher，避免 N+1 查询。
- `snippet` 字段根据节点类型选择字段集（Inspiration: 前提条件/操作步骤/已知实例；Question: 问题类型/当前现状/未解决部分）。
- 精化链按粒度升序排列，`direction` 通过比较粒度确定。

## 重试与错误

- CLI 命令本身不内置重试，由外部 agent 决定重试策略。
- 节点不存在时 `inspect` 返回 `{"id": "...", "error": "节点不存在"}`。
