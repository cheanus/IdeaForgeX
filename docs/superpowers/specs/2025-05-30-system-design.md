# 系统设计规格

## 1. 概述

IdeaForgeX 是一个 Agent 框架，训练时通过 LLM 从论文中提炼可复用的方法灵感（Inspiration）和研究问题（Question），写入 Neo4j 知识图谱；推理时外部 agent 通过 CLI 检索图数据，自行编排创新点生成。

## 2. 核心设计决策

- **双节点类型**：实践库包含 `Inspiration`（方法/模式）和 `Question`（问题/缺口），平等参与语义检索。
- **论文节点**：`Paper` 节点存储论文元数据，通过 `PAPER_CONTRIBUTES` 边与灵感/问题双向关联，替代独立的 SQLite 论文库。
- **多粒度精化链**：每个方法概念以 N 个 Inspiration 节点表示（N ≥ 1），`粒度` 整数字段（1=抽象范式/2=通用方法/3=技术实现）区分层级，精化边（有向，低→高，严格 N→N+1 递进）串联。
- **边附权重**：每条边附带 `weight: float`，LLM 给定 0~1 分，用于检索排序。
- **论文来源**：AMiner API 发现全网论文 + 获取摘要（付费，成本极低）；arXiv API 仅作全文备选（免费）。
- **配置驱动**：Embedding 维度、检索 top-k、hop 上限、分数衰减等变量全部可配置，不在代码中硬编码。

## 3. 三大库

### 3.1 实践库

Neo4j 异构图，存储 Inspiration、Question 和 Paper 三种节点及五种边。向量检索通过 HNSW 索引。

训练时 LLM 可对已有节点进行增量属性修改（如更新当前现状、补充操作步骤）。不可修改节点的 ID、核心描述。

### 3.2 论文去重

通过 Neo4j `Paper` 节点的 uniqueness 约束实现。训练前检查 `Paper` 节点是否存在，训练完成后 `MERGE` 创建 Paper 节点。无独立 SQLite 库。

### 3.3 失败库

当前 A-only 版本不写失败库。

## 4. Neo4j 数据模型

### 4.1 Inspiration（方法灵感）

代表一个被抽象化、泛化后的方法模式/技术思想。

| 属性 | 含义 |
|---|---|
| id | 唯一标识，`insp-{uuid}` |
| 粒度 | 1=抽象范式, 2=通用方法, 3=技术实现 |
| 核心描述 | 该粒度的描述文本，也是 embedding 源 |
| 向量 | embedding 输出，驱动向量检索 |
| 前提条件 | 适用场景 |
| 操作步骤 | 转化为创新的步骤 |
| 已知实例 | 体现该模式的论文，可缺省 |

### 4.2 Question（研究问题）

代表一个未被完全解决的研究缺口/问题。

| 属性 | 含义 |
|---|---|
| id | 唯一标识，`q-{uuid}` |
| 核心描述 | 一句话描述，也是 embedding 源 |
| 向量 | embedding 输出 |
| 问题类型 | 理论缺口 / 工程瓶颈 / 评估缺失 / 跨领域空白 |
| 当前现状 | 已有方法解决的部分 |
| 未解决部分 | 具体未解决的问题 |

### 4.3 节点更新规则

训练时 LLM 可对已有节点进行增量属性修改。

- 可更新：`粒度`、`前提条件`、`操作步骤`（Inspiration）；`问题类型`、`当前现状`、`未解决部分`（Question）
- 不可更新：`id`、`核心描述`

### 4.3 Paper（论文元数据）

论文的元数据容器，不参与语义检索。

| 属性 | 含义 |
|---|---|
| id | 唯一标识，`paper-{paper_id}` |
| title | 论文标题 |
| year | 发表年份 |
| abstract | 摘要文本（前 2000 字符） |
| trained_at | 训练完成的 ISO 时间戳 |

### 4.4 五种边

| 边 | 类型 | 方向 | 含义 |
|---|---|---|---|
| 灵感精化边 | `INSP_REFINES` | 有向 | 低粒度→高粒度（N→N+1，严格递进），同方法概念的纵向展开 |
| 灵感组合边 | `INSP_COMBINES` | 无向 | 两种不同方法组合产生新方向 |
| 灵感问题边 | `INSP_QUESTION` | 无向 | 某方法可驱动某问题的解决 |
| 问题组合边 | `QUESTION_COMBINES` | 无向 | 两问题交集定义新的、更有趣的研究缺口 |
| 论文贡献边 | `PAPER_CONTRIBUTES` | 有向 | Paper → 实践节点，记录论文对节点的贡献关系 |

### 4.5 向量索引

在 `Inspiration.向量` 和 `Question.向量` 上各建立一个余弦相似度 HNSW 向量索引，维度由 embedding 模型配置决定。启动时幂等创建（`IF NOT EXISTS`）。

## 5. 检索遍历

5 阶段策略：向量搜索（Inspiration + Question 混合 top-k）→ 精化链双向展开（不限深）→ BFS 图扩展（可配深度，每层分数衰减）→ 去重按分数截断。

扩展开销由 `k_hits`、`max_neighbors`、`max_depth`、`score_decay`、`final_k` 控制，全部可配置。

## 6. 训练流程

7 节点 StateGraph，2 次 LLM 调用，2 处条件边（去重检查 + 提炼判断）：

```
load_paper → check_duplicate → {skip | generate_query → retrieve
  → llm_a_judge → commit_candidates} → END
```

写入顺序：先 `MATCH+SET` 更新已有节点，再 `CREATE` 新节点和边。

并发安全：`check_duplicate` 通过 `CREATE Paper` + uniqueness constraint 实现原子预留，两个并发训练同一论文时第二个触发 `ConstraintError` 直接跳过。节点 ID 通过 `paper_id` 前缀天然隔离。

## 7. 推理流程

推理不在 IdeaForgeX 内部完成。外部 agent 通过 CLI 命令编排创新点生成：

```
外部 agent 加载论文 → CLI retrieve → CLI inspect → agent 调 LLM 生成创新点
```

CLI 只负责查询（`retrieve` / `inspect` / `random` / `relate`）。范式调用、创新点生成、文献查重、多轮纠偏全部由外部 agent 完成。详见 CLI 规格文档。

## 8. 技术架构

### 8.1 技术栈

| 层次 | 选型 |
|---|---|
| Agent 框架 | LangGraph |
| LLM SDK | openai |
| 图数据库 | Neo4j Community |
| 论文来源 | AMiner API (付费) + arXiv (免费备选) |
| 向量检索 | Neo4j HNSW 向量索引 |
| 数据验证 | pydantic |
| 包管理 | uv |
| Python | >=3.11 |

### 8.2 项目结构

```
IdeaForgeX/
├── docs/
│   └── superpowers/
│       └── specs/           # 设计规格文档
├── src/
│   ├── main.py              # CLI 入口
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── agent/               # LangGraph 训练工作流
│   ├── neo4j/               # 图数据库操作
│   ├── llm/                 # LLM 调用与 prompt
│   ├── paper/               # 论文获取与解析
│   └── cli/                 # CLI 查询命令
└── tests/
```

各子模块编码规范见对应 `src/*/CONVENTIONS.md`。

### 8.3 数据流

```
AMiner API (发现+摘要) → 论文内容 → LangGraph Agent (LLM A) → OpenAI 兼容 API
  → Neo4j (Inspiration×N + Question 节点 + 4种边 + HNSW向量索引)
```

### 8.4 LLM 角色

V1 仅保留 LLM A。训练用 2 次 LLM 调用：

| 角色 | 职责 | 阶段 |
|---|---|---|
| LLM A (查询提炼) | 从论文提取 1-2 句检索查询 | 训练 |
| LLM A (判断/生成) | 判断已有节点修改/新增，生成节点和边 | 训练 |

## 9. 错误处理

- LLM JSON 解析失败：自动重试，超过 `max_retries` 终止
- Neo4j 连接失败：抛异常退出，不静默
- AMiner API 失败：重试 3 次，仍失败跳过该论文
- arXiv 全文获取失败：降级为仅用摘要
- 向量索引不存在：启动时幂等创建
