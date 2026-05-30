## 概述

CLI 暴露知识图谱的**训练层**和**查询层**。训练时通过 `bootstrap` / `train` 构建知识图谱，推理时外部 agent 调用查询命令自行编排创新点生成。

命令一览：

| 命令 | 类别 | 语义 |
|---|---|---|
| `bootstrap` | 管理 | 初始化 Neo4j schema（约束 + 向量索引） |
| `reset` | 管理 | 清空实践库（删除所有 Inspiration/Question 节点） |
| `train` | 训练 | 解析论文 → LLM 提炼灵感/问题 → 写入知识图谱 |
| `retrieve` | 查询 | 广度优先向量+图检索，返回候选列表 |
| `inspect` | 查询 | 深度钻取单个节点的全字段 + 精化链 + 边详情 |
| `random` | 查询 | 随机探索，打破检索排序路径依赖 |
| `relate` | 查询 | 查找两节点间最短路径 |
| `stats` | 管理 | 知识图谱统计信息（待接入） |

---

## `bootstrap` — 初始化

```bash
ideaforgex bootstrap
```

幂等操作：创建 `Inspiration` / `Question` 标签的 uniqueness 约束，以及 `idx_insp_vector` / `idx_q_vector` 两个余弦向量索引。已存在则跳过。

---

## `reset` — 清空实践库

```bash
ideaforgex reset
```

`DETACH DELETE` 所有 Inspiration 和 Question 节点及其关联边，同时清空 SQLite 论文库。**不可逆操作。**

---

## `train` — 论文训练

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `paper` | string | ✅ | 论文 ID（arXiv ID / AMiner ID）或标题，支持多级降级解析 |

### 流程

1. **获取论文** — arXiv ID 直查 → arXiv 标题搜索（含 PDF 全文降级） → AMiner 语义搜索
2. **去重检查** — 查询 SQLite 论文库，已训练则跳过
3. **生成检索查询** — LLM 从论文文本中提炼核心研究问题
4. **图检索** — 向量搜索 + 图遍历，查找相关节点
5. **LLM 分析** — 判断论文是否可提炼新灵感/问题，并生成候选节点 + 边 + 节点更新
6. **提交** — 写入新节点和边，更新已有节点
7. **标记已训练** — 写入 SQLite 论文库

### 输出

无标准输出（写入 Neo4j）。通过 `LOG_LEVEL=INFO` 查看阶段日志。

---

## `stats` — 图谱统计

```bash
ideaforgex stats
```

（待接入）计划输出：Inspiration 数、Question 数、总边数、各粒度分布。

---

---

## `retrieve` — 图检索

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | 查询文本（论文摘要 / 一句话想法 / 关键词） |
| `top_k` | int | 否 | 向量命中数，默认 8 |
| `expand_hops` | int | 否 | 非精化边最大扩展深度，默认 1 |
| `max_per_node` | int | 否 | 每节点扩展上限，默认 2 |
| `decay` | float | 否 | 分数衰减因子，默认 0.5 |
| `final_limit` | int | 否 | 最终截断数，默认 15 |

### 输出

`retrieve` 返回精简视图，仅含节点识别信息和检索质量元数据。agent 根据 `core_description` 和 `source` 判断哪些节点值得深入，再通过 `inspect` 获取全字段 + 精化链 + 边详情。

```json
{
  "query": "使用扩散模型做医学图像分割的少样本学习",
  "nodes": [
    {
      "id": "insp-3f2a1...",
      "type": "Inspiration",
      "score": 0.92,
      "source": "vector",
      "granularity": 1,
      "core_description": "将生成式先验注入判别式模型的少样本学习范式"
    },
    {
      "id": "insp-1d9e...",
      "type": "Inspiration",
      "score": 0.87,
      "source": "chain",
      "granularity": 2,
      "core_description": "StableDiffusion特征金字塔+UNet分割头"
    },
    {
      "id": "insp-c77f...",
      "type": "Inspiration",
      "score": 0.46,
      "source": "1-hop",
      "granularity": 2,
      "core_description": "多尺度特征对齐"
    }
  ],
  "meta": {
    "total_hits": 15,
    "runtime_ms": 85
  }
}
```

### 输出字段说明

| 字段 | 说明 |
|---|---|
| `nodes[].id` | 节点唯一 ID，供 `inspect` 钻取 |
| `nodes[].type` | 节点类型：`Inspiration` 或 `Question` |
| `nodes[].score` | 检索得分，用于 agent 排序判断相关性 |
| `nodes[].source` | 来源：`vector`(向量命中)、`chain`(精化链展开)、`1-hop`/`2-hop`(图扩展)。agent 据此加权信任度 |
| `nodes[].granularity` | 仅 Inspiration：方法粒度（0=高层次范式，1=具体方法，2=实例化技术） |
| `nodes[].core_description` | 该节点核心描述文本，agent 第一眼看节点就靠它 |
| `meta.total_hits` | 检索命中总数，agent 据此判断检索质量 |
| `meta.runtime_ms` | 检索耗时（毫秒） |

---

## `random` — 随机探索

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `count` | int | 否 | 返回节点数，默认 5 |
| `query` | string | 否 | 有则主题加权随机（向量检索 top 50 后随机抽样），无则全图均匀随机 |

### 输出

与 `retrieve` 瘦身格式一致，`source` 标记为 `random` 或 `random-weighted`。

```json
{
  "mode": "random",
  "nodes": [
    {
      "id": "insp-7a2b...",
      "type": "Inspiration",
      "source": "random",
      "granularity": 2,
      "core_description": "多模态融合中的注意力对齐"
    }
  ],
  "meta": {
    "total_hits": 5,
    "runtime_ms": 12
  }
}
```

> `random` 的目的是打破检索排序的路径依赖，为 brainstorm 引入意外发现。agent 应把随机结果视为"灵感种子"而非精确答案。

---

## `inspect` — 节点详情

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✅ | 节点 ID（支持逗号分隔多个） |
| `expand_edges` | bool | 否 | 是否展开边目标节点详情，默认 true |

### 输出

```json
{
  "node": {
    "id": "insp-3f2a1...",
    "type": "Inspiration",
    "granularity": 1,
    "core_description": "将生成式先验注入判别式模型的少样本学习范式",
    "前提条件": "有大规模预训练生成模型可用；目标域与预训练域有分布偏移时需微调",
    "操作步骤": "1) 将预训练扩散模型的去噪过程作为特征提取器 2) 通过时间步条件注入分割头 3) 联合训练判别损失+生成一致性损失",
    "已知实例": "DDPMSeg (MICCAI 2023), DiffuseIR (arXiv 2024)"
  },
  "chain": [
    {"id": "insp-8b4c...", "granularity": 0, "core_description": "生成先验注入判别模型", "direction": "coarser"},
    {"id": "insp-3f2a...", "granularity": 1, "core_description": "生成式先验注入判别式模型的少样本学习范式", "direction": "self"},
    {"id": "insp-1d9e...", "granularity": 2, "core_description": "StableDiffusion特征金字塔+UNet分割头", "direction": "finer"}
  ],
  "edges": [
    {
      "type": "灵感组合边",
      "target": {"id": "insp-c77f...", "type": "Inspiration", "core_description": "多尺度特征对齐", "granularity": 2},
      "weight": 0.83
    },
    {
      "type": "灵感问题边",
      "target": {"id": "q-4a2e...", "type": "Question", "core_description": "医学图像标注成本高导致少样本过拟合", "问题类型": "工程瓶颈"},
      "weight": 0.76
    },
    {
      "type": "问题组合边",
      "target": {"id": "q-b33f...", "type": "Question", "core_description": "分割模型在域偏移下的泛化崩溃", "问题类型": "理论缺口"},
      "weight": 0.68
    }
  ]
}
```

### 与 `retrieve` 的差异

| | `retrieve` | `inspect` |
|---|---|---|
| 节点字段 | 仅 `id` + `type` + `score` + `source` + `core_description`（瘦身） | 全字段展开（前提条件 / 操作步骤 / 已知实例 / 问题类型 / 当前现状 / 未解决部分） |
| `chain` | ❌ | ✅ 精化链（coarser / self / finer） |
| `edges` | ❌ | ✅ 展开为 mini-inspect（target 含 id / type / core_description / 关键字段） |
| `score` | ✅ | ❌（不关心如何被搜出） |

---

## `relate` — 路径查询

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id_a` | string | ✅ | 起始节点 ID |
| `id_b` | string | ✅ | 目标节点 ID |
| `max_len` | int | 否 | 最长路径跳数，默认 6 |

### 输出

```json
{
  "connected": true,
  "hops": 2,
  "nodes": [
    {"id": "insp-001", "type": "Inspiration", "core_description": "起点方法", "granularity": 1},
    {"id": "q-001", "type": "Question", "core_description": "中间问题", "granularity": null},
    {"id": "insp-002", "type": "Inspiration", "core_description": "终点方法", "granularity": 2}
  ],
  "edges": [
    {"type": "INSP_QUESTION", "weight": 0.8},
    {"type": "QUESTION_COMBINES", "weight": 0.6}
  ],
  "meta": {"runtime_ms": 5}
}
```

| 情况 | 输出 |
|---|---|
| 路径存在 | `connected: true` + `hops` + `nodes` + `edges` |
| 无路径 | `connected: false` + `reason` |
| 同一节点 | `connected: true` + `hops: 0` + `node` |

> 路径上中间节点仅出 `id/type/core_description`（与 `retrieve` 瘦身风格一致），agent 感兴趣再 `inspect`。

---

## agent 编排示意

```
# 训练阶段（运维）
$ ideaforgex bootstrap          # 首次初始化
$ ideaforgex train 1706.03762   # 训练 Attention Is All You Need
$ ideaforgex train ViT           # 训练 ViT
$ ideaforgex stats               # 查看图谱规模

# 推理阶段（agent 编排）
用户：「我有篇论文摘要……帮我找创新方向」
  → agent 调 retrieve(query=摘要) → 拿到 top-15 节点（仅 id + core_description + source）
  → agent 展示节点列表给用户确认/过滤
  → 用户：「第三个节点的精化链里更细那个具体怎么做？」
  → agent 调 inspect(id=insp-1d9e...) → 展开全字段

用户：「我还想看一些意外的关联」
  → agent 调 random(count=5) → 拿到 5 个随机节点作为灵感种子
  → agent 调 random(query="domain generalization") → 在相关范围内随机探索

用户：「节点 A 和节点 B 之间有什么联系？」
  → agent 调 relate(id_a=..., id_b=...) → 最短路径 + 中间节点

用户：「把节点 A 和 B 组合一下，用类比推理范式」
  → agent 调 inspect(id=A), inspect(id=B) → 获取全字段
  → agent 自己调 LLM 基于两个节点的全字段信息生成候选创新点
  → 循环，直到满意
```

CLI 只负责训练和查询。范式调用、创新点生成、文献查重、多轮纠偏——全部由外部 agent 完成。
