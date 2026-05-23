## 概述

CLI 仅暴露知识图谱的**查询层**。生成创新点、多轮交互、论文发现等均由外部 agent 框架（如 Hermes Agent）调用 CLI 命令自行编排。

两个核心命令：

| 命令 | 语义 | 适用场景 |
|---|---|---|---|
| `retrieve` | 广度优先扫射 | agent 拿到候选列表展示给用户 |
| `inspect` | 深度优先钻取 | agent 在用户选中某节点后深入理解上下文 |
| `random` | 随机探索 | agent 做头脑风暴时引入意外发现 |
| `relate` | 路径查询 | agent 想知道两个节点之间有何联系 |

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
| 节点字段 | `core_description` + 折叠 `snippet` | 全字段展开 |
| 边的 `target` | 仅 `target_summary`（字符串） | 展开为 mini-inspect（id / type / core_description / 关键字段） |
| `chain` 格式 | 统一列表（同格式） | 统一列表（同格式） |
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
用户：「我有篇论文摘要……帮我找创新方向」
  → agent 调 retrieve(query=摘要) → 拿到 top-15 节点（仅 id + core_description + source）
  → agent 展示节点列表给用户确认/过滤
  → 用户：「第三个节点的精化链里更细那个具体怎么做？」
  → agent 调 inspect(id=insp-1d9e...) → 展开全字段
  → 用户：「把节点 A 和 B 组合一下，用类比推理范式」
  → agent 自己调 LLM 基于两个节点的全字段信息生成候选创新点
  → 循环，直到满意
```

CLI 只负责查询。范式调用、创新点生成、文献查重、多轮纠偏——全部由外部 agent 完成。
