## 概述

CLI 仅暴露知识图谱的**查询层**。生成创新点、多轮交互、论文发现等均由外部 agent 框架（如 Hermes Agent）调用 CLI 命令自行编排。

两个核心命令：

| 命令 | 语义 | 适用场景 |
|---|---|---|
| `retrieve` | 广度优先扫射 | agent 拿到候选列表展示给用户 |
| `inspect` | 深度优先钻取 | agent 在用户选中某节点后深入理解上下文 |

---

## `retrieve` — 图检索

### 输入

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 查询文本（论文摘要 / 一句话想法 / 关键词） |
| `top_k` | int | 否 | 向量命中数，默认 10 |
| `expand_hops` | int | 否 | 非精化边最大扩展深度，默认 2 |
| `max_per_node` | int | 否 | 每节点扩展上限，默认 3 |
| `decay` | float | 否 | 分数衰减因子，默认 0.7 |
| `final_limit` | int | 否 | 最终截断数，默认 20 |

### 输出

```json
{
  "query": "使用扩散模型做医学图像分割的少样本学习",
  "nodes": [
    {
      "id": "insp-3f2a1...",
      "type": "Inspiration",
      "score": 0.92,
      "granularity": 1,
      "core_description": "将生成式先验注入判别式模型的少样本学习范式",
      "snippet": {
        "前提条件": "有大规模预训练生成模型可用",
        "操作步骤": "1) 将预训练扩散模型的去噪过程作为特征提取器...",
        "已知实例": "DDPMSeg (MICCAI 2023)"
      },
      "chain": [
        {"id": "insp-8b4c...", "granularity": 0, "core_description": "生成先验注入判别模型", "direction": "coarser"},
        {"id": "insp-3f2a...", "granularity": 1, "core_description": "生成式先验注入判别式模型的少样本学习范式", "direction": "self"},
        {"id": "insp-1d9e...", "granularity": 2, "core_description": "StableDiffusion特征金字塔+UNet分割头", "direction": "finer"}
      ],
      "edges": [
        {"type": "灵感组合边", "target": "insp-c77f...", "weight": 0.83, "target_summary": "多尺度特征对齐"},
        {"type": "灵感问题边", "target": "q-4a2e...", "weight": 0.76, "target_summary": "医学图像标注成本高导致少样本过拟合"}
      ]
    }
  ],
  "meta": {
    "total_hits": 47,
    "expansion_hops": 2,
    "decay_factor": 0.7,
    "runtime_ms": 120
  }
}
```

### 输出字段说明

| 字段 | 说明 |
|---|---|
| `nodes[].core_description` | 该粒度描述文本，agent 第一眼看节点就靠它 |
| `nodes[].snippet` | 折叠次要字段（前提条件 / 操作步骤 / 已知实例），供 agent 判断方向价值 |
| `nodes[].chain` | 内联精化链（按粒度升序），包含当前节点及其上下游，`direction` 标注 `self` / `coarser` / `finer` |
| `nodes[].edges` | 1-hop 扩展边，`target_summary` 仅给摘要不展开，agent 感兴趣再 `inspect` |
| `meta.total_hits` | 检索命中总数，agent 据此判断检索质量（过低时可提示用户换查询词） |

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

## agent 编排示意

```
用户：「我有篇论文摘要……帮我找创新方向」
  → agent 调 retrieve(query=摘要) → 拿到 top-20 节点
  → agent 展示节点列表给用户确认/过滤
  → 用户：「第三个节点的精化链里更细那个具体怎么做？」
  → agent 调 inspect(id=insp-1d9e...) → 展开全字段
  → 用户：「把节点 A 和 B 组合一下，用类比推理范式」
  → agent 自己调 LLM 基于两个节点的全字段信息生成候选创新点
  → 循环，直到满意
```

CLI 只负责查询。范式调用、创新点生成、文献查重、多轮纠偏——全部由外部 agent 完成。
