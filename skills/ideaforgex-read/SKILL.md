---
name: ideaforgex-read
description: "Query IdeaForgeX knowledge graph (inspirations, questions, papers) via HTTP API for research ideation."
version: 1.0.0
author: cheanus
metadata:
  hermes:
    tags: [Research, Knowledge Graph, Idea Generation, Academic]
---

# IdeaForgeX 知识图谱只读查询

## 前置条件

必须设置环境变量 `API_ENDPOINT`，指向 IdeaForgeX 的 FastAPI 服务地址：

```bash
export API_ENDPOINT="http://localhost:2048"   # 本机
export API_ENDPOINT="https://ideaforgex.example.com"  # 远程
```

以下所有命令均使用 `$API_ENDPOINT` 作为基础 URL。返回值均为 JSON。

## 端点一览

| 端点 | 用途 |
|------|------|
| `GET /` | 健康检查 |
| `POST /retrieve` | 语义检索，返回相关灵感和问题 |
| `GET /inspect/{ids}` | 查看节点全字段详情（支持逗号分隔多 ID） |
| `GET /stats` | 图谱规模统计 |
| `GET /random` | 随机探索，打破检索偏差 |
| `GET /relate/{from}/{to}` | 两节点间最短路径 |

所有端点为**只读**。不暴露训练、删除、压缩等写操作。

## 知识图谱数据模型

图中有三类节点：

| 类型 | 字段 | 说明 |
|------|------|------|
| `Inspiration` | `core_description`、`粒度`(1/2/3)、`前提条件`、`操作步骤` | 方法灵感，粗→细三层粒度 |
| `Question` | `core_description`、`问题类型`、`当前现状`、`未解决部分` | 研究问题 |
| `Paper` | `title`、`year`、`abstract` | 来源论文 |

边类型：`INSP_COMBINES`（灵感组合）、`INSP_QUESTION`（灵感→问题）、`QUESTION_COMBINES`（问题组合）、`INSP_REFINES`（精化链，同灵感不同粒度间）、`PAPER_CONTRIBUTES`（论文→贡献节点）。

## 端点详解

### `POST /retrieve` — 语义检索

传入自然语言查询，返回匹配的灵感和问题。这是创新点探索的**主入口**。

```bash
curl -s -X POST "$API_ENDPOINT/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "使用扩散模型做医学图像分割的少样本学习"}'
```

**参数**（JSON body）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | string | **必填** | 查询文本 |
| `top_k` | int | 8 | 向量命中数 |
| `expand_hops` | int | 1 | BFS 扩展深度 |
| `max_per_node` | int | 2 | 每节点扩展邻居上限 |
| `decay` | float | 0.5 | 分数衰减因子 |
| `final_limit` | int | 15 | 最终截断数 |

**返回格式**：

```json
{
  "query": "...",
  "nodes": [
    {
      "id": "insp-3f2a...",
      "type": "Inspiration",
      "score": 0.92,
      "source": "vector",
      "granularity": 1,
      "core_description": "将生成式先验注入判别式模型的少样本学习范式"
    }
  ],
  "meta": {"total_hits": 15, "runtime_ms": 85}
}
```

`source` 含义：`vector`=向量直接命中（最可靠）、`chain`=精化链展开、`1-hop`/`2-hop`=图扩展（相关性递减）。agent 应据此加权信任度。

### `GET /inspect/{ids}` — 节点详情

从 retrieve 结果中拿到 `id` 后，深入查看节点的完整信息。

```bash
# 单个节点
curl -s "$API_ENDPOINT/inspect/insp-3f2a..."

# 多个节点（逗号分隔）
curl -s "$API_ENDPOINT/inspect/insp-001,insq-002"

# 跳过边展开（仅返回节点 + 精化链）
curl -s "$API_ENDPOINT/inspect/insp-3f2a...?expand_edges=false"
```

**查询参数**：`expand_edges`，默认 `true`。设为 `false` 可减少响应体积。

**返回格式**（数组，每元素一个节点）：

```json
[{
  "node": {
    "id": "insp-3f2a...",
    "type": "Inspiration",
    "granularity": 1,
    "core_description": "...",
    "前提条件": "...",
    "操作步骤": "...",
    "papers": [{"id": "paper-1706.03762", "title": "Attention Is All You Need", "year": "2017"}]
  },
  "chain": [
    {"id": "...", "granularity": 0, "core_description": "...", "direction": "coarser"},
    {"id": "...", "granularity": 2, "core_description": "...", "direction": "finer"}
  ],
  "edges": [
    {
      "type": "灵感组合边",
      "target": {"id": "insp-c77f", "type": "Inspiration", "core_description": "多尺度特征对齐", "granularity": 2},
      "weight": 0.83
    }
  ]
}]
```

`chain` 展示同一灵感在不同粒度下的上下游（coarser=更抽象、finer=更具体），有助于理解灵感层次结构。

### `GET /stats` — 图谱统计

了解图谱当前规模，辅助判断检索覆盖面。

```bash
curl -s "$API_ENDPOINT/stats"
```

**返回**：

```json
{
  "nodes": {"inspiration_total": 3882, "question_total": 1017, "paper_total": 1572},
  "inspiration_granularity": {"粒1": 845, "粒2": 1492, "粒3": 1545},
  "question_types": {"工程瓶颈": 709, "理论缺口": 207},
  "edges": {"PAPER_CONTRIBUTES": 6332, "INSP_REFINES": 3037},
  "paper_years": {"2021": 1514, "2022": 58}
}
```

### `GET /random` — 随机探索

打破检索排序的路径依赖，引入意外发现。适合头脑风暴初期。

```bash
# 全图均匀随机
curl -s "$API_ENDPOINT/random?count=5"

# 主题加权随机（在相关范围内抽样）
curl -s "$API_ENDPOINT/random?count=3&query=跨模态注意力"
```

**查询参数**：`count`(默认 5)、`query`(可选，有则主题加权)。

**返回**：

```json
{
  "mode": "random",
  "nodes": [
    {"id": "insp-7a2b", "type": "Inspiration", "source": "random", "granularity": 2, "core_description": "多模态融合中的注意力对齐"}
  ],
  "meta": {"total_hits": 5, "runtime_ms": 12}
}
```

### `GET /relate/{from}/{to}` — 路径查询

查找两个节点在图中的关联路径，用于理解概念之间的连接关系。

```bash
curl -s "$API_ENDPOINT/relate/insp-001/insp-002"
curl -s "$API_ENDPOINT/relate/insp-001/q-xyz?max_len=4"
```

**查询参数**：`max_len`(默认 6，最大跳数)。

**返回**：

```json
{
  "connected": true,
  "hops": 2,
  "nodes": [
    {"id": "insp-001", "type": "Inspiration", "core_description": "起点", "granularity": 1},
    {"id": "q-001", "type": "Question", "core_description": "中间问题", "granularity": null},
    {"id": "insp-002", "type": "Inspiration", "core_description": "终点", "granularity": 2}
  ],
  "edges": [
    {"type": "INSP_QUESTION", "weight": 0.8},
    {"type": "QUESTION_COMBINES", "weight": 0.6}
  ],
  "meta": {"runtime_ms": 5}
}
```

无路径时返回 `{"connected": false, "reason": "无路径"}`。

## 典型工作流

```
1. /stats          → 了解图谱规模，判断覆盖面
2. /retrieve       → 输入研究想法，获取相关灵感和问题
3. /inspect/{ids}  → 对感兴趣的结果深入查看全字段 + 精化链 + 关联边
4. /random         → 引入意外发现，拓宽思路
5. /relate/{a}/{b} → 理解两个概念之间的连接路径
```

## 推荐技能

完成知识图谱探索后，建议结合以下技能深化创新点生成：

- **[brainstorming-research-ideas](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/21-research-ideation/brainstorming-research-ideas)** — 基于检索到的灵感和问题，系统评估并深化研究想法
- **[creative-thinking-for-research](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/21-research-ideation/creative-thinking-for-research)** — 创造性发散思考，突破思维定式，发掘跨领域连接
