# 并行训练与图压缩 (Compact) 设计

## 概述

当前 `train` 命令单论文串行训练。本设计引入两样新功能：

1. **`batch-train`** — 多线程并行训练多篇论文，加速批量入图
2. **`compact`** — 合并图中语义相近的节点，解决并行训练窗口内产生的重复节点问题

核心策略：**optimistic 并行训练 + 定期异步纠错**。训练线程独立写图（利用现有 Neo4j 唯一性约束保证低层安全），compact 定期收敛重复节点。

## 问题场景

并行训练时，进程 A 训论文甲、进程 B 训论文乙。甲创建节点 X，而乙在 A 尚未提交 X 时检索图未命中 X，独立创建语义相近的节点 Y。compact 负责事后合并 X 与 Y。

## 数据模型约束

压缩必须保持以下语义约束：

- **细化链（INSP_REFINES）必须是树结构**：一条细化链对应一个方法概念，从粒度 1 向粒度 3 逐步细化。链内各粒度层不允许多个分支（一个节点只有一个上游 INSP_REFINES 父节点）。
- **合并范围仅限于同粒度**：Inspiration 节点只与同粒度节点合并，避免跨粒度合并破坏树结构和产生环。
- **合并充分条件**：两个同粒度 G 的 Inspiration 节点可合并，当且仅当它们各自细化链上所有上游祖先（粒度 1, 2, ..., G）均属于同一合并组。粒度 1 节点无上游祖先，仅凭自身相似度决定。算法自顶向下（先粒 1 后粒 2 再粒 3）执行，上游合并组已决定后才判定下游，transitive 地满足全链一致约束。

### 合并后语义

- 合并后的节点代表"同一方法概念在该粒度层次的统一表达"
- 被合并节点的所有边（INSP_REFINES、INSP_COMBINES、INSP_QUESTION、PAPER_CONTRIBUTES）转移到存活节点
- PAPER_CONTRIBUTES 边保留：被合并节点所关联的论文仍然记录为对合并后节点的贡献

## batch-train 命令

### CLI 接口

```bash
# 命令行直接列多篇
python main.py batch-train 2301.0001 2301.0002 2301.0003

# 从文件读取（每行一个论文查询）
python main.py batch-train --queries papers.txt

# 从 JSONL 文件读取（跳过 API 解析）
python main.py batch-train --jsonl papers.jsonl

# 混用
python main.py batch-train 2301.0001 --queries papers.txt
python main.py batch-train --queries papers.txt --jsonl papers.jsonl
```

### 执行流程

```
batch-train(papers):
    executor = ThreadPoolExecutor(max_workers=config.batch_concurrency)
    completed = 0
    running = queue of all papers

    while running or executor has active futures:
        if len(active_futures) < max_workers and running:
            paper = running.pop()
            submit train_one(paper)
        
        wait for any future to complete
        completed += 1
        
        if completed % config.compact_interval == 0:
            wait_all_active_futures_to_complete()
            compact_all(client, config)
    
    wait_all_futures()
    compact_all(client, config)  # 最终压缩
```

### 并发策略

- 每个 worker 线程运行一个完整的训练 `StateGraph`（`load_paper → ... → commit_candidates`）
- 线程间通过 `ThreadPoolExecutor` 管理，不直接通信
- compact 期间**暂停所有训练线程**：等当前进行中的训练全部完成后执行 compact，再恢复后续训练
- 利用现有的 Neo4j 唯一性约束保证 Paper 级去重

### 进度与错误

- 进度信息 `print()` 到 stdout：`[batch] 完成 5/20, 1 失败`
- 单篇训练失败不中断 batch，记录到 `failed_papers` 列表
- 错误日志通过 logger 输出到 stderr

### 输出格式

batch-train 完成后输出 JSON 到 stdout：

```json
{
  "总论文数": 20,
  "成功数": 18,
  "失败数": 2,
  "成功列表": ["2301.0001", "2301.0003", ...],
  "失败列表": [
    {"论文": "2301.0002", "错误": "LLM JSON 解析失败，已达最大重试次数"},
    {"论文": "2301.0005", "错误": "无法解析论文标识"}
  ]
}
```

## compact 命令

### CLI 接口

```bash
python main.py compact              # 全图压缩
python main.py compact --dry-run    # 仅报告，不执行
```

### 算法

```
compact_all(client, config):
    compact_inspirations(client, config.compact_threshold, config.compact_topk)
    compact_questions(client, config.compact_threshold, config.compact_topk)
    deduplicate_edges(client)
```

#### Inspiration 合并（自顶向下，按粒度分层）

```
compact_inspirations(client, threshold, topk):
    report = CompactReport()
    
    for granularity in [1, 2, 3]:
        # 1. 获取该粒度所有节点
        nodes = MATCH (n:Inspiration) WHERE n.粒度 = granularity RETURN n
        
        # 2. HNSW 近邻搜索，找出相似候选对
        pairs = []
        for node in nodes:
            neighbors = db.index.vector.queryNodes('idx_insp_vector', node.向量, k=topk)
            for neighbor in neighbors:
                if neighbor.粒度 == node.粒度 and cosine_sim > threshold:
                    pairs.append((node.id, neighbor.id, cosine_sim))
        
        # 3. 链约束过滤（粒度 2/3）
        if granularity > 1:
            parent_groups = get_parent_merge_groups(granularity - 1)
            pairs = [(a, b, sim) for (a, b, sim) in pairs
                     if parent_of(a) and parent_of(b) are in same parent_groups]
        
        # 4. 构建合并组（并查集连通分量）
        groups = connected_components(pairs)
        
        # 5. 对每个合并组：选存活节点，转移边，删除冗余
        for group in groups:
            survivor = pick_survivor(group)  # 选连接边最多的
            merge_node_group(client, survivor, [n for n in group if n != survivor])
            report.merged_inspirations += len(group) - 1
    
    return report
```

#### Question 合并

```
compact_questions(client, threshold, topk):
    # 无链约束，纯相似度合并
    nodes = MATCH (n:Question) RETURN n
    pairs = find_similar_pairs(nodes, threshold, topk)
    groups = connected_components(pairs)
    for group in groups:
        merge_node_group(client, survivor, victims)
```

#### 单组合并操作

```
merge_node_group(client, survivor_id, victim_ids):
    # 在一个事务内完成
    for victim_id in victim_ids:
        # 1. 转移所有入边和出边到 survivor
        MATCH (v {id: victim_id})-[r]-(neighbor)
        WHERE type(r) IN ['INSP_REFINES', 'INSP_COMBINES', 'INSP_QUESTION', 'PAPER_CONTRIBUTES']
        MERGE (survivor)-[new_r:type(r)]-(neighbor)
        SET new_r = properties(r)
        
        # 2. 合并可变属性
        # 核心描述：保留更长的
        # 前提条件/操作步骤/已知实例/当前现状/未解决部分：concat 去重
        SET survivor.property = coalesce_richer(survivor.property, v.property)
        
        # 3. 删除冗余节点
        DETACH DELETE v
```

#### 边去重

```
deduplicate_edges(client):
    # 合并后可能产生重复边（同一节点对同类型多条边）
    MATCH (a)-[r]->(b)
    MATCH (a)-[r2]->(b)
    WHERE type(r) = type(r2) AND elementId(r) < elementId(r2)
    DELETE r2
```

### 合并报告

compact 输出 JSON 报告到 stdout：

```json
{
  "merged_inspirations": {"总量": 5, "粒1": 2, "粒2": 2, "粒3": 1},
  "merged_questions": {"总量": 3},
  "removed_duplicate_edges": 4,
  "dry_run": false
}
```

`--dry-run` 模式下 `dry_run: true`，不实际执行删除。

### 存活节点选择策略

在一个合并组内选取存活节点的优先级：

1. 有最多下游 `INSP_REFINES` 子节点的（细化链更完整）
2. 有最多 `PAPER_CONTRIBUTES` 入边的（更多论文佐证）
3. `核心描述` 更长的（信息更丰富）
4. 任意（唯一 ID 更小）

## 配置项

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `batch_concurrency` | int | 4 | batch-train 最大并发线程数 |
| `compact_interval` | int | 10 | batch-train 内每 N 篇完成触发 inline compact |
| `compact_threshold` | float | 0.95 | 余弦相似度阈值，高于此值的节点对进入合并候选 |
| `compact_topk` | int | 5 | HNSW 近邻搜索的 k 值 |

## 模块划分

| 文件 | 操作 | 内容 |
|---|---|---|
| `src/neo4j/compact.py` | 新建 | 核心合并算法 |
| `src/cli/commands.py` | 扩展 | `cmd_batch_train`, `cmd_compact` |
| `src/main.py` | 扩展 | 新 subcommand 分发 |
| `config.yaml` | 扩展 | 4 个新配置项 |
| `config.example.yaml` | 扩展 | 同上 |

### `src/neo4j/compact.py` 函数签名

```python
def compact_inspirations(client: Neo4jClient, threshold: float, topk: int) -> CompactReport: ...
def compact_questions(client: Neo4jClient, threshold: float, topk: int) -> CompactReport: ...
def merge_node_group(client: Neo4jClient, survivor_id: str, victim_ids: list[str]) -> None: ...
def compact_all(client: Neo4jClient, config: Config) -> CompactReport: ...
```

## 边界情况

| 场景 | 处理 |
|---|---|
| batch-train 输入为空 | 输出提示，退出 |
| batch 中某篇论文训练失败 | 记录失败，继续下一篇 |
| compact 时无相似节点 | 空报告，正常退出 |
| 被合并节点同时被训练线程引用 | compact 期间暂停训练，不存在此问题 |
| 合并后产生自环边 | `merge_node_group` 中过滤 survivor→survivor 的边 |
| HNSW 索引不存在 | compact 先调用 `bootstrap_hnsw` 幂等创建 |

## 测试要点

- batch-train 端到端：多篇 arXiv 论文并行训练，验证无数据丢失
- compact 单元测试：Mock Neo4j 图，验证链约束过滤、合并组构建、边转移
- compact 幂等性：连续两次 compact 结果一致
- compact --dry-run：验证不修改图
- 并发测试：训练线程和 compact 的暂停/恢复逻辑
- 边界：合并粒度 1 节点后，下游粒度 2 可合并性正确
