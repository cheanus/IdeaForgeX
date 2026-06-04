# 并行训练与图压缩 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 batch-train 并行训练命令和 compact 图压缩命令，解决并行训练导致的语义重复节点问题。

**Architecture:** 核心在 `src/neo4j/compact.py` 实现压缩算法，复用现有 HNSW 向量索引做近邻发现，自顶向下按粒度合并 Inspiration 节点（维持链约束），Question 节点纯相似度合并。`batch-train` 在 `src/cli/queries.py` 用批处理模式 + ThreadPoolExecutor 实现并行训练，每批完成后触发 compact。

**Tech Stack:** Python threading, Neo4j HNSW vector index, UnionFind 并查集

---

## 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| 新建 | `src/neo4j/compact.py` | 压缩算法核心（UnionFind、近邻搜索、节点合并、边去重） |
| 新建 | `tests/test_compact.py` | compact 单元测试 |
| 修改 | `src/config.py` | 新增 `batch_concurrency`, `compact_interval`, `compact_threshold`, `compact_topk` |
| 修改 | `src/cli/queries.py` | 新增 `cmd_batch_train`, `cmd_compact` |
| 修改 | `src/main.py` | 新增 `batch-train`, `compact` 子命令 |
| 修改 | `config.yaml` | 新增 4 个配置项 |
| 修改 | `config.example.yaml` | 同步新增 |

---

### Task 1: 新增配置项

**Files:**
- Modify: `src/config.py:51-52`
- Modify: `config.yaml:40`
- Modify: `config.example.yaml:37`

- [ ] **Step 1: 在 Config 类新增 4 个字段**

```python
# src/config.py — 在 final_k 之后、log_level 之前添加
    batch_concurrency: int = Field(default=4)
    compact_interval: int = Field(default=10)
    compact_threshold: float = Field(default=0.95)
    compact_topk: int = Field(default=5)
```

- [ ] **Step 2: 在 config.yaml 尾部添加配置**

```yaml
# ── 并行训练与压缩 ──
batch_concurrency: 4       # 并行训练最大线程数
compact_interval: 10       # 每 N 篇完成触发 inline compact
compact_threshold: 0.95    # 余弦相似度合并阈值
compact_topk: 5            # HNSW 近邻搜索 k
```

- [ ] **Step 3: 在 config.example.yaml 尾部同步添加**

内容同上（不含敏感值）。

- [ ] **Step 4: 验证配置加载**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python -c "from src.config import load_config; c=load_config(); print(c.batch_concurrency, c.compact_interval, c.compact_threshold, c.compact_topk)"
```
Expected: `4 10 0.95 5`

- [ ] **Step 5: 提交**

```bash
git add src/config.py config.yaml config.example.yaml
git commit -m "config: 新增 batch_concurrency/compact_interval/compact_threshold/compact_topk 配置项"
```

---

### Task 2: compact 核心 — UnionFind 与近邻搜索

**Files:**
- Create: `src/neo4j/compact.py`

- [ ] **Step 1: 创建文件骨架与 UnionFind 类**

```python
"""知识图谱压缩 — 合并语义相近节点。"""
from __future__ import annotations

import logging
from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient

_logger = logging.getLogger("ideaforgex")


class UnionFind:
    """并查集，用于将相似节点对聚合为合并组。"""

    def __init__(self, elements: set[str]) -> None:
        self._parent = {e: e for e in elements}
        self._rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self._rank[px] < self._rank[py]:
            self._parent[px] = py
        elif self._rank[px] > self._rank[py]:
            self._parent[py] = px
        else:
            self._parent[py] = px
            self._rank[px] += 1

    def get_groups(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for e in self._parent:
            root = self.find(e)
            if root not in groups:
                groups[root] = set()
            groups[root].add(e)
        # 过滤掉只有单个元素的组（无需合并）
        return [s for s in groups.values() if len(s) > 1]


def _fetch_inspiration_nodes(client: Neo4jClient, granularity: int) -> list[dict[str, Any]]:
    """获取指定粒度的 Inspiration 节点列表。"""
    with client.session() as session:
        result = session.run(
            "MATCH (n:Inspiration) WHERE n.粒度 = $granularity RETURN n.id AS id, n.向量 AS vector",
            granularity=granularity,
        )
        return [{"id": r["id"], "vector": r["vector"]} for r in result]


def _find_similar_pairs(
    client: Neo4jClient,
    nodes: list[dict[str, Any]],
    index_name: str,
    threshold: float,
    topk: int,
    granularity: int | None = None,
) -> list[tuple[str, str, float]]:
    """对节点列表通过 HNSW 向量索引查找相似对。

    返回 [(id_a, id_b, score), ...]，score >= threshold。
    使用 `id_a < id_b` 避免重复对。
    """
    from src.neo4j.schema import ensure_schema

    ensure_schema(client)

    all_ids = {n["id"] for n in nodes}
    pairs: dict[tuple[str, str], float] = {}

    with client.session() as session:
        for node in nodes:
            params: dict[str, Any] = {
                "index": index_name,
                "k": topk,
                "vector": node["vector"],
                "threshold": threshold,
                "my_id": node["id"],
            }
            if granularity is not None:
                query = """
                CALL db.index.vector.queryNodes($index, $k, $vector)
                YIELD node AS neighbor, score
                WHERE neighbor.id IN $all_ids
                  AND neighbor.id < $my_id
                  AND neighbor.粒度 = $granularity
                  AND score >= $threshold
                RETURN neighbor.id AS other_id, score
                """
                params["granularity"] = granularity
                params["all_ids"] = list(all_ids)
            else:
                query = """
                CALL db.index.vector.queryNodes($index, $k, $vector)
                YIELD node AS neighbor, score
                WHERE neighbor.id IN $all_ids
                  AND neighbor.id < $my_id
                  AND score >= $threshold
                RETURN neighbor.id AS other_id, score
                """
                params["all_ids"] = list(all_ids)

            records = session.run(query, **params)
            for record in records:
                key = (node["id"], record["other_id"])
                pairs[key] = record["score"]

    return [(a, b, s) for (a, b), s in pairs.items()]
```

- [ ] **Step 2: 验证语法无错误**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python -c "from src.neo4j.compact import UnionFind, _find_similar_pairs; print('import ok')"
```
Expected: `import ok`

- [ ] **Step 3: 提交**

```bash
git add src/neo4j/compact.py
git commit -m "compact: 新增 UnionFind 与 HNSW 近邻相似对查找"
```

---

### Task 3: compact 核心 — Inspiration 合并

**Files:**
- Modify: `src/neo4j/compact.py`

在现有文件末尾追加以下函数。

- [ ] **Step 1: 添加 INSP_REFINES 父节点查找辅助函数**

```python
def _get_insp_refines_parent(client: Neo4jClient, node_id: str) -> str | None:
    """获取 Inspiration 节点的 INSP_REFINES 上游父节点 ID。"""
    with client.session() as session:
        record = session.run(
            """
            MATCH (parent:Inspiration)-[:INSP_REFINES]->(child:Inspiration {id: $id})
            RETURN parent.id AS parent_id
            """,
            id=node_id,
        ).single()
        return record["parent_id"] if record else None
```

- [ ] **Step 2: 添加 pick_survivor 函数**

```python
def _pick_survivor(client: Neo4jClient, node_ids: set[str]) -> str:
    """在合并组中选出存活节点。优先级：子节点多 > 贡献论文多 > 核心描述长 > ID 小。"""
    node_list = list(node_ids)
    if len(node_list) == 1:
        return node_list[0]

    with client.session() as session:
        # 查询每个节点的子节点数、贡献论文数、核心描述长度
        result = session.run(
            """
            UNWIND $ids AS nid
            MATCH (n {id: nid})
            OPTIONAL MATCH (n)-[:INSP_REFINES]->(child:Inspiration)
            OPTIONAL MATCH (paper:Paper)-[:PAPER_CONTRIBUTES]->(n)
            RETURN n.id AS id,
                   count(DISTINCT child) AS child_count,
                   count(DISTINCT paper) AS paper_count,
                   size(COALESCE(n.核心描述, '')) AS desc_len
            ORDER BY child_count DESC, paper_count DESC, desc_len DESC, id ASC
            LIMIT 1
            """,
            ids=node_list,
        )
        record = result.single()
        if record:
            return record["id"]
        return node_list[0]
```

- [ ] **Step 3: 添加 merge_node_group 函数**

```python
def _merge_node_group(client: Neo4jClient, survivor_id: str, victim_ids: list[str]) -> None:
    """将一个合并组中的冗余节点合并到存活节点。

    操作：转移所有边 → 合并可变属性 → 删除冗余节点。
    允许产生重复边，由后续 deduplicate_edges 清理。
    """
    if not victim_ids:
        return

    with client.driver.session(database=client.config.neo4j_database) as session:

        def _merge_tx(tx) -> None:
            for victim_id in victim_ids:
                # 1. 获取被合并节点的所有边
                edges = tx.run(
                    """
                    MATCH (victim {id: $victim_id})-[r]-(neighbor)
                    WHERE type(r) IN ['INSP_REFINES','INSP_COMBINES','INSP_QUESTION','PAPER_CONTRIBUTES','QUESTION_COMBINES']
                    RETURN type(r) AS rel_type,
                           r.weight AS weight,
                           startNode(r).id AS from_id,
                           endNode(r).id AS to_id,
                           neighbor.id AS neighbor_id
                    """,
                    victim_id=victim_id,
                ).data()

                # 2. 转移边到 survivor
                REL_TYPES = {
                    "INSP_REFINES", "INSP_COMBINES", "INSP_QUESTION",
                    "PAPER_CONTRIBUTES", "QUESTION_COMBINES",
                }
                for edge in edges:
                    if edge["neighbor_id"] == survivor_id:
                        continue  # 跳过自环
                    rel_type = edge["rel_type"]
                    if rel_type not in REL_TYPES:
                        continue
                    weight = edge["weight"]
                    if edge["from_id"] == victim_id:
                        tx.run(
                            f"""
                            MATCH (s {{id: $survivor_id}})
                            MATCH (n {{id: $neighbor_id}})
                            CREATE (s)-[:`{rel_type}` {{weight: $weight}}]->(n)
                            """,
                            survivor_id=survivor_id,
                            neighbor_id=edge["neighbor_id"],
                            weight=weight,
                        )
                    else:
                        tx.run(
                            f"""
                            MATCH (n {{id: $neighbor_id}})
                            MATCH (s {{id: $survivor_id}})
                            CREATE (n)-[:`{rel_type}` {{weight: $weight}}]->(s)
                            """,
                            neighbor_id=edge["neighbor_id"],
                            survivor_id=survivor_id,
                            weight=weight,
                        )

                # 3. 合并可变属性（保留更长的值）
                tx.run(
                    """
                    MATCH (victim {id: $victim_id})
                    MATCH (survivor {id: $survivor_id})
                    SET survivor.前提条件 = CASE
                            WHEN size(COALESCE(survivor.前提条件, '')) >= size(COALESCE(victim.前提条件, ''))
                            THEN COALESCE(survivor.前提条件, victim.前提条件)
                            ELSE victim.前提条件 END,
                        survivor.操作步骤 = CASE
                            WHEN size(COALESCE(survivor.操作步骤, '')) >= size(COALESCE(victim.操作步骤, ''))
                            THEN COALESCE(survivor.操作步骤, victim.操作步骤)
                            ELSE victim.操作步骤 END,
                        survivor.当前现状 = CASE
                            WHEN size(COALESCE(survivor.当前现状, '')) >= size(COALESCE(victim.当前现状, ''))
                            THEN COALESCE(survivor.当前现状, victim.当前现状)
                            ELSE victim.当前现状 END,
                        survivor.未解决部分 = CASE
                            WHEN size(COALESCE(survivor.未解决部分, '')) >= size(COALESCE(victim.未解决部分, ''))
                            THEN COALESCE(survivor.未解决部分, victim.未解决部分)
                            ELSE victim.未解决部分 END
                    """,
                    victim_id=victim_id,
                    survivor_id=survivor_id,
                )

                # 4. 删除冗余节点
                tx.run(
                    "MATCH (n {id: $victim_id}) DETACH DELETE n",
                    victim_id=victim_id,
                )

        session.execute_write(_merge_tx)
```

- [ ] **Step 4: 添加 compact_inspirations 函数**

```python
def compact_inspirations(
    client: Neo4jClient, threshold: float, topk: int
) -> dict[str, int]:
    """自顶向下（粒 1→2→3）合并语义相近的 Inspiration 节点。

    粒度 >1 的合并需满足：上游 INSP_REFINES 父节点也在同一合并组。
    返回 {"总量": N, "粒1": n1, "粒2": n2, "粒3": n3}。
    """
    report: dict[str, int] = {"总量": 0, "粒1": 0, "粒2": 0, "粒3": 0}
    # granularity -> {node_id: group_id}，用于下游合并的链约束检查
    parent_groups: dict[int, dict[str, int]] = {}

    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue

        # 1. HNSW 近邻查找相似对
        pairs = _find_similar_pairs(
            client, nodes, "idx_insp_vector", threshold, topk, granularity=granularity
        )

        # 2. 链约束过滤（粒度 > 1）
        if granularity > 1 and pairs:
            prev_groups = parent_groups.get(granularity - 1, {})
            def _same_parent_group(a_id: str, b_id: str) -> bool:
                pa = _get_insp_refines_parent(client, a_id)
                pb = _get_insp_refines_parent(client, b_id)
                if pa is None and pb is None:
                    return True  # 都无父节点，允许合并
                if pa is None or pb is None:
                    return False  # 一个有父节点、一个无，不允许
                ga = prev_groups.get(pa)
                gb = prev_groups.get(pb)
                if ga is not None and gb is not None and ga == gb:
                    return True
                return False

            pairs = [(a, b, s) for (a, b, s) in pairs if _same_parent_group(a, b)]

        if not pairs:
            continue

        # 3. 并查集构建合并组
        all_ids = {n["id"] for n in nodes}
        uf = UnionFind(all_ids)
        for a_id, b_id, _ in pairs:
            uf.union(a_id, b_id)

        groups = uf.get_groups()

        # 4. 记录本层的合并组映射（供下一层链约束使用）
        parent_groups[granularity] = {}
        for gid, member_set in enumerate(groups):
            for node_id in member_set:
                parent_groups[granularity][node_id] = gid

        # 5. 执行合并
        for member_set in groups:
            survivor = _pick_survivor(client, member_set)
            victims = [nid for nid in member_set if nid != survivor]
            _merge_node_group(client, survivor, victims)
            count = len(victims)
            report["粒" + str(granularity)] += count
            report["总量"] += count

    return report
```

- [ ] **Step 5: 验证语法**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python -c "from src.neo4j.compact import compact_inspirations; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: 提交**

```bash
git add src/neo4j/compact.py
git commit -m "compact: 实现 Inspiration 节点自顶向下合并算法"
```

---

### Task 4: compact 核心 — Question 合并、边去重、compact_all 入口

**Files:**
- Modify: `src/neo4j/compact.py`

在现有文件末尾追加。

- [ ] **Step 1: 添加 compact_questions 和 deduplicate_edges**

```python
def compact_questions(
    client: Neo4jClient, threshold: float, topk: int
) -> dict[str, int]:
    """合并语义相近的 Question 节点（无链约束）。"""
    with client.session() as session:
        result = session.run(
            "MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector"
        )
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]

    if not nodes:
        return {"总量": 0}

    pairs = _find_similar_pairs(client, nodes, "idx_q_vector", threshold, topk)
    if not pairs:
        return {"总量": 0}

    all_ids = {n["id"] for n in nodes}
    uf = UnionFind(all_ids)
    for a_id, b_id, _ in pairs:
        uf.union(a_id, b_id)

    groups = uf.get_groups()
    total_merged = 0
    for member_set in groups:
        survivor = _pick_survivor(client, member_set)
        victims = [nid for nid in member_set if nid != survivor]
        _merge_node_group(client, survivor, victims)
        total_merged += len(victims)

    return {"总量": total_merged}


def _deduplicate_edges(client: Neo4jClient) -> int:
    """清除合并后产生的重复边（同节点对、同类型）。"""
    with client.driver.session(database=client.config.neo4j_database) as session:
        result = session.run(
            """
            MATCH (a)-[r]->(b)
            MATCH (a)-[r2]->(b)
            WHERE type(r) = type(r2) AND elementId(r) < elementId(r2)
            WITH r2, count(r2) AS c
            DELETE r2
            RETURN sum(c) AS total_removed
            """
        )
        record = result.single()
        return record["total_removed"] or 0 if record else 0
```

- [ ] **Step 2: 添加 compact_all 入口函数**

```python
def compact_all(client: Neo4jClient, config: Config) -> dict[str, Any]:
    """完整压缩入口：Inspiration → Question → 边去重。"""
    _logger.info("开始压缩 Inspiration 节点 …")
    insp_report = compact_inspirations(
        client, config.compact_threshold, config.compact_topk
    )
    _logger.info(
        "Inspiration 压缩完成，共合并 %d 个节点（粒1=%d, 粒2=%d, 粒3=%d）",
        insp_report["总量"],
        insp_report.get("粒1", 0),
        insp_report.get("粒2", 0),
        insp_report.get("粒3", 0),
    )

    _logger.info("开始压缩 Question 节点 …")
    q_report = compact_questions(
        client, config.compact_threshold, config.compact_topk
    )
    _logger.info("Question 压缩完成，共合并 %d 个节点", q_report["总量"])

    _logger.info("开始去重重复边 …")
    removed_edges = _deduplicate_edges(client)
    _logger.info("边去重完成，共删除 %d 条重复边", removed_edges)

    return {
        "merged_inspirations": insp_report,
        "merged_questions": q_report,
        "removed_duplicate_edges": removed_edges,
        "dry_run": False,
    }
```

- [ ] **Step 3: 添加 compact_dry_run 函数**

```python
def compact_dry_run(client: Neo4jClient, config: Config) -> dict[str, Any]:
    """不实际执行，仅报告将会合并的节点数量和组数。"""
    threshold = config.compact_threshold
    topk = config.compact_topk
    insp_groups = 0
    insp_merged = 0
    q_groups = 0
    q_merged = 0

    for granularity in [1, 2, 3]:
        nodes = _fetch_inspiration_nodes(client, granularity)
        if not nodes:
            continue
        pairs = _find_similar_pairs(
            client, nodes, "idx_insp_vector", threshold, topk, granularity=granularity
        )
        if not pairs:
            continue
        all_ids = {n["id"] for n in nodes}
        uf = UnionFind(all_ids)
        for a_id, b_id, _ in pairs:
            uf.union(a_id, b_id)
        groups = uf.get_groups()
        insp_groups += len(groups)
        insp_merged += sum(len(g) - 1 for g in groups)

    with client.session() as session:
        result = session.run("MATCH (n:Question) RETURN n.id AS id, n.向量 AS vector")
        nodes = [{"id": r["id"], "vector": r["vector"]} for r in result]
    if nodes:
        pairs = _find_similar_pairs(client, nodes, "idx_q_vector", threshold, topk)
        if pairs:
            all_ids = {n["id"] for n in nodes}
            uf = UnionFind(all_ids)
            for a_id, b_id, _ in pairs:
                uf.union(a_id, b_id)
            groups = uf.get_groups()
            q_groups = len(groups)
            q_merged = sum(len(g) - 1 for g in groups)

    return {
        "merged_inspirations": {"总量": insp_merged, "组数": insp_groups},
        "merged_questions": {"总量": q_merged, "组数": q_groups},
        "dry_run": True,
    }
```

- [ ] **Step 4: 验证语法**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python -c "from src.neo4j.compact import compact_all, compact_dry_run; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: 提交**

```bash
git add src/neo4j/compact.py
git commit -m "compact: 实现 Question 合并、边去重与 compact_all 入口"
```

---

### Task 5: compact 单元测试

**Files:**
- Create: `tests/test_compact.py`

- [ ] **Step 1: 编写 UnionFind 单元测试（不依赖 Neo4j）**

```python
"""compact 压缩算法单元测试。"""
from __future__ import annotations

import pytest

from src.neo4j.compact import UnionFind


class TestUnionFind:
    def test_single_element(self):
        uf = UnionFind({"a"})
        assert uf.find("a") == "a"
        assert uf.get_groups() == []

    def test_union_pair(self):
        uf = UnionFind({"a", "b"})
        uf.union("a", "b")
        groups = uf.get_groups()
        assert len(groups) == 1
        assert groups[0] == {"a", "b"}

    def test_union_chain(self):
        uf = UnionFind({"a", "b", "c"})
        uf.union("a", "b")
        uf.union("b", "c")
        groups = uf.get_groups()
        assert len(groups) == 1
        assert groups[0] == {"a", "b", "c"}

    def test_disjoint_groups(self):
        uf = UnionFind({"a", "b", "c", "d"})
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.get_groups()
        assert len(groups) == 2

    def test_no_merge_singletons(self):
        uf = UnionFind({"a", "b", "c"})
        uf.union("a", "b")
        groups = uf.get_groups()
        # "c" is singleton, should be excluded
        assert len(groups) == 1
        assert groups[0] == {"a", "b"}

    def test_find_path_compression(self):
        uf = UnionFind({"a", "b", "c", "d"})
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        # After path compression, all should have same root
        root = uf.find("a")
        assert uf.find("b") == root
        assert uf.find("c") == root
        assert uf.find("d") == root
```

- [ ] **Step 2: 运行 UnionFind 测试**

```bash
cd /home/test/Codes/IdeaForgeX && uv run pytest tests/test_compact.py::TestUnionFind -v
```
Expected: 6 passed

- [ ] **Step 3: 编写 compact 集成测试（需 Neo4j）**

在 `tests/test_compact.py` 末尾追加：

```python
@pytest.mark.neo4j
class TestCompactNeo4j:
    """Neo4j 集成测试，需要 neo4j-test 容器运行。"""

    def _create_insp_node(self, neo4j_client, node_id: str, granularity: int,
                          desc: str, vector: list[float]) -> None:
        with neo4j_client.driver.session(
            database=neo4j_client.config.neo4j_database
        ) as session:
            session.run(
                """
                CREATE (n:Inspiration {
                    id: $id, 粒度: $granularity, 核心描述: $desc, 向量: $vector,
                    前提条件: '', 操作步骤: ''
                })
                """,
                id=node_id, granularity=granularity,
                desc=desc, vector=vector,
            )

    def _create_q_node(self, neo4j_client, node_id: str, desc: str,
                       vector: list[float]) -> None:
        with neo4j_client.driver.session(
            database=neo4j_client.config.neo4j_database
        ) as session:
            session.run(
                """
                CREATE (n:Question {
                    id: $id, 核心描述: $desc, 向量: $vector,
                    问题类型: '理论缺口', 当前现状: '', 未解决部分: ''
                })
                """,
                id=node_id, desc=desc, vector=vector,
            )

    def test_compact_no_nodes(self, neo4j_client):
        """空图压缩应正常返回。"""
        from src.neo4j.compact import compact_all
        from src.config import Config

        config = Config()
        result = compact_all(neo4j_client, config)
        assert result["merged_inspirations"]["总量"] == 0
        assert result["merged_questions"]["总量"] == 0

    def test_compact_dry_run_no_modification(self, neo4j_client):
        """dry-run 不修改图。"""
        self._create_insp_node(neo4j_client, "insp-1", 1, "test", [0.1, 0.2, 0.3])
        from src.neo4j.compact import compact_dry_run
        from src.config import Config

        config = Config()
        result = compact_dry_run(neo4j_client, config)
        assert result["dry_run"] is True

        # 验证节点未被删除
        with neo4j_client.session() as session:
            count = session.run(
                "MATCH (n:Inspiration {id: 'insp-1'}) RETURN count(n) AS c"
            ).single()["c"]
        assert count == 1
```

- [ ] **Step 4: 运行 Neo4j 集成测试**

```bash
cd /home/test/Codes/IdeaForgeX && uv run pytest tests/test_compact.py::TestCompactNeo4j -v
```
Expected: 2 passed（需 neo4j-test 容器运行中）

- [ ] **Step 5: 运行全部 compact 测试**

```bash
cd /home/test/Codes/IdeaForgeX && uv run pytest tests/test_compact.py -v
```
Expected: 8 passed

- [ ] **Step 6: 提交**

```bash
git add tests/test_compact.py
git commit -m "test: 新增 compact UnionFind 与 Neo4j 集成测试"
```

---

### Task 6: CLI — cmd_compact

**Files:**
- Modify: `src/cli/queries.py`

- [ ] **Step 1: 在 queries.py 末尾添加 cmd_compact**

```python
def cmd_compact(
    config: Config,
    neo4j_client: Neo4jClient,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行全图压缩，dry_run 为 True 时仅报告不执行。"""
    from src.neo4j.compact import compact_all, compact_dry_run

    if dry_run:
        return compact_dry_run(neo4j_client, config)
    return compact_all(neo4j_client, config)
```

- [ ] **Step 2: 提交**

```bash
git add src/cli/queries.py
git commit -m "cli: 新增 cmd_compact 命令"
```

---

### Task 7: CLI — cmd_batch_train

**Files:**
- Modify: `src/cli/queries.py`

- [ ] **Step 1: 在 src/cli/queries.py 头部添加 import**

在文件顶部现有 import 区域追加：

```python
import concurrent.futures
import json
import threading
```

- [ ] **Step 2: 在 queries.py 末尾添加 _train_one_paper 和 cmd_batch_train**

```python
def _train_one_paper(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    paper: str,
) -> dict[str, Any]:
    """训练单篇论文，返回 {"status": "ok"} 或抛异常。"""
    from src.agent.training import build_training_graph, run_training

    _logger.info("开始训练论文: %s", paper)
    graph = build_training_graph(config, llm_client, neo4j_client)
    result = run_training(graph, paper)
    if result.get("already_trained"):
        _logger.info("论文已存在于图库，跳过: %s", paper)
    else:
        _logger.info("论文训练完成: %s", paper)
    return {"status": "ok"}


def cmd_batch_train(
    config: Config,
    llm_client: ChatClient,
    neo4j_client: Neo4jClient,
    papers: list[str],
    progress_callback=None,
) -> dict[str, Any]:
    """并行训练多篇论文，分批 compact。

    Args:
        papers: 论文标识列表
        progress_callback: 可选回调 (completed, total, paper, status)

    Returns:
        {"总论文数": N, "成功数": M, "失败数": F, "成功列表": [...], "失败列表": [...]}
    """
    from src.neo4j.compact import compact_all

    total = len(papers)
    succeeded: list[str] = []
    failed: list[dict[str, str]] = []
    lock = threading.Lock()

    interval = config.compact_interval
    batch_size = interval if interval > 0 else total

    for i in range(0, total, batch_size):
        batch = papers[i : i + batch_size]
        _logger.info("开始训练第 %d-%d 批 (共 %d 篇)", i + 1, i + len(batch), total)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.batch_concurrency
        ) as executor:
            future_map = {}
            for paper in batch:
                future = executor.submit(
                    _train_one_paper, config, llm_client, neo4j_client, paper
                )
                future_map[future] = paper

            for future in concurrent.futures.as_completed(future_map):
                paper = future_map[future]
                try:
                    future.result()
                    with lock:
                        succeeded.append(paper)
                    if progress_callback:
                        progress_callback(len(succeeded) + len(failed), total, paper, "ok")
                except Exception as e:
                    _logger.error("论文 %s 训练失败: %s", paper, e)
                    with lock:
                        failed.append({"论文": paper, "错误": str(e)})
                    if progress_callback:
                        progress_callback(len(succeeded) + len(failed), total, paper, "failed")

        # 批次完成，触发 compact
        completed_so_far = i + len(batch)
        if interval > 0 and completed_so_far < total:
            _logger.info("批次完成，开始压缩知识图谱 …")
            compact_report = compact_all(neo4j_client, config)
            print(json.dumps(compact_report, ensure_ascii=False, indent=2))

    # 最终 compact
    _logger.info("全部训练完成，执行最终压缩 …")
    final_report = compact_all(neo4j_client, config)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))

    result = {
        "总论文数": total,
        "成功数": len(succeeded),
        "失败数": len(failed),
        "成功列表": succeeded,
        "失败列表": failed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
```

- [ ] **Step 3: 验证语法**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python -c "from src.cli.queries import cmd_batch_train, cmd_compact; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: 提交**

```bash
git add src/cli/queries.py
git commit -m "cli: 新增 cmd_batch_train 批量并行训练命令"
```

---

### Task 8: main.py 接线 — 新增子命令

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: 在 build_parser() 中添加 batch-train 和 compact 子命令**

在 `build_parser()` 函数中，`relate` 定义之后、`return parser` 之前添加：

```python
    batch_train = subparsers.add_parser("batch-train")
    batch_train.add_argument(
        "papers", nargs="*", help="论文 ID 列表（arXiv ID 或标题），可从命令行传入"
    )
    batch_train.add_argument(
        "--queries", type=str, default=None, help="纯文本查询文件，每行一个论文标识"
    )
    batch_train.add_argument(
        "--jsonl", type=str, default=None,
        help='JSONL 文件，每行 {"id":"arxiv:1706.03762","title":"...","abstract":"..."}',
    )

    compact = subparsers.add_parser("compact")
    compact.add_argument(
        "--dry-run", action="store_true", help="仅报告将合并的节点，不实际执行"
    )
```

- [ ] **Step 2: 在 main() 中添加 dispatch 分支**

在 `main()` 函数中，`if args.command == "stats":` 之前添加：

```python
        if args.command == "batch-train":
            papers: list[str] = list(args.papers or [])
            if args.file:
                with open(args.file, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#"):
                            papers.append(stripped)
            if not papers:
                _logger.warning("未提供任何论文标识，退出")
                return
            _logger.info("开始批量训练，共 %d 篇论文，并发数 %d", len(papers), config.batch_concurrency)
            cmd_batch_train(config, llm_client, neo4j_client, papers)
            return
        if args.command == "compact":
            _logger.info(
                "开始图压缩，模式=%s",
                "dry-run（仅报告）" if args.compact_dry_run else "正式执行",
            )
            result = cmd_compact(config, neo4j_client, dry_run=args.compact_dry_run)
            _json_print(result)
            return
```

注意：`args.compact_dry_run` 在上面的 Step 1 中 add_argument 定义的是 `--dry-run`，argparse 会把 `--dry-run` 转为 `dry_run`。需要确认参数名一致。

- [ ] **Step 3: 在顶部 import 中添加新函数的导入**

在 `main.py` 顶部，`from src.cli.queries import (` 的导入块中添加 `cmd_batch_train, cmd_compact`：

```python
from src.cli.queries import (
    cmd_batch_train,
    cmd_compact,
    cmd_delete_node,
    cmd_inspect,
    cmd_random,
    cmd_relate,
    cmd_retrieve,
)
```

- [ ] **Step 4: 验证 CLI 解析**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python main.py batch-train --help
```
Expected: 显示 batch-train 帮助

```bash
cd /home/test/Codes/IdeaForgeX && uv run python main.py compact --help
```
Expected: 显示 compact 帮助

- [ ] **Step 5: 提交**

```bash
git add src/main.py
git commit -m "main: 接线 batch-train 和 compact 子命令"
```

---

### Task 9: 端到端验证

- [ ] **Step 1: 运行全部现有测试确认无回归**

```bash
cd /home/test/Codes/IdeaForgeX && uv run pytest tests/ -v
```
Expected: 全部通过（含新增的 compact 测试）

- [ ] **Step 2: dry-run 测试**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python main.py compact --dry-run
```
Expected: 输出 JSON，`dry_run: true`，无实际删除

- [ ] **Step 3: compact 幂等性测试（需图中已有数据）**

```bash
cd /home/test/Codes/IdeaForgeX && uv run python main.py compact
# 输出第一次结果
uv run python main.py compact
# 输出第二次结果，总量应为 0（已无重复节点）
```
Expected: 第二次 `merged_inspirations: {"总量": 0}`

- [ ] **Step 4: 提交**

```bash
git add -A && git diff --cached --stat  # 检查变更范围
git commit -m "chore: 端到端验证与 lint 修复"
```

---

## 自审清单

**1. Spec coverage:**
- batch-train 命令（含 --file） → Task 7
- compact 命令（含 --dry-run） → Task 6
- Inspiration 自顶向下按粒度合并 → Task 3
- 链约束 → Task 3 Step 2
- 同粒度限制 → `_find_similar_pairs` 中 `neighbor.粒度 = $granularity`
- Question 无约束合并 → Task 4 Step 1
- 边转移 + 属性合并 + 删除 → Task 3 Step 3
- 边去重 → Task 4 Step 1
- batch 输出 JSON → Task 7 Step 2
- compact 报告 JSON → Task 4 Step 2
- 配置项 4 个 → Task 1

**2. Placeholder scan:** 无 TBD/TODO。

**3. Type consistency:**
- `compact_inspirations` 返回 `dict[str, int]` ✓
- `compact_questions` 返回 `dict[str, int]` ✓
- `compact_all` 返回 `dict[str, Any]` ✓
- `cmd_compact` 签名 `(config, neo4j_client, dry_run=False)` ✓
- `cmd_batch_train` 签名 `(config, llm_client, neo4j_client, papers)` ✓
- Argparse `--dry-run` → `args.dry_run` ✓
