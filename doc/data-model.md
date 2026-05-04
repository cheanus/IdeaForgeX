# Neo4j 图数据模型

## 1. 节点标签

两种标签：`Inspiration` 和 `Question`。

## 2. 约束

```cypher
CREATE CONSTRAINT insp_id_unique IF NOT EXISTS
FOR (n:Inspiration) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT q_id_unique IF NOT EXISTS
FOR (n:Question) REQUIRE n.id IS UNIQUE;
```

## 3. 向量索引

```cypher
CREATE VECTOR INDEX idx_insp_vector IF NOT EXISTS
FOR (n:Inspiration) ON (n.向量)
    OPTIONS {indexConfig: {
    `vector.dimensions`: $embedding_dim,
    `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX idx_q_vector IF NOT EXISTS
FOR (n:Question) ON (n.向量)
    OPTIONS {indexConfig: {
    `vector.dimensions`: $embedding_dim,
    `vector.similarity_function`: 'cosine'
}};
```

## 4. 节点属性

### Inspiration

```
(:Inspiration {
    id: string,              -- "insp-{uuid}"
    粒度: int,               -- 0, 1, 2, ... 越小越抽象
    核心描述: string,
    向量: list[float],       -- 维度由 embedding 配置决定
    前提条件: string,
    操作步骤: string,
    已知实例: string         -- 可缺省
})
```

### Question

```
(:Question {
    id: string,              -- "q-{uuid}"
    核心描述: string,
    向量: list[float],       -- 维度由 embedding 配置决定
    问题类型: string,        -- "理论缺口" | "工程瓶颈" | "评估缺失" | "跨领域空白"
    当前现状: string,
    未解决部分: string
})
```

## 5. 边

### 5.1 灵感组合边

```cypher
MATCH (a:Inspiration {id: $id_a}), (b:Inspiration {id: $id_b})
CREATE (a)-[:INSP_COMBINES {weight: $weight}]->(b)
```

无向语义——遍历时忽略方向：`(a)-[:INSP_COMBINES]-(b)`。

约束：a 和 b 必须属于不同方法概念（不同精化链）。

### 5.2 灵感问题边

```cypher
MATCH (a:Inspiration {id: $id_a}), (b:Question {id: $id_b})
CREATE (a)-[:INSP_QUESTION {weight: $weight}]->(b)
```

无向。

### 5.3 问题组合边

```cypher
MATCH (a:Question {id: $id_a}), (b:Question {id: $id_b})
CREATE (a)-[:QUESTION_COMBINES {weight: $weight}]->(b)
```

无向。语义：两问题的交集定义了一个新研究缺口。

### 5.4 灵感精化边

```cypher
MATCH (coarse:Inspiration {id: $id_coarse}), (fine:Inspiration {id: $id_fine})
WHERE coarse.粒度 < fine.粒度
CREATE (coarse)-[:INSP_REFINES {weight: 1.0}]->(fine)
```

有向，从低粒度指向高粒度。权重固定 1.0。

约束：同一精化链上所有节点同属一个方法概念。LLM A 确保它们串联为单链，无分支。

## 6. 完整 Cypher 示例

### 创建一个方法概念（粒度 0/1/2）

```cypher
// 冷启动时需要手动给定 id。训练时 LLM A 生成 UUID。
CREATE (c:Inspiration {
    id: 'insp-a1b2c3-0',
    粒度: 0,
    核心描述: '将复杂变换分解为低维子空间组合',
    向量: $emb_0,
    前提条件: '存在高维非线性映射可分解为多个近似独立子结构',
    操作步骤: '1) 识别子结构边界 2) 独立建模 3) 合成',
    已知实例: 'LoRA, SVD-based fine-tuning'
})
CREATE (m:Inspiration {
    id: 'insp-a1b2c3-1',
    粒度: 1,
    核心描述: '用低秩分解解耦预训练权重更新',
    向量: $emb_1,
    前提条件: '前提条件同上',
    操作步骤: '操作步骤同上',
    已知实例: '同上'
})
CREATE (f:Inspiration {
    id: 'insp-a1b2c3-2',
    粒度: 2,
    核心描述: '以 LoRA 低秩矩阵替代全量微调中的权重增量',
    向量: $emb_2,
    前提条件: '前提条件同上',
    操作步骤: '操作步骤同上',
    已知实例: '同上'
})

// 精化边
CREATE (c)-[:INSP_REFINES {weight: 1.0}]->(m)
CREATE (m)-[:INSP_REFINES {weight: 1.0}]->(f)
```

注意：前提条件/操作步骤/已知实例 在三节点间一致。LLM A 生成时复制同概念的结构字段。

### 创建一个 Question

```cypher
CREATE (:Question {
    id: 'q-d4e5f6',
    核心描述: '大模型长文本处理中段信息利用率显著低于首尾',
    向量: $emb_q,
    问题类型: '理论缺口',
    当前现状: '已有方法通过位置编码外推或滑动窗口缓解',
    未解决部分: '中间段信息丢失的根本原因尚未被理论解释'
})
```

### 创建组合边

```cypher
// 灵感组合边
MATCH (a:Inspiration {id: 'insp-a1b2c3-2'}), (b:Inspiration {id: 'insp-x7y8z9-1'})
CREATE (a)-[:INSP_COMBINES {weight: 0.85}]->(b)

// 灵感问题边
MATCH (a:Inspiration {id: 'insp-a1b2c3-2'}), (q:Question {id: 'q-d4e5f6'})
CREATE (a)-[:INSP_QUESTION {weight: 0.9}]->(q)

// 问题组合边
MATCH (q1:Question {id: 'q-d4e5f6'}), (q2:Question {id: 'q-aaa111'})
CREATE (q1)-[:QUESTION_COMBINES {weight: 0.75}]->(q2)
```

## 7. 遍历查询

### 精化链双向展开

```cypher
MATCH (hit:Inspiration {id: $hit_id})
CALL {
    MATCH (hit)-[:INSP_REFINES*0..10]->(finer:Inspiration) RETURN finer
    UNION
    MATCH (coarser:Inspiration)-[:INSP_REFINES*0..10]->(hit) RETURN coarser AS finer
}
RETURN DISTINCT finer
```

### 1-hop 按权重取 top-3

```cypher
MATCH (n {id: $node_id})-[r:INSP_COMBINES|INSP_QUESTION|QUESTION_COMBINES]-(m)
RETURN m, labels(m)[0] AS label, r.weight AS weight
ORDER BY weight DESC
LIMIT 3
```

## 8. 重置

```cypher
// 删除所有实践库节点和边（保留约束和索引）
MATCH (n:Inspiration) DETACH DELETE n
MATCH (n:Question) DETACH DELETE n
```

## 9. 失败库记录

当前 A-only 版本不写失败库；本节仅保留历史说明，后续若重新引入反馈环再恢复最小快照定义。
