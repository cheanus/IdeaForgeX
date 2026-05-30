# CLI 使用指南

训练时：把论文喂给知识图谱。推理时：外部 AI agent 通过 CLI 查询图数据，自行编排创新点。

## 命令一览

| 命令 | 说明 |
|------|------|
| `bootstrap` | 初始化 Neo4j schema，幂等 |
| `train` | 将一篇论文提炼为灵感和问题，写入图谱 |
| `retrieve` | 用自然语言描述搜索图谱，返回相关节点 |
| `inspect` | 查看某个节点的全部细节（精化链 + 关联边 + 贡献论文） |
| `random` | 随机探索，打破检索排序的路径依赖 |
| `relate` | 查找两个节点之间的最短路径 |
| `delete-node` | 删除节点（含级联清理） |
| `reset` | 清空图谱（不可逆） |

## 训练

```bash
# 初始化图谱
uv run python main.py bootstrap

# 训练论文（支持 arXiv ID、标题、AMiner ID）
uv run python main.py train 1706.03762
uv run python main.py train "Attention Is All You Need"
uv run python main.py train ViT
```

`bootstrap` 是幂等的——已经初始化过再跑也不会有副作用。

## 检索

```bash
# 用自然语言搜索相关灵感和问题
uv run python main.py retrieve "使用扩散模型做医学图像分割的少样本学习"

# 调整参数
uv run python main.py retrieve "few-shot learning" --top-k 5 --final-limit 10
```

输出精简视图（id、类型、得分、来源、核心描述）。agent 据此判断哪些节点值得深入。

## 深度查看

```bash
# 查看单个节点的全字段 + 精化链 + 关联边
uv run python main.py inspect insp-abc123

# 同时查看多个节点
uv run python main.py inspect insp-abc123,q-xyz789
```

输出完整属性（前提条件、操作步骤、已知实例等）、精化链（更粗↔更细）、所有关联边及其目标节点信息。

## 随机探索

```bash
# 全图均匀随机
uv run python main.py random --count 5

# 主题加权随机（在相关范围内随机抽样）
uv run python main.py random --query "跨模态注意力" --count 3
```

目的是为头脑风暴引入意外发现，打破纯检索排序的路径依赖。

## 路径查询

```bash
uv run python main.py relate insp-abc123 insp-xyz789

# 限制路径长度
uv run python main.py relate insp-abc123 insp-xyz789 --max-len 4
```

查找两个节点之间的最短路径，返回中间节点序列和边序列。

## 删除节点

```bash
# 对论文、灵感、问题统一接口
uv run python main.py delete-node insp-abc123
uv run python main.py delete-node paper-1706.03762
```

级联规则：删除 Paper 时自动清理仅依赖它的实践节点；删除实践节点时自动清理仅依赖它的 Paper。

## 重置

```bash
uv run python main.py reset
```

删除所有灵感和问题节点及关联边，同时清空论文去重库。**不可逆。**

## 日志

```bash
# 开发调试
LOG_LEVEL=DEBUG uv run python main.py train 1706.03762

# 仅关键信息
LOG_LEVEL=INFO uv run python main.py bootstrap
```

不设 `LOG_LEVEL` 时默认静默（`WARNING`），`print()` 输出结果到 stdout，状态消息通过 logger 输出到 stderr。

## CLI 规格

命令输出格式的完整 JSON schema 见 [`docs/superpowers/specs/2025-05-30-cli-spec.md`](superpowers/specs/2025-05-30-cli-spec.md)。
