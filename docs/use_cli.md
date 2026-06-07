# CLI 使用指南

训练时：把论文喂给知识图谱。推理时：外部 AI agent 通过 CLI 查询图数据，自行编排创新点。

## 命令一览

| 命令 | 说明 |
|------|------|
| `bootstrap` | 初始化 Neo4j schema，幂等 |
| `train` | 将一篇论文提炼为灵感和问题，写入图谱 |
| `batch-train` | 并行训练多篇论文，定期压缩合并重复节点 |
| `compact` | 合并图谱中语义相近的重复节点 |
| `retrieve` | 用自然语言描述搜索图谱，返回相关节点 |
| `inspect` | 查看某个节点的全部细节（精化链 + 关联边 + 贡献论文） |
| `random` | 随机探索，打破检索排序的路径依赖 |
| `relate` | 查找两个节点之间的最短路径 |
| `stats` | 查看图谱规模统计（节点数、粒度分布、边类型等） |
| `server` | 启动 FastAPI HTTP 只读服务，对外暴露查询接口 |
| `delete-node` | 删除节点（含级联清理） |
| `reset` | 清空图谱（不可逆） |

## 训练

```bash
# 初始化图谱
uv run ifx bootstrap

# 训练论文（支持 arXiv ID、标题）
uv run ifx train 1706.03762
uv run ifx train "Attention Is All You Need"
uv run ifx train ViT
```

`bootstrap` 是幂等的——已经初始化过再跑也不会有副作用。

## 检索

```bash
# 用自然语言搜索相关灵感和问题
uv run ifx retrieve "使用扩散模型做医学图像分割的少样本学习"

# 调整参数
uv run ifx retrieve "few-shot learning" --top-k 5 --final-limit 10
```

输出精简视图（id、类型、得分、来源、核心描述）。agent 据此判断哪些节点值得深入。

## 深度查看

```bash
# 查看单个节点的全字段 + 精化链 + 关联边
uv run ifx inspect insp-abc123

# 同时查看多个节点
uv run ifx inspect insp-abc123,q-xyz789
```

输出完整属性（前提条件、操作步骤等）、精化链（更粗↔更细）、所有关联边及其目标节点信息。

## 随机探索

```bash
# 全图均匀随机
uv run ifx random --count 5

# 主题加权随机（在相关范围内随机抽样）
uv run ifx random --query "跨模态注意力" --count 3
```

目的是为头脑风暴引入意外发现，打破纯检索排序的路径依赖。

## 路径查询

```bash
uv run ifx relate insp-abc123 insp-xyz789

# 限制路径长度
uv run ifx relate insp-abc123 insp-xyz789 --max-len 4
```

查找两个节点之间的最短路径，返回中间节点序列和边序列。

## 图谱统计

```bash
uv run ifx stats
```

输出 JSON 格式的统计概览：各类节点数、粒度分布、问题类型分布、边类型分布、论文年份分布。

## HTTP 只读服务

```bash
# 启动服务（默认 0.0.0.0:2048）
uv run ifx server

# 自定义地址和端口
uv run ifx server --host 127.0.0.1 --port 3000
```

启动后访问 `http://host:port/docs` 查看交互式 API 文档。

### 暴露的端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 健康检查 |
| `POST` | `/retrieve` | 图检索，JSON body 传参 |
| `GET` | `/inspect/{ids}` | 节点详情（支持逗号分隔多个 ID），`?expand_edges=false` 跳过边展开 |
| `GET` | `/stats` | 图谱统计 |
| `GET` | `/random?count=5&query=xxx` | 随机探索，`query` 可选 |
| `GET` | `/relate/{from_id}/{to_id}?max_len=6` | 最短路径查询 |

所有端点参数与 CLI 同名命令一致。不暴露任何写操作（train、batch-train、delete-node、compact、bootstrap、reset）。CORS 默认允许所有来源，可通过 `config.yaml` 中 `server_cors_origins` 限制。

### 配置

```yaml
# config.yaml
server_host: "0.0.0.0"       # 绑定地址
server_port: 2048            # 监听端口
server_cors_origins: ["*"]   # CORS 允许来源
```

启动时通过 `--host` / `--port` 命令行参数可覆盖配置文件中的值。

## 并行训练

```bash
# 命令行直接列多篇
uv run ifx batch-train 1706.03762 2010.11929 2103.00020 2103.14030

# 从文件读取（每行一个论文标识）
uv run ifx batch-train --queries queries.txt

# 从 JSONL 文件读取（跳过 API 解析，直接使用预加载内容）
uv run ifx batch-train --jsonl papers.jsonl
# JSONL 格式：每行一个 JSON 对象，支持以下字段：
#   必填: "id"（唯一标识，任意字符串，支持 prefix:body 格式如 arxiv:xxxx
#         或 openalex:Wxxxx；也可直接用自定义 ID 如 my-paper-001），
#         "title", "abstract"
#   可选: "year"
#   id 用作论文的唯一键，重复提交自动跳过
#   示例: {"id": "arxiv:1706.03762", "title": "Attention Is All You Need", "abstract": "...", "year": "2017"}
#   # 开头的行和空行自动跳过；传 - 则从 stdin 读取（支持 pipe）
# OpenAlex dump 转 JSONL 见：scripts/openalex_to_jsonl.py

# 混用
uv run ifx batch-train 1706.03762 --queries queries.txt
uv run ifx batch-train --queries queries.txt --jsonl papers.jsonl
```

并行训练用多线程同时训练多篇论文，瓶颈在 LLM API 调用（I/O）。每 `compact_interval` 篇新训练完成后自动触发一次图压缩，全部完成后执行最终压缩。

**断点续传**：`batch-train` 支持中断后重新运行同一命令。
- 已完成的论文在 Neo4j 中有 Paper 节点，重启后自动跳过
- 中断时正在执行的论文因 `check_duplicate` 为只读 MATCH 查询，未落库任何脏数据，重启后正常重训
- `commit_candidates` 阶段若进程断开，Neo4j 原子事务自动回滚，不残留半写状态

| 参数 | 类型 | 说明 |
|---|---|---|
| `papers` | string[] | 论文标识列表（arXiv ID 或标题），命令行直接传入 |
| `--queries` | string | 文件路径，每行一个论文标识（纯文本） |
| `--jsonl` | string | JSONL 文件路径（或 `-` 表示 stdin），每行一条 JSON 记录，跳过 OpenAlex/arXiv API 直接使用预加载内容。必填字段：`id`（自定义字符串，支持 `prefix:body` 格式，如 `arxiv:xxxx`/`openalex:Wxxxx`，也可直接用自定义 ID）、`title`、`abstract`；可选字段：`year`。`id` 用作论文唯一键，重复自动跳过。以 `#` 开头的行和空行自动跳过 |

输出 JSON 格式的结果摘要：
```json
{
  "总论文数": 20,
  "成功数": 18,
  "失败数": 2,
  "成功列表": ["1706.03762", "2010.11929", ...],
  "失败列表": [
    {"论文": "2301.00001", "错误": "LLM JSON 解析失败"}
  ]
}
```

## 图压缩

```bash
# 执行全图压缩，合并语义相近的重复节点
uv run ifx compact

# 仅报告将合并的节点，不实际执行
uv run ifx compact --dry-run
```

并行训练可能产生语义相近的重复节点（并发窗口内互不可见），`compact` 负责事后合并。合并保持细化链树结构（仅同粒度且上游一致的节点才合并）。

输出 JSON 格式的压缩报告：
```json
{
  "merged_inspirations": {"总量": 5, "粒1": 2, "粒2": 2, "粒3": 1},
  "merged_questions": {"总量": 3},
  "removed_duplicate_edges": 4,
  "dry_run": false
}
```

## 删除节点

```bash
# 对论文、灵感、问题统一接口
uv run ifx delete-node insp-abc123
uv run ifx delete-node paper-1706.03762
```

级联规则：删除 Paper 时自动清理仅依赖它的实践节点；删除实践节点时自动清理仅依赖它的 Paper。

## 重置

```bash
uv run ifx reset
```

删除所有灵感和问题节点及关联边，同时清空论文去重库。**不可逆。**

## 日志

```bash
# 开发调试
LOG_LEVEL=DEBUG uv run ifx train 1706.03762

# 仅关键信息
LOG_LEVEL=INFO uv run ifx bootstrap
```

不设 `LOG_LEVEL` 时默认静默（`WARNING`），`print()` 输出结果到 stdout，状态消息通过 logger 输出到 stderr。

## CLI 规格

命令输出格式的完整 JSON schema 见 [`docs/superpowers/specs/2025-05-30-cli-spec.md`](superpowers/specs/2025-05-30-cli-spec.md)。
