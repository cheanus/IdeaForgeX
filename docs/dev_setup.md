# 开发环境指南

面向开发者和贡献者的环境搭建、测试与运维说明。

## 环境要求

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) 包管理器
- Docker + Docker Compose（运行 Neo4j）

## 安装与启动

```bash
# 1. 克隆并安装依赖
git clone https://github.com/<your-org>/IdeaForgeX.git
cd IdeaForgeX
uv sync

# 2. 启动 Neo4j（测试库 7687 + 个人库 7688，见 docker-compose.yml）
docker compose up -d

# 3. 从模板创建配置文件
cp config.example.yaml config.yaml
```

编辑 `config.yaml` 填入 API 密钥（LLM、Embedding、OpenAlex、Neo4j 密码）。

```bash
# 4. 初始化图谱 schema（幂等）
uv run ifx bootstrap
```

完整 CLI 使用指南见 [`docs/use_cli.md`](use_cli.md)。

## 项目结构

| 目录 | 职责 |
|---|---|
| `src/agent/` | LangGraph 训练工作流 |
| `src/llm/` | prompt 模板与 chat/embedding 客户端 |
| `src/paper/` | OpenAlex 发现、arXiv PDF 提取、论文解析 |
| `src/neo4j/` | schema 初始化、检索遍历、图维护 |
| `src/cli/` | CLI 查询命令 |
| `src/server/` | FastAPI 只读 HTTP 服务 |
| `skills/` | 外部 agent 技能定义 |
| `tests/` | 回归测试 |
| `docs/` | 设计与使用文档 |

## 运行测试

```bash
uv run pytest -v
```

测试覆盖训练、检索、CLI 输出、配置和 LLM 客户端。需 Neo4j 实例运行中。

### 新增测试规范

- Neo4j 集成测试用 `@pytest.mark.neo4j` 标记，使用测试库（默认连接 `bolt://localhost:7687`）
- 纯单元测试无标记，不依赖外部服务
- 新功能应有对应的测试覆盖

## 重置图谱

测试或开发过程中需要清空数据时：

```bash
# 清空测试库（port 7687）
./scripts/neo4j-reset.sh test

# 清空个人库（port 7688）
./scripts/neo4j-reset.sh personal
```

脚本使用 `MATCH (n) DETACH DELETE n` 删除整个库的所有节点和边。

> CLI 内置的 `reset` 命令只删除 `Inspiration` / `Question` / `Paper` 三种应用节点，保留 Neo4j 约束和索引。见 [`docs/use_cli.md`](use_cli.md)。

## 配置

所有参数通过 `config.yaml` 配置（默认值见 `config.example.yaml`）。
环境变量前缀 `IDEAFORGEX_` 可在运行时覆盖任意字段。

| 区域 | 关键字段 |
|---|---|
| LLM | `llm_base_url`、`llm_api_key`、`llm_model_name` |
| Embedding | `embedding_base_url`、`embedding_api_key`、`embedding_model_name`、`embedding_dim` |
| 论文 | `openalex_api_key`、`short_abstract_threshold` |
| Neo4j | `neo4j_uri`、`neo4j_user`、`neo4j_password`、`neo4j_database` |
| 检索 | `k_hits`、`max_neighbors`、`max_depth`、`score_decay`、`final_k` |
| 日志 | `log_level` — `DEBUG` / `INFO` / `WARNING` / `ERROR`（可用 `LOG_LEVEL` 环境变量覆盖） |

## 日志调试

```bash
# 开发调试
LOG_LEVEL=DEBUG uv run ifx train 1706.03762

# 仅关键信息
LOG_LEVEL=INFO uv run ifx bootstrap
```

不设 `LOG_LEVEL` 时默认 `WARNING`，`print()` 输出结果到 stdout，状态消息通过 logger 输出到 stderr。

## 提交 PR

- `uv run pytest -v` 通过
- `bootstrap`、`train`、`retrieve`、`inspect`、`random`、`relate` 在本地可运行
- 新行为应有测试覆盖
