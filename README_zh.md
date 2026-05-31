# IdeaForgeX

[English](README.md) | [简体中文](README_zh.md)

**AI 论文创新点知识图谱系统** — 开源。训练时喂论文，让 LLM 提炼可复用的方法灵感和研究问题，写入 Neo4j 知识图谱；推理时外部 AI agent 通过 CLI 查询图数据，自行编排研究方向。

[![License](https://img.shields.io/badge/license-AGPLv3-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-community-green)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/framework-langgraph-orange)](https://langchain-ai.github.io/langgraph/)

---

## 为什么选 IdeaForgeX

多数论文转灵感的管线止于摘要或依赖大量人工整理。IdeaForgeX 构建了**结构化、以图为基础的知识库**，想法本身就是一等实体——可溯源、可复用、可持续改进。推理层刻意轻量（仅 CLI 查询 API），外部 AI agent 可自行搭建创新流水线。

## 亮点

- 🧠 **单一 LLM，干净闭环** — 由一个 LLM-A 判断论文是可提炼出创新点还是只需记录，无反馈回路噪声
- 🕸️ **Neo4j 知识图谱** — 双节点类型（`Inspiration` + `Question`），四种边，多粒度精化链
- 🔌 **Agent 友好 CLI** — 四个查询命令（`retrieve`、`inspect`、`random`、`relate`），返回结构化 JSON 供外部 agent 消费
- 🐳 **本地 Docker 支持** — 通过 `docker compose` 运行测试库和个人库 Neo4j 实例
- 📄 **OpenAlex + arXiv** — 分层论文解析，自动回退
- ✅ **完整回归测试** — `uv run pytest -v` 覆盖训练、检索、CLI 输出

## 项目结构

| 目录 | 职责 |
|---|---|
| `src/agent/` | LangGraph 训练工作流 |
| `src/llm/` | prompt 模板与 chat/embedding 客户端 |
| `src/paper/` | OpenAlex 发现、arXiv PDF 提取、论文解析 |
| `src/neo4j/` | schema 初始化、检索遍历、图维护 |
| `src/cli/` | CLI 查询命令（`retrieve` / `inspect` / `random` / `relate`） |
| `tests/` | 训练、CLI、配置、LLM 客户端的回归覆盖 |
| `docs/` | 设计、架构、数据模型、CLI 使用指南 |

## 快速开始

### 环境要求

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) 包管理器
- Docker + Docker Compose（运行 Neo4j）

### 安装

```bash
# 1. 克隆并安装依赖
git clone https://github.com/<your-org>/IdeaForgeX.git
cd IdeaForgeX
uv sync

# 2. 启动 Neo4j
docker compose up -d

# 3. 从模板创建配置文件
cp config.example.yaml config.yaml
```

> 编辑 `config.yaml`，填入 API 密钥：
> - `llm_api_key` — 你的 LLM 提供商（DeepSeek、OpenAI 等）
> - `embedding_api_key` — 你的 Embedding 提供商
> - `openalex_api_key` — [OpenAlex](https://openalex.org/) API 密钥，用于论文发现
> - `neo4j_password` — Neo4j 数据库密码

```bash
# 4. 初始化图谱 schema（幂等）
uv run python main.py bootstrap
```

### 使用

```bash
# 将论文训练进知识图谱
uv run python main.py train 1706.03762          # arXiv ID
uv run python main.py train "Attention Is All You Need"  # 标题搜索

# 查询知识图谱（外部 agent 调用这些命令）
uv run python main.py retrieve "使用扩散模型做少样本学习"
uv run python main.py inspect INSP_4
uv run python main.py random --count 5
uv run python main.py random --query "跨模态注意力" --count 3
uv run python main.py relate INSP_1 INSP_10
```

CLI 使用指南见 [`docs/use_cli.md`](docs/use_cli.md)。完整 JSON schema 见 [`docs/superpowers/specs/2025-05-30-cli-spec.md`](docs/superpowers/specs/2025-05-30-cli-spec.md)。

### 重置图

```bash
./scripts/neo4j-reset.sh test
./scripts/neo4j-reset.sh personal
```

## 测试

```bash
uv run pytest -v
```

## 配置

所有参数来自 `config.yaml`（默认值见 `config.example.yaml`）。环境变量前缀 `IDEAFORGEX_` 可在运行时覆盖任意字段。

| 区域 | 关键字段 |
|---|---|
| LLM | `llm_base_url`、`llm_api_key`、`llm_model_name` |
| Embedding | `embedding_base_url`、`embedding_api_key`、`embedding_model_name`、`embedding_dim` |
| 论文 | `openalex_api_key`、`arxiv_short_abstract_threshold` |
| Neo4j | `neo4j_uri`、`neo4j_user`、`neo4j_password`、`neo4j_database` |
| 检索 | `k_hits`、`max_neighbors`、`max_depth`、`score_decay`、`final_k` |
| 日志 | `log_level` — `DEBUG` / `INFO` / `WARNING` / `ERROR`（可用 `LOG_LEVEL` 环境变量覆盖） |

## 工作方式

**训练**：LangGraph 状态机加载论文 → 生成检索查询 → 搜索图谱 → LLM 判断是否产生新颖灵感 → 事务写入 Neo4j。

**推理**：外部 AI agent 调用 CLI 命令探索图谱——`retrieve` 按相关性排序检索、`inspect` 深度钻取、`random` 意外发现、`relate` 路径分析——然后用自身 LLM 组合生成创新方案。

## 许可证

本项目采用 GNU AGPLv3 许可。详见 [`LICENSE`](LICENSE)。

## 贡献

欢迎提交 issue 与 PR。提交 PR 前请确保：

- `uv run pytest -v` 通过
- `bootstrap`、`train`、`retrieve`、`inspect`、`random`、`relate` 在本地可运行
- 新行为应有测试覆盖
