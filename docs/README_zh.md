# IdeaForgeX

AI 论文创新点知识图谱系统。训练时通过 LLM 从论文中提炼方法灵感和研究问题，写入 Neo4j 知识图谱；推理时外部 agent 通过 CLI 检索图数据，自行编排创新点生成。

## 功能

- 从 AMiner 发现并加载论文，必要时回退到 arXiv 获取全文
- 在 Neo4j 中构建实践图，包含 `Inspiration` 和 `Question` 节点
- 运行 LLM A-only 的训练循环，判断论文是直接记录还是需要生成结构化候选项
- 暴露 CLI 查询命令（`retrieve` / `inspect`），供外部 agent 检索知识图谱
- 提供可在本地 Docker 服务上测试的完整回归测试集

## 目标

大多数将论文转化为想法的系统要么止于摘要，要么需要过多人工整理。IdeaForgeX 旨在保持工作流结构化、可复现，并以图为后端，让想法可以被重用、追踪与持续改进。推理侧刻意轻量化，仅提供查询 API，外部 agent 可自行编排创新管线。

## 亮点

- 单一 LLM A 训练工作流，避免反馈回路噪声
- 基于 Neo4j 的知识图谱，用于管理想法与问题
- CLI 纯查询推理：`retrieve` / `inspect` 命令供外部 agent 使用
- 本地 Docker 支持用于测试与个人数据库
- AMiner + arXiv 回退机制用于论文加载
- 完整的回归测试套件

## 项目结构

- `src/agent/` — LangGraph 训练工作流
- `src/llm/` — 提示构建与聊天客户端
- `src/paper/` — AMiner、arXiv 与论文加载工具
- `src/neo4j/` — schema、维护与检索逻辑
- `src/cli/` — CLI 查询命令（`retrieve` / `inspect`）
- `tests/` — 主要流程的回归覆盖
- `docs/` — 设计、架构、数据模型与 CLI 规范文档

## 要求

- Python 3.11+
- `uv`
- Docker 与 Docker Compose
- 运行中的 Neo4j 实例，或使用仓库提供的 compose 服务

## 快速开始

第一次运行：

```bash
docker compose up -d
cp config_example.yaml config.yaml
uv run python main.py bootstrap
```

使用：

```bash
uv run python main.py train <paper_id>
uv run python main.py retrieve <查询文本>
uv run python main.py inspect <节点id>
```

示例：

```bash
uv run python main.py train 1706.03762
uv run python main.py retrieve "使用扩散模型做医学图像分割的少样本学习"
uv run python main.py inspect insp-3f2a1...
```

CLI 命令详细规范见 `docs/use_cli.md`。

## Neo4j 目标

- `neo4j-test` 使用 `bolt://localhost:7687`
- `neo4j-personal` 使用 `bolt://localhost:7688`
- 重置图：

```bash
./scripts/neo4j-reset.sh test
./scripts/neo4j-reset.sh personal
```

## 测试

运行全部测试：

```bash
./scripts/test.sh
```

或直接：

```bash
uv run pytest -q
```

## 配置

所有设置来自 `config.yaml` 与环境变量，通过 `src/config.py` 加载。

主要字段包括：

- LLM base URL 与 API key
- embedding base URL 与 API key
- AMiner API key
- Neo4j URI、用户、密码与数据库
- 检索 top-k 与遍历设置

## 开发说明

- 训练以 LangGraph 状态机实现。
- LLM 输出通过 Pydantic 模型验证后再写入 Neo4j。
- 若验证失败，工作流会根据配置的重试策略重试。
- `retrieve` 和 `inspect` CLI 命令为无状态设计，编排逻辑由外部 agent 完成。

## 许可证

本项目采用 GNU AGPLv3 许可。详见 `LICENSE`。

## 贡献

欢迎提交 issue 与 PR。

在打开 PR 之前，请确保：

- `uv run pytest -q` 通过
- `bootstrap`、`train`、`retrieve`、`inspect` 在本地可运行
- 在适当情况下为新行为添加测试覆盖
