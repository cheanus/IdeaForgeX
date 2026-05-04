# IdeaForgeX

IdeaForgeX 将研究论文转换为结构化的创新候选项，采用以 LLM 为中心的工作流。

它读取论文文本，将想法提取到 Neo4j 图中，并支持用于论文创新挖掘的训练与推理流程。

## 功能

- 从 AMiner 发现并加载论文，必要时回退到 arXiv 获取全文
- 在 Neo4j 中构建实践图，包含 `Inspiration` 和 `Question` 节点
- 运行 LLM A-only 的训练循环，判断论文是直接记录还是需要生成结构化候选项
- 运行带检索与遍历的推理流程，基于实践图展开
- 提供可在本地 Docker 服务上测试的完整回归测试集

## 目标

大多数将论文转化为想法的系统要么止于摘要，要么需要过多人工整理。IdeaForgeX 旨在保持工作流结构化、可复现，并以图为后端，让想法可以被重用、追踪与持续改进。

## 亮点

- 单一 LLM A 工作流，避免反馈回路噪声
- 基于 Neo4j 的知识图谱，用于管理想法与问题
- 本地 Docker 支持用于测试与个人数据库
- AMiner + arXiv 回退机制用于论文加载
- 完整的回归测试套件

## 项目结构

- `src/agent/`：训练与推理的工作流图
- `src/llm/`：提示构建与聊天客户端
- `src/paper/`：AMiner、arXiv 与论文加载工具
- `src/neo4j/`：schema、维护与检索逻辑
- `tests/`：主要流程的回归覆盖
- `docs/`：设计、架构与数据模型文档

## 要求

- Python 3.11+
- `uv`
- Docker 与 Docker Compose
- 运行中的 Neo4j 实例，或使用仓库提供的 compose 服务

## 快速开始

```bash
docker compose up -d
cp config_example.yaml config.yaml
uv run python main.py bootstrap
uv run python main.py train <paper_id>
uv run python main.py infer <paper_id>
```

示例：

```bash
uv run python main.py train 1706.03762
uv run python main.py infer 1706.03762
```

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

- 训练与推理以 LangGraph 状态机实现。
- LLM 输出通过 Pydantic 模型验证后再写入 Neo4j。
- 若验证失败，工作流会根据配置的重试策略重试。

## 许可证

本项目采用 GNU AGPLv3 许可。详见 `LICENSE`。

## 贡献

欢迎提交 issue 与 PR。

在打开 PR 之前，请确保：

- `uv run pytest -q` 通过
- `bootstrap`、`train` 与 `infer` 在本地可运行
- 在适当情况下为新行为添加测试覆盖
