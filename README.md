# IdeaForgeX

[English](README.md) | [简体中文](docs/README_zh.md)

AI 论文创新点知识图谱系统。训练时通过 LLM 从论文中提炼方法灵感和研究问题，写入 Neo4j 知识图谱；推理时外部 agent 通过 CLI 检索图数据，自行编排创新点生成。

## What It Does

- Discover and load papers from AMiner, with arXiv fallback for full text
- Build a practice graph in Neo4j with `Inspiration` and `Question` nodes
- Run an LLM A-only training loop to decide whether a paper can be recorded directly or needs structured candidate generation
- Expose CLI search commands (`retrieve` / `inspect`) for external agents to query the knowledge graph
- Keep the whole system testable with local Docker services

## Why It Exists

Most paper-to-idea systems either stop at summaries or require too much manual curation. IdeaForgeX keeps the workflow structured, reproducible, and graph-backed so ideas can be reused, traced, and improved over time. The inference side is intentionally thin — just a query API — so external agents can assemble their own innovation pipelines on top.

## Highlights

- Single LLM A workflow for training, no feedback loop noise
- Neo4j-backed knowledge graph for ideas and questions
- CLI-only inference: `retrieve` and `inspect` commands for external agents
- Local Docker setup for test and personal databases
- AMiner + arXiv fallback paper loading
- Full regression test suite

## Project Structure

- `src/agent/` — LangGraph training workflow
- `src/llm/` — prompt construction and chat client
- `src/paper/` — AMiner, arXiv, and paper loading helpers
- `src/neo4j/` — schema, maintenance, and retrieval logic
- `src/cli/` — CLI query commands (`retrieve` / `inspect`)
- `tests/` — regression coverage for the main flows
- `docs/` — design, architecture, data model, and CLI spec docs

## Requirements

- Python 3.11+
- `uv`
- Docker and Docker Compose
- A running Neo4j instance, or the provided compose services

## Quick Start

First time setup:

```bash
docker compose up -d
cp config_example.yaml config.yaml
uv run python main.py bootstrap
```

Usage:

```
uv run python main.py train <paper_id/keyword>
uv run python main.py retrieve <query_text>
uv run python main.py inspect <node_id>
```

Examples:

```bash
uv run python main.py train 1706.03762
uv run python main.py retrieve "使用扩散模型做医学图像分割的少样本学习"
uv run python main.py inspect insp-3f2a1...
```

CLI 命令详细规范见 `docs/use_cli.md`。

## Neo4j Targets

- `neo4j-test` uses `bolt://localhost:7687`
- `neo4j-personal` uses `bolt://localhost:7688`
- Reset a graph:

```bash
./scripts/neo4j-reset.sh test
./scripts/neo4j-reset.sh personal
```

## Testing

Run the full suite:

```bash
./scripts/test.sh
```

Or directly:

```bash
uv run pytest -q
```

## Configuration

All settings come from `config.yaml` and environment variables through `src/config.py`.

Key fields include:

- LLM base URL and API key
- embedding base URL and API key
- AMiner API key
- Neo4j URI, user, password, and database
- retrieval top-k and traversal settings

## Development Notes

- Training is implemented as a LangGraph state machine.
- LLM output is validated through Pydantic models before any Neo4j writes.
- If validation fails, the workflow retries according to the configured policy.
- The `retrieve` and `inspect` CLI commands are stateless; all orchestration is done by external agents.

## License

This project is licensed under the GNU AGPLv3. See `LICENSE` for details.

## Contributing

Issues and pull requests are welcome.

Before opening a PR, please make sure:

- `uv run pytest -q` passes
- `bootstrap`, `train`, `retrieve`, and `inspect` work locally
- New behavior is covered by tests when practical
