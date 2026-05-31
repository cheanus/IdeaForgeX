# IdeaForgeX

[English](README.md) | [简体中文](README_zh.md)

**Open-source knowledge graph for AI research ideas.** Train by feeding papers and let LLMs distill reusable method inspirations and research questions into a Neo4j graph. At inference time, external AI agents query the graph via CLI to compose novel research directions.

[![License](https://img.shields.io/badge/license-AGPLv3-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/neo4j-community-green)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/framework-langgraph-orange)](https://langchain-ai.github.io/langgraph/)

---

## Why IdeaForgeX

Most paper-to-idea pipelines stop at summaries or require manual curation. IdeaForgeX builds a **structured, graph-backed knowledge base** where ideas are first-class entities — traceable, reusable, and improvable over time. The inference layer is intentionally thin (just a CLI query API) so external AI agents can assemble their own innovation workflows on top.

## Highlights

- 🧠 **One LLM, clean loop** — single LLM-A judge decides whether a paper yields extractable ideas or just needs recording. No feedback-loop noise.
- 🕸️ **Neo4j knowledge graph** — dual node types (`Inspiration` + `Question`), four edge types, multi-granularity refinement chains
- 🔌 **Agent-friendly CLI** — four query commands (`retrieve`, `inspect`, `random`, `relate`) return structured JSON for external agents
- 🐳 **Local Docker setup** — test and personal Neo4j instances via `docker compose`
- 📄 **OpenAlex + arXiv** — tiered paper resolution with automatic fallback
- ✅ **Full test suite** — `uv run pytest -v` covers training, retrieval, and CLI output

## Project Structure

| Directory | Purpose |
|---|---|
| `src/agent/` | LangGraph training workflow |
| `src/llm/` | Prompt templates and chat/embedding client |
| `src/paper/` | OpenAlex discovery, arXiv PDF extraction, paper resolver |
| `src/neo4j/` | Schema bootstrap, retrieval traversal, graph maintenance |
| `src/cli/` | CLI query commands (`retrieve` / `inspect` / `random` / `relate`) |
| `tests/` | Regression coverage for training, CLI, config, and LLM client |
| `docs/` | Design, architecture, data model, CLI usage guide |

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for Neo4j)

### Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/<your-org>/IdeaForgeX.git
cd IdeaForgeX
uv sync

# 2. Start Neo4j
docker compose up -d

# 3. Create config from template
cp config.example.yaml config.yaml
```

> Edit `config.yaml` and fill in your API keys:
> - `llm_api_key` — your LLM provider (DeepSeek, OpenAI, etc.)
> - `embedding_api_key` — your embedding provider
> - `openalex_api_key` — [OpenAlex](https://openalex.org/) API key for paper discovery
> - `neo4j_password` — Neo4j database password

```bash
# 4. Bootstrap the graph schema (idempotent)
uv run python main.py bootstrap
```

### Usage

```bash
# Train a paper into the knowledge graph
uv run python main.py train 1706.03762          # arXiv ID
uv run python main.py train "Attention Is All You Need"  # title search

# Query the graph (external agents call these commands)
uv run python main.py retrieve "few-shot learning with diffusion models"
uv run python main.py inspect INSP_4
uv run python main.py random --count 5
uv run python main.py random --query "cross-modal attention" --count 3
uv run python main.py relate INSP_1 INSP_10
```

For CLI usage guide, see [`docs/use_cli.md`](docs/use_cli.md). For full JSON schemas, see [`docs/superpowers/specs/2025-05-30-cli-spec.md`](docs/superpowers/specs/2025-05-30-cli-spec.md).

### Graph Reset

```bash
# Reset test or personal database
./scripts/neo4j-reset.sh test
./scripts/neo4j-reset.sh personal
```

## Testing

```bash
uv run pytest -v
```

## Configuration

All parameters live in `config.yaml` (see `config.example.yaml` for defaults). Environment variables prefixed with `IDEAFORGEX_` can override any field at runtime.

| Section | Key fields |
|---|---|
| LLM | `llm_base_url`, `llm_api_key`, `llm_model_name` |
| Embedding | `embedding_base_url`, `embedding_api_key`, `embedding_model_name`, `embedding_dim` |
| Paper | `openalex_api_key`, `short_abstract_threshold` |
| Neo4j | `neo4j_uri`, `neo4j_user`, `neo4j_password`, `neo4j_database` |
| Retrieval | `k_hits`, `max_neighbors`, `max_depth`, `score_decay`, `final_k` |
| Logging | `log_level` — `DEBUG` / `INFO` / `WARNING` / `ERROR` (overridden by `LOG_LEVEL` env var) |

## How It Works

**Training**: A LangGraph state machine loads a paper, generates a retrieval query, searches the graph, and calls the LLM to decide whether the paper introduces novel ideas worth extracting. New `Inspiration` and `Question` nodes are written transactionally into Neo4j.

**Inference**: External AI agents call CLI commands to explore the graph — `retrieve` for relevance-ranked search, `inspect` for deep dives, `random` for serendipity, and `relate` for path discovery. The agent then composes novelty proposals using its own LLM.

## License

This project is licensed under the GNU AGPLv3. See [`LICENSE`](LICENSE) for details.

## Contributing

Issues and pull requests are welcome. Before opening a PR:

- `uv run pytest -v` must pass
- `bootstrap`, `train`, `retrieve`, `inspect`, `random`, and `relate` should work locally
- New behavior should be covered by tests
