# IdeaForgeX

[English](README.md) | [简体中文](docs/README_zh.md)

IdeaForgeX turns research papers into structured innovation candidates with an LLM-first workflow.

It reads paper text, extracts ideas into a Neo4j graph, and supports both training and inference flows for paper innovation mining.

## What It Does

- Discover and load papers from AMiner, with arXiv fallback for full text
- Build a practice graph in Neo4j with `Inspiration` and `Question` nodes
- Run an LLM A-only training loop to decide whether a paper can be recorded directly or needs structured candidate generation
- Run inference with retrieval + traversal over the practice graph
- Keep the whole system testable with local Docker services

## Why It Exists

Most paper-to-idea systems either stop at summaries or require too much manual curation. IdeaForgeX aims to keep the workflow structured, reproducible, and graph-backed so ideas can be reused, traced, and improved over time.

## Highlights

- Single LLM A workflow, no feedback loop noise
- Neo4j-backed knowledge graph for ideas and questions
- Local Docker setup for test and personal databases
- AMiner + arXiv fallback paper loading
- Full regression test suite

## Project Structure

- `src/agent/` workflow graphs for training and inference
- `src/llm/` prompt construction and chat client
- `src/paper/` AMiner, arXiv, and paper loading helpers
- `src/neo4j/` schema, maintenance, and retrieval logic
- `tests/` regression coverage for the main flows
- `doc/` design, architecture, and data model docs

## Requirements

- Python 3.11+
- `uv`
- Docker and Docker Compose
- A running Neo4j instance, or the provided compose services

## Quick Start

```bash
docker compose up -d
cp config_example.yaml config.yaml
uv run python main.py bootstrap
uv run python main.py train <paper_id>
uv run python main.py infer <paper_id>
```

Example:

```bash
uv run python main.py train 1706.03762
uv run python main.py infer 1706.03762
```

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

- Training and inference are implemented as LangGraph state machines.
- LLM output is validated through Pydantic models before any Neo4j writes.
- If validation fails, the workflow retries according to the configured policy.

## License

This project is licensed under the GNU AGPLv3. See `LICENSE` for details.

## Contributing

Issues and pull requests are welcome.

Before opening a PR, please make sure:

- `uv run pytest -q` passes
- `bootstrap`, `train`, and `infer` work locally
- New behavior is covered by tests when practical
