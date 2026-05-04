#!/usr/bin/env bash
set -euo pipefail

./scripts/neo4j-reset.sh test
uv run pytest -q
