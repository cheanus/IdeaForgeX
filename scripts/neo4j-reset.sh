#!/usr/bin/env bash
set -euo pipefail

target="${1:-test}"
case "$target" in
  test)
    uri="bolt://localhost:7687"
    ;;
  personal)
    uri="bolt://localhost:7688"
    ;;
  *)
    printf 'unknown target: %s\n' "$target" >&2
    exit 1
    ;;
esac

uv run python - <<PY
from neo4j import GraphDatabase

driver = GraphDatabase.driver("$uri", auth=None)
try:
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
finally:
    driver.close()
PY
