#!/usr/bin/env bash
set -euo pipefail

target="${1:-test}"
case "$target" in
  test)
    uri="bolt://localhost:7687"
    password="test-password"
    ;;
  personal)
    uri="bolt://localhost:7688"
    password="personal-password"
    ;;
  *)
    printf 'unknown target: %s\n' "$target" >&2
    exit 1
    ;;
esac

uv run python - <<PY
from neo4j import GraphDatabase

driver = GraphDatabase.driver("$uri", auth=("neo4j", "$password"))
try:
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
finally:
    driver.close()
PY
