# IdeaForgeX

LLM 论文创新点生成系统。

## 运行

```bash
docker compose up -d
uv run python main.py bootstrap
uv run python main.py train <paper_id>
uv run python main.py infer <paper_id>
```

## Neo4j

- `neo4j-test`：`bolt://localhost:7687`
- `neo4j-personal`：`bolt://localhost:7688`
- 清库：`./scripts/neo4j-reset.sh test` 或 `./scripts/neo4j-reset.sh personal`

## 测试

```bash
./scripts/test.sh
```
