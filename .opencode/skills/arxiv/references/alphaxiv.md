# alphaXiv — superset alternative to raw arXiv

[alphaXiv](https://alphaxiv.org) wraps all arXiv papers with AI-generated summaries, similar-paper recommendations, and a clean JSON API. Free. Requires an API key (generated in the alphaXiv web app), set as `ALPHAXIV_API_KEY`.

Python SDK: [alphaxiv-py](https://github.com/petroslamb/alphaxiv-py) (`uv add alphaxiv-py`).

## When to prefer alphaXiv over raw arXiv

| Scenario | Use |
|---|---|
| Need AI summary, not just original abstract | alphaXiv `paper summary` |
| Need similar/recommended papers | alphaXiv `paper similar` |
| Want machine-readable JSON, not Atom XML | alphaXiv `--json` |
| Building an agent pipeline (search → read → analyze) | alphaXiv Python SDK |
| Quick one-off paper lookup by title | raw arXiv `curl` is fine |

## Key operations (Python SDK, async)

```python
from alphaxiv import AlphaXivClient

client = AlphaXivClient(api_key="axv1_...")

# Search
papers = await client.search.papers("graph neural networks", limit=50)

# AI summary (richer than original abstract)
summary = await client.paper.summary("2402.03300")

# Original abstract
abstract = await client.paper.abstract("2402.03300")

# Full text (paginated)
text = await client.paper.text("2402.03300", page=1)

# Similar papers
similar = await client.paper.similar("2402.03300", limit=10)
```

## For IdeaForgeX-style pipelines

The optimal path: `search` → `paper summary` (AI summary) + `paper abstract` (original) → LLM analysis. Only fall back to `paper text` (full body) when the summaries lack enough detail for the LLM task.

For literature dedup: use `search.papers(idea_text)` to check if an innovation idea already exists.
