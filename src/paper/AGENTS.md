# src/paper — 论文获取模块约定

## 两个 API，不同职责

| API | 职责 | 认证 | 成本 |
|---|---|---|---|
| AMiner | 论文发现 + 摘要获取 + 查重 | `AMINER_API_KEY` | ¥0.05~0.10/批 |
| arXiv | 全文获取（备选） | 免费，无需认证 | 免费 |

## AMiner 客户端

AMiner 是 HTTP REST API，用 `httpx` 直接调。不引入 SDK。base URL 通过 `config.aminer_base_url` 配置。

```python
import httpx

class AMinerClient:
    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": api_key,
            "X-Platform": "ideaforgex",
            "Content-Type": "application/json;charset=utf-8"
        }

    # ── 论文发现 ──

    def search_papers(self, query: str, limit: int = 50) -> list[dict]:
        """paper_qa_search: 语义搜索论文"""
        resp = httpx.post(
            f"{self.base_url}/api/paper/qa/search",
            headers=self.headers,
            json={"query": query, "size": limit, "sci_flag": True}
        )
        return resp.json()["data"]

    # ── 摘要获取 ──

    def get_abstracts_batch(self, ids: list[str]) -> list[dict]:
        """paper_info: 批量获取摘要"""
        resp = httpx.post(
            f"{self.base_url}/api/paper/info",
            headers=self.headers,
            json={"ids": ids}
        )
        return resp.json()["data"]

    def get_paper_detail(self, paper_id: str) -> dict:
        """paper_detail: 单篇完整元数据"""
        resp = httpx.get(
            f"{self.base_url}/api/paper/detail?id={paper_id}",
            headers=self.headers
        )
        return resp.json()["data"]

    # ── 查重 ──

    def dedup_search(self, idea_text: str, limit: int = 5) -> list[dict]:
        """用创新点描述搜索已有论文"""
        return self.search_papers(idea_text, limit=limit)
```

### 成本控制

| 操作 | API | 单价 | 使用频率 |
|---|---|---|---|
| 按主题发现论文 | `paper_qa_search` | ¥0.05 | 训练阶段，每批一次 |
| 批量获取摘要 | `paper_info` | **免费** | 训练阶段，每批一次 |
| 单篇详细元数据 | `paper_detail` | ¥0.01 | 按需，仅目标论文 |
| 文献查重 | `paper_qa_search` | ¥0.05 | 推理阶段，每次 1~2 次 |

> `paper_info` 是免费批量 API，返回 `abstract_slice`。训练流程主路径用它。仅当摘要不足需要完整 abstract/authors/keywords 时才降级到 `paper_detail`。

## arXiv 全文获取（备选）

仅在 AMiner 摘要对 LLM A 不够用时触发。

```python
import re

ARXIV_API = "https://export.arxiv.org/api/query"

def find_arxiv_id(paper: dict) -> str | None:
    """从 AMiner 返回的 paper 中提取 arXiv ID"""
    # paper["doi"] 或 paper["title"] 搜索 arXiv
    title = paper.get("title", "")
    resp = httpx.get(ARXIV_API, params={
        "search_query": f"ti:{title}",
        "max_results": 1
    })
    # 解析 Atom XML 提取 arXiv ID
    match = re.search(r'<id>http://arxiv\.org/abs/([^<]+)</id>', resp.text)
    return match.group(1) if match else None

def fetch_full_text(arxiv_id: str) -> str:
    """web_extract arXiv PDF → markdown 文本"""
    # 由调用方通过 web_extract 工具执行
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    return web_extract(url)  # 伪代码，实际在 agent 层调用
```

> arXiv 搜索有速率限制（~1 req/3s），全文获取不阻塞训练主流程。

## 论文解析流程

实际训练/推理中，`resolver.py` 提供 `resolve_paper_spec()` 实现多级降级解析：
1. arXiv ID 格式 → 直接 arXiv 查询
2. AMiner ID 直接查询
3. AMiner 语义搜索（按标题）
4. arXiv 标题搜索 → 全文 PDF 降级

```python
from src.paper.resolver import resolve_paper_spec

record = resolve_paper_spec(config, "Attention Is All You Need")
# => {"paper_id": ..., "title": ..., "text": ..., "paper": ...}
```

## 文件职责

| 文件 | 职责 |
|---|---------|
| `discovery.py` | `AMinerClient` 论文搜索与详情获取 |
| `extractor.py` | `ArxivExtractor` — arXiv API 查询 + pymupdf PDF 全文提取 |
| `resolver.py` | `resolve_paper_spec` 多级降级解析 + `build_practice_summary` |
