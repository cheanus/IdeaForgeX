# src/paper — 论文获取模块约定

## 两个 API，不同职责

| API | 职责 | 认证 | 成本 |
|---|---|---|---|
| OpenAlex | 论文发现 + 摘要获取 | `OPENALEX_API_KEY` | $0.001/搜索（免费配额充足） |
| arXiv | 全文获取（备选） | 免费，无需认证 | 免费 |

## OpenAlex 客户端

OpenAlex 是 HTTP REST API，用 `httpx` 直接调，不引入 SDK。所有调用通过 `discovery.py` 的 `OpenAlexClient` 类封装。

### 摘要格式

OpenAlex 的 abstract 存储为 `abstract_inverted_index`（词→位置映射字典），客户端内置 `_reconstruct_abstract()` 还原为纯文本。

### 成本控制

| 操作 | API | 单价 | 使用频率 |
|---|---|---|---|
| 按标题搜索论文 | `GET /works?filter=title.search:` | $0.001 | 训练阶段，每批一次 |
| 单篇详情 | `GET /works/{id}` | $0.0001 | 按需 |

> OpenAlex 搜索返回完整 work 对象（含摘要），通常不需要额外的详情调用。

### 重试策略

`OpenAlexClient._request()` 内置指数退避重试（最多 3 次），处理 429 限流和网络错误。

## arXiv 全文获取（备选）

仅在 OpenAlex 摘要对 LLM A 不够用时触发。arXiv 搜索有速率限制（~1 req/3s），全文获取不阻塞训练主流程。

## 论文解析流程

`resolver.py` 提供 `resolve_paper_spec()` 实现多级降级解析：

1. arXiv ID 格式 → 直接 arXiv 查询 → `paper_id = "arxiv-{id}"`
2. arXiv 标题搜索 → 全文 PDF 降级 → `paper_id = "arxiv-{id}"`
3. OpenAlex 搜索 → `paper_id = "openalex-{work_id}"`

所有 Paper 节点 ID 统一为 `paper-{paper_id}` 格式（如 `paper-arxiv-1706.03762`、`paper-openalex-W3167826310`）。

## 文件职责

| 文件 | 职责 |
|---|---|
| `discovery.py` | `OpenAlexClient` 论文搜索与详情获取 |
| `extractor.py` | `ArxivExtractor` — arXiv API 查询 + pymupdf PDF 全文提取 |
| `resolver.py` | `resolve_paper_spec` 多级降级解析 + `build_practice_summary` |
