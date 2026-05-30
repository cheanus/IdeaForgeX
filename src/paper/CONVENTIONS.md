# src/paper — 论文获取模块约定

## 两个 API，不同职责

| API | 职责 | 认证 | 成本 |
|---|---|---|---|
| AMiner | 论文发现 + 摘要获取 | `AMINER_API_KEY` | ¥0.05~0.10/批 |
| arXiv | 全文获取（备选） | 免费，无需认证 | 免费 |

## AMiner 客户端

AMiner 是 HTTP REST API，用 `httpx` 直接调，不引入 SDK。所有调用通过 `discovery.py` 的 `AMinerClient` 类封装。

### 成本控制

| 操作 | API | 单价 | 使用频率 |
|---|---|---|---|
| 按主题发现论文 | `paper_search` | ¥0.05 | 训练阶段，每批一次 |
| 批量获取摘要 | `paper_info` | **免费** | 训练阶段，每批一次 |
| 单篇详细元数据 | `paper_detail` | ¥0.01 | 按需，仅目标论文 |

> `paper_info` 是免费批量 API，返回 `abstract_slice`。训练流程主路径用它。仅当摘要不足需要完整 abstract/authors/keywords 时才降级到 `paper_detail`。

## arXiv 全文获取（备选）

仅在 AMiner 摘要对 LLM A 不够用时触发。arXiv 搜索有速率限制（~1 req/3s），全文获取不阻塞训练主流程。

## 论文解析流程

`resolver.py` 提供 `resolve_paper_spec()` 实现多级降级解析：

1. arXiv ID 格式 → 直接 arXiv 查询
2. arXiv 标题搜索 → 全文 PDF 降级
3. AMiner 语义搜索

## 文件职责

| 文件 | 职责 |
|---|---|
| `discovery.py` | `AMinerClient` 论文搜索与详情获取 |
| `extractor.py` | `ArxivExtractor` — arXiv API 查询 + pymupdf PDF 全文提取 |
| `resolver.py` | `resolve_paper_spec` 多级降级解析 + `build_practice_summary` |
