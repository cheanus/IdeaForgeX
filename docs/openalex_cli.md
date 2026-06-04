## 概述

[OpenAlex](https://openalex.org) 是一个开放学术图谱（270M+ 论文），提供免费 API。其官方 CLI `openalex-official` 的 `download` 子命令支持按过滤器批量下载论文元数据，配合本项目的转换脚本，可高效构建大规模训练数据。

## 流水线

```
openalex download 下载元数据
        ↓
scripts/openalex_to_jsonl.py 转换为 JSONL
        ↓
batch-train --jsonl 训练入库
```

## 前置条件

1. 安装 OpenAlex CLI：
```bash
pip install openalex-official
```
2. 免费 API Key（申请：https://openalex.org/settings/api），设环境变量：
```bash
export OPENALEX_API_KEY=***
```

## 第一步：下载元数据

### 常用过滤器

`openalex download --filter` 透传 OpenAlex API 的 filter 语法：

| 过滤器 | 示例 | 说明 |
|---|---|---|
| 发表日期范围 | `from_publication_date:2023-06-01` | ISO 日期 |
| 发表年份 | `publication_year:2024` | 精确年 |
| 会议/期刊 ID | `primary_location.source.id:S2764453411` | 按 venue 过滤 |
| Topic ID | `topics.id:T10325` | 按研究主题过滤 |
| 论文类型 | `type:article` | article / book / dataset |

**语法：** AND 用逗号 `,`，OR 用竖线 `|`，范围用 `year:2020-2024`，比较用 `cited_by_count:>50`。

### 示例：下载 CCF-A 人工智能会议论文

| 会议 | Source ID |
|---|---|
| AAAI | S5407048695 |
| NeurIPS | S4306420609 |
| ICML | S4306419644 |
| CVPR | S4363607701 |
| ICCV | S4363607764 |
| ACL | S2729999759 |
| IJCAI | S4306419999 |

```bash
openalex download \
  --api-key "$OPENALEX_API_KEY" \
  --output ./ccf-a-ai-2021-202606 \
  --filter "from_publication_date:2021-01-01,primary_location.source.id:S5407048695|S4306420609|S4306419644|S4363607701|S4363607764|S2729999759|S4306419999"
```

### 输出结构

```
ccf-a-ai-2021-202606/
├── W2741809807.json    # 每篇论文一个文件
├── W3947180291.json
└── .openalex-checkpoint.json   # 断点续传标记
```

每个 `.json` 文件的核心字段：

```json
{
  "id": "https://openalex.org/W3138516171",
  "display_name": "Swin Transformer: ...",
  "publication_year": 2021,
  "abstract_inverted_index": {
    "This": [0],
    "paper": [1],
    "proposes": [2]
  }
}
```

## 第二步：转换为 JSONL

```bash
python scripts/openalex_to_jsonl.py ./ccf-a-ai-2021-202606 -o papers.jsonl
```

字段映射：

| OpenAlex JSON | JSONL 字段 |
|---|---|
| `id` → `https://openalex.org/W3138516171` | `id` → `openalex:W3138516171` |
| `title` → `Swin Transformer: ...` | `title` → `Swin Transformer: ...` |
| `abstract_inverted_index` 解码 | `abstract` → 纯文本 |
| `publication_year` → `2021` | `year` → `2021`（可选）|

输出示例（每行一条 JSON）：

```json
{"id": "openalex:W3138516171", "title": "Swin Transformer: ...", "abstract": "This paper proposes ...", "year": "2021"}
```

## 第三步：训练入库

```bash
# 从文件
uv run python main.py batch-train --jsonl papers.jsonl

# 管道直连（跳过中间文件）
python scripts/openalex_to_jsonl.py ./ccf-a-ai-2021-202606 | uv run python main.py batch-train --jsonl -
```

## 大规模下载建议

- OpenAlex 免费 API 每天 $1 额度，metadata 下载免费
- CLI 支持断点续传（`.openalex-checkpoint.json`），中断后可重跑
- 默认并发 50 线程，可调 `--workers` 控制
- 建议先小范围测试过滤条件，再全量下载
