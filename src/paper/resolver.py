"""论文记录解析与降级。

resolve_paper_spec() 接受论文 ID 或标题，支持来源前缀（arxiv:/openalex:）和 ID 自动识别。
无前缀时的路由优先级：来源前缀 > ID 正则匹配 > 标题搜索（OpenAlex 优先，arXiv 兜底）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from typing_extensions import TypedDict

from src.config import Config
from src.neo4j.client import Neo4jClient

_logger = logging.getLogger("ideaforgex")
from src.paper.discovery import OpenAlexClient, _extract_work_id, _reconstruct_abstract
from src.paper.extractor import ArxivExtractor

_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$|^[a-z-]+/\d{7}(?:v\d+)?$")
_OPENALEX_ID_PATTERN = re.compile(r"^W\d{7,}$", re.IGNORECASE)
_PREFIX_PATTERN = re.compile(r"^([a-zA-Z][\w-]*):(.+)", re.IGNORECASE)
_MAX_TEXT_LENGTH = 12000


class PaperRecord(TypedDict):
    paper_id: str
    title: str
    text: str
    year: str
    paper: dict[str, Any]


def _strip_prefix(spec: str) -> tuple[str | None, str]:
    """解析来源前缀，返回 (来源, 无前缀文本)。无前缀时来源为 None。"""
    m = _PREFIX_PATTERN.match(spec)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None, spec


def _is_arxiv_id(spec: str) -> bool:
    """判断输入是否为 arXiv ID 格式。"""
    return bool(_ARXIV_ID_PATTERN.match(spec.strip()))


def _is_openalex_id(spec: str) -> bool:
    """判断输入是否为 OpenAlex work ID 格式（W + 数字）。"""
    return bool(_OPENALEX_ID_PATTERN.match(spec.strip()))


def _resolve_text(
    config: Config, extractor: ArxivExtractor, arxiv_id: str, abstract: str
) -> str:
    """必要时降级到全文 PDF 获取更长文本。"""
    if len(abstract) < config.short_abstract_threshold:
        try:
            full_text = extractor.fetch_full_text(arxiv_id)
            if full_text:
                return full_text
        except Exception as exc:
            _logger.warning("arXiv 全文获取失败，降级为摘要: %s", exc)
    return abstract


def _load_from_arxiv(config: Config, arxiv_id: str) -> PaperRecord:
    """已知 arXiv ID，直接从 arXiv 获取论文数据。"""
    extractor = ArxivExtractor(config)
    detail = extractor.get_paper_detail(arxiv_id)
    title = detail.get("title", arxiv_id)
    text = _resolve_text(config, extractor, arxiv_id, detail.get("abstract", ""))
    year = detail.get("year", "")
    return {
        "paper_id": f"arxiv-{arxiv_id}",
        "title": title or arxiv_id,
        "text": text,
        "year": year,
        "paper": detail,
    }


def _load_from_openalex(config: Config, work_id: str) -> PaperRecord:
    """已知 OpenAlex work ID，直接查询论文详情。"""
    client = OpenAlexClient(config)
    short_id = _extract_work_id(work_id)
    paper = client.get_paper_detail(short_id)
    title = paper.get("title", work_id)
    abstract = _reconstruct_abstract(paper.get("abstract_inverted_index"))
    year = str(paper.get("publication_year", ""))
    text = abstract
    if len(abstract) < config.short_abstract_threshold and short_id:
        _logger.info("OpenAlex 摘要过短，尝试下载 PDF …")
        full_text = client.download_pdf(short_id)
        if full_text:
            text = full_text
    return {
        "paper_id": f"openalex-{short_id}" if short_id else f"openalex-{work_id}",
        "title": title,
        "text": text,
        "year": year,
        "paper": paper,
    }


def _try_openalex_search(config: Config, query: str) -> PaperRecord | None:
    """通过 OpenAlex 标题搜索查找论文，返回第一篇的详细信息。"""
    client = OpenAlexClient(config)
    papers = client.search_papers(query, limit=3)
    if not papers:
        return None
    best = papers[0]
    work_id = best.get("id", "")
    short_id = _extract_work_id(work_id) if work_id else ""
    title = best.get("title", query)
    abstract = _reconstruct_abstract(best.get("abstract_inverted_index"))
    year = str(best.get("publication_year", ""))
    text = abstract
    if len(abstract) < config.short_abstract_threshold and short_id:
        _logger.info("OpenAlex 摘要过短，尝试下载 PDF …")
        full_text = client.download_pdf(short_id)
        if full_text:
            text = full_text
    return {
        "paper_id": f"openalex-{short_id}" if short_id else query,
        "title": title,
        "text": text,
        "year": year,
        "paper": best,
    }


def _try_arxiv_fallback(config: Config, title: str) -> PaperRecord | None:
    """通过 arXiv 标题搜索获取论文，必要时降级到全文 PDF。"""
    extractor = ArxivExtractor(config)
    arxiv_id = extractor.find_arxiv_id({"title": title})
    if not arxiv_id:
        return None
    detail = extractor.get_paper_detail(arxiv_id)
    result_title = detail.get("title", title)
    text = _resolve_text(config, extractor, arxiv_id, detail.get("abstract", ""))
    year = detail.get("year", "")
    return {
        "paper_id": f"arxiv-{arxiv_id}",
        "title": result_title,
        "text": text,
        "year": year,
        "paper": detail,
    }


def _enforce_text_limit(text: str) -> str:
    if len(text) > _MAX_TEXT_LENGTH:
        return text[:_MAX_TEXT_LENGTH]
    return text


def resolve_paper_spec(config: Config, spec: str) -> PaperRecord:
    """智能解析论文描述，支持来源前缀、ID 正则识别和多级降级。

    路由优先级：
    1. 来源前缀（arxiv: / openalex:） → 直查，失败报错
    2. ID 正则匹配 → 优先直查，失败降级到标题搜索
    3. 标题搜索 → OpenAlex 优先，arXiv 兜底
    """
    spec = spec.strip()

    # ── 1. 来源前缀 ──
    source, raw_spec = _strip_prefix(spec)
    if source == "arxiv":
        _logger.info("来源前缀 arxiv:，直查 arXiv …")
        result = _load_from_arxiv(config, raw_spec)
        result["text"] = _enforce_text_limit(result["text"])
        return result
    if source == "openalex":
        _logger.info("来源前缀 openalex:，直查 OpenAlex …")
        result = _load_from_openalex(config, raw_spec)
        result["text"] = _enforce_text_limit(result["text"])
        return result

    # ── 2. ID 正则优先直查（失败则继续降级） ──
    if _is_arxiv_id(spec):
        try:
            result = _load_from_arxiv(config, spec)
            result["text"] = _enforce_text_limit(result["text"])
            _logger.info("论文解析完成，数据源: arXiv ID 直查")
            return result
        except Exception as exc:
            _logger.info("arXiv ID 查询失败: %s，继续降级 …", exc)

    if _is_openalex_id(spec):
        try:
            result = _load_from_openalex(config, spec)
            result["text"] = _enforce_text_limit(result["text"])
            _logger.info("论文解析完成，数据源: OpenAlex ID 直查")
            return result
        except Exception as exc:
            _logger.info("OpenAlex ID 查询失败: %s，继续降级 …", exc)

    # ── 3. 标题搜索：OpenAlex 优先，arXiv 兜底 ──
    _logger.info("尝试标题搜索，OpenAlex 优先 …")
    try:
        result = _try_openalex_search(config, spec)
        if result and result["text"].strip():
            result["text"] = _enforce_text_limit(result["text"])
            _logger.info("论文解析完成，数据源: OpenAlex 标题搜索")
            return result
    except Exception as exc:
        _logger.warning("OpenAlex 标题搜索失败: %s", exc)

    _logger.info("OpenAlex 未命中，尝试 arXiv 标题搜索 …")
    try:
        result = _try_arxiv_fallback(config, spec)
        if result and result["text"].strip():
            result["text"] = _enforce_text_limit(result["text"])
            _logger.info("论文解析完成，数据源: arXiv 标题搜索")
            return result
    except Exception as exc:
        _logger.warning("arXiv 标题搜索失败: %s", exc)

    raise ValueError(f"无法解析论文 '{spec}'：所有数据源均失败，未能获取论文内容")


def build_practice_summary(client: Neo4jClient, limit: int = 12) -> str:
    """汇总当前实践库，供 LLM A 判断。"""

    def _records(result: Any) -> list[Any]:
        if result is None:
            return []
        return list(result)

    with client.session() as session:

        def _extract_node(record: Any) -> dict[str, Any]:
            if hasattr(record, "data"):
                data = record.data()
                if isinstance(data, dict):
                    node = data.get("n", data)
                    if isinstance(node, dict):
                        return node
                    if hasattr(node, "data"):
                        inner = node.data()
                        return inner if isinstance(inner, dict) else {}
            if isinstance(record, dict):
                node = record.get("n", record)
                if isinstance(node, dict):
                    return node
                if hasattr(node, "data"):
                    inner = node.data()
                    return inner if isinstance(inner, dict) else {}
            return {}

        inspiration_result = session.run(
            """
            MATCH (n:Inspiration)
            RETURN n
            LIMIT $limit
            """,
            limit=limit,
        )
        question_result = session.run(
            """
            MATCH (n:Question)
            RETURN n
            LIMIT $limit
            """,
            limit=limit,
        )

        inspirations = [
            f"{node.get('id', '')}: {node.get('核心描述') or node.get('core', '')} (粒度 {node.get('粒度', node.get('grain', 0))})"
            for record_ in _records(inspiration_result)
            for node in [_extract_node(record_)]
        ]
        questions = [
            f"{node.get('id', '')}: {node.get('核心描述') or node.get('core', '')} ({node.get('问题类型') or node.get('qtype', '理论缺口')})"
            for record_ in _records(question_result)
            for node in [_extract_node(record_)]
        ]

    sections: list[str] = []
    if inspirations:
        sections.append("Inspiration:\n" + "\n".join(inspirations))
    if questions:
        sections.append("Question:\n" + "\n".join(questions))
    return "\n\n".join(sections) or "暂无实践库节点"


def paper_from_record(record: dict[str, Any]) -> PaperRecord:
    """从预加载的 JSONL 行构建 PaperRecord（跳过 API 解析）。

    将冒号分隔的 id（如 arxiv:1706.03762）转为内部格式（arxiv-1706.03762）。
    """
    raw_id = record.get("id", "")
    if not raw_id:
        raise ValueError("JSONL 行缺少必填字段: id")
    source, body = _strip_prefix(raw_id)
    paper_id = f"{source}-{body}" if source else raw_id
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    if not title:
        raise ValueError(f"JSONL 行缺少必填字段: title (id={raw_id})")
    if not abstract:
        raise ValueError(f"JSONL 行缺少必填字段: abstract (id={raw_id})")
    return {
        "paper_id": paper_id,
        "title": title,
        "text": abstract,
        "year": str(record.get("year", "")),
        "paper": record,
    }
    """汇总当前实践库，供 LLM A 判断。"""

    def _records(result: Any) -> list[Any]:
        if result is None:
            return []
        return list(result)

    with client.session() as session:

        def _extract_node(record: Any) -> dict[str, Any]:
            if hasattr(record, "data"):
                data = record.data()
                if isinstance(data, dict):
                    node = data.get("n", data)
                    if isinstance(node, dict):
                        return node
                    if hasattr(node, "data"):
                        inner = node.data()
                        return inner if isinstance(inner, dict) else {}
            if isinstance(record, dict):
                node = record.get("n", record)
                if isinstance(node, dict):
                    return node
                if hasattr(node, "data"):
                    inner = node.data()
                    return inner if isinstance(inner, dict) else {}
            return {}

        inspiration_result = session.run(
            """
            MATCH (n:Inspiration)
            RETURN n
            LIMIT $limit
            """,
            limit=limit,
        )
        question_result = session.run(
            """
            MATCH (n:Question)
            RETURN n
            LIMIT $limit
            """,
            limit=limit,
        )

        inspirations = [
            f"{node.get('id', '')}: {node.get('核心描述') or node.get('core', '')} (粒度 {node.get('粒度', node.get('grain', 0))})"
            for record in _records(inspiration_result)
            for node in [_extract_node(record)]
        ]
        questions = [
            f"{node.get('id', '')}: {node.get('核心描述') or node.get('core', '')} ({node.get('问题类型') or node.get('qtype', '理论缺口')})"
            for record in _records(question_result)
            for node in [_extract_node(record)]
        ]

    sections: list[str] = []
    if inspirations:
        sections.append("Inspiration:\n" + "\n".join(inspirations))
    if questions:
        sections.append("Question:\n" + "\n".join(questions))
    return "\n\n".join(sections) or "暂无实践库节点"
