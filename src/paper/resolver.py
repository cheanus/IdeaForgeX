"""论文记录解析与降级。

resolve_paper_spec() 接受论文 ID 或标题，按以下优先级尝试获取数据：
1. arXiv ID 格式 → 直接 arXiv 查询
2. arXiv 标题搜索 → 全文 PDF 降级
3. AMiner 语义搜索
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient

_logger = logging.getLogger("ideaforgex")
from src.paper.discovery import AMinerClient
from src.paper.extractor import ArxivExtractor

_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$|^[a-z-]+/\d{7}(?:v\d+)?$")
_MAX_TEXT_LENGTH = 12000


def _is_arxiv_id(spec: str) -> bool:
    """判断输入是否为 arXiv ID 格式。"""
    return bool(_ARXIV_ID_PATTERN.match(spec.strip()))


def _load_from_arxiv(config: Config, arxiv_id: str) -> dict[str, Any]:
    """已知 arXiv ID，直接从 arXiv 获取论文数据。"""
    extractor = ArxivExtractor(config)
    detail = extractor.get_paper_detail(arxiv_id)
    title = detail.get("title", arxiv_id)
    text = detail.get("abstract", "")

    if len(text) < config.arxiv_short_abstract_threshold:
        try:
            full_text = extractor.fetch_full_text(arxiv_id)
            if full_text:
                text = full_text
        except Exception:
            pass

    year = detail.get("year", "")
    return {
        "paper_id": arxiv_id,
        "title": title or arxiv_id,
        "text": text,
        "year": year,
        "paper": detail,
    }


def _try_aminer_search(config: Config, query: str) -> dict[str, Any] | None:
    """通过 AMiner 语义搜索查找论文，返回第一篇的详细信息。"""
    papers = AMinerClient(config).search_papers(query, limit=3)
    if not papers:
        return None
    best_id = papers[0].get("id")
    if not best_id:
        return None
    paper = AMinerClient(config).get_paper_detail(best_id)
    title = paper.get("title", query)
    text = paper.get("abstract_slice") or paper.get("abstract") or ""
    year = str(paper.get("year", ""))
    return {
        "paper_id": best_id,
        "title": title,
        "text": text,
        "year": year,
        "paper": paper,
    }


def _try_arxiv_fallback(config: Config, title: str) -> dict[str, Any] | None:
    """通过 arXiv 标题搜索获取论文，必要时降级到全文 PDF。"""
    extractor = ArxivExtractor(config)
    arxiv_id = extractor.find_arxiv_id({"title": title})
    if not arxiv_id:
        return None
    detail = extractor.get_paper_detail(arxiv_id)
    result_title = detail.get("title", title)
    text = detail.get("abstract", "")

    if len(text) < config.arxiv_short_abstract_threshold:
        try:
            full_text = extractor.fetch_full_text(arxiv_id)
            if full_text:
                text = full_text
        except Exception:
            pass

    year = detail.get("year", "")
    return {
        "paper_id": arxiv_id,
        "title": result_title,
        "text": text,
        "year": year,
        "paper": detail,
    }


def _enforce_text_limit(text: str) -> str:
    if len(text) > _MAX_TEXT_LENGTH:
        return text[:_MAX_TEXT_LENGTH]
    return text


def resolve_paper_spec(config: Config, spec: str) -> dict[str, Any]:
    """智能解析论文描述（ID 或标题），多级降级获取数据。

    优先级：
    1. arXiv ID 格式 → 直接 arXiv 查询
    2. arXiv 标题搜索 → 全文 PDF 降级
    3. AMiner 语义搜索
    """
    spec = spec.strip()

    # 优先级1：arXiv ID 快速路径
    if _is_arxiv_id(spec):
        try:
            result = _load_from_arxiv(config, spec)
            result["text"] = _enforce_text_limit(result["text"])
            _logger.info("论文解析完成，数据源: arXiv ID 直查")
            return result
        except Exception:
            _logger.info("arXiv ID 查询失败，尝试标题搜索 …")
        # arXiv ID 查询失败，继续尝试其他方式

    title = spec
    text = ""
    paper: dict[str, Any] = {}
    paper_id = spec

    # 优先级2：arXiv 标题搜索 + 全文降级
    try:
        result = _try_arxiv_fallback(config, title)
        if result:
            paper = result["paper"]
            title = result["title"]
            text = result["text"]
            paper_id = result["paper_id"]
    except Exception:
        pass

    if text.strip():
        _logger.info("论文解析完成，数据源: arXiv 标题搜索")

    # 优先级3：AMiner 语义搜索（arxiv 未获取到内容时启用）
    if not text.strip():
        _logger.info("arXiv 未获取到内容，尝试 AMiner 搜索 …")
        try:
            result = _try_aminer_search(config, spec)
            if result:
                paper = result["paper"]
                title = result["title"]
                text = result["text"]
                paper_id = result["paper_id"]
                _logger.info("论文解析完成，数据源: AMiner 语义搜索")
        except Exception:
            _logger.warning("AMiner 搜索也失败")

    if not text.strip():
        raise ValueError(f"无法解析论文 '{spec}'：所有数据源均失败，未能获取论文内容")

    year = paper.get("year", "")
    return {
        "paper_id": paper_id,
        "title": title or spec,
        "text": _enforce_text_limit(text),
        "year": year,
        "paper": paper,
    }


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
