"""论文记录解析与降级。"""

from __future__ import annotations

from typing import Any

from src.config import Config
from src.neo4j.client import Neo4jClient
from src.neo4j.schema import reset_practice_graph
from src.paper.discovery import AMinerClient
from src.paper.extractor import ArxivExtractor


def load_paper_record(config: Config, paper_id: str) -> dict[str, Any]:
    """优先读 AMiner，失败时降级到 arXiv。"""

    title = ""
    text = ""
    paper: dict[str, Any] = {}

    try:
        paper = AMinerClient(config).get_paper_detail(paper_id)
        title = paper.get("title", "")
        text = paper.get("abstract_slice") or paper.get("abstract") or title
    except Exception:
        title = paper_id
        text = ""
        paper = {}

    if len(text) < config.arxiv_short_abstract_threshold:
        try:
            arxiv = ArxivExtractor(config)
            find_arxiv_id = getattr(arxiv, "find_arxiv_id", None)
            arxiv_id = paper_id
            if callable(find_arxiv_id):
                arxiv_id = (
                    find_arxiv_id({"title": title, "paper_id": paper_id}) or paper_id
                )
            full_text = arxiv.fetch_full_text(arxiv_id)
            if full_text:
                text = full_text
        except Exception:
            pass

    if len(text) > 12000:
        text = text[:12000]

    return {
        "paper_id": paper_id,
        "title": title or paper_id,
        "text": text,
        "paper": paper,
    }


def build_practice_summary(client: Neo4jClient, limit: int = 12) -> str:
    """汇总当前实践库，供 LLM A 判断。"""

    def _records(result: Any) -> list[Any]:
        if result is None:
            return []
        return list(result)

    with client.driver.session(database=client.config.neo4j_database) as session:

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
