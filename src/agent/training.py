"""训练工作流。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from src.agent.common import (
    parse_llm_a_candidate,
    parse_query_text,
    validate_and_fix_refinement_edges,
)
from src.config import Config
from src.llm.client import ChatClient
from src.llm.prompts import build_llm_a_judge_messages, build_query_generation_messages
from src.llm.service import call_with_retry
from src.models import Edge, InspirationNode, NodeUpdate, PaperNode, QuestionNode
from src.neo4j.client import Neo4jClient
from src.neo4j.retrieval import retrieve_with_traversal
from src.neo4j.schema import (
    batch_update,
    batch_write,
    create_paper,
    create_paper_contributes_edge,
)
from src.paper.resolver import (
    build_practice_summary,
    paper_from_record,
    resolve_paper_spec,
)

_logger = logging.getLogger("ideaforgex")


class _TrainingStateRequired(TypedDict):
    paper_id: str


class TrainingState(_TrainingStateRequired, total=False):
    paper_title: str
    paper_year: str
    paper_text: str
    paper_record: (
        dict | None
    )  # 预加载的 PaperRecord（来自 JSONL），跳过 resolve_paper_spec
    query_text: str
    retrieved_nodes: list[dict]
    can_infer: bool
    already_trained: bool
    retry_count: int
    llm_a: dict
    inspirations: list[dict]
    questions: list[dict]
    edges: list[dict]
    node_updates: list[dict]
    messages: Annotated[list, add_messages]


def _make_paper_id(raw_id: str) -> str:
    """将原始论文 ID 转为 Paper 节点 ID。"""
    return f"paper-{raw_id}"


def build_training_graph(config: Config, client: ChatClient, neo4j_client: Neo4jClient):
    def load_paper(state: TrainingState) -> dict:
        record = state.get("paper_record")
        if record:
            _logger.info(
                "使用预加载论文数据: %s",
                record.get("title", record.get("paper_id", "")),
            )
            pr = paper_from_record(record)
            return {
                "paper_id": pr["paper_id"],
                "paper_title": pr["title"],
                "paper_year": pr["year"],
                "paper_text": pr["text"],
            }
        paper_text = state.get("paper_text", "")
        if paper_text:
            return {}
        paper_id = state["paper_id"]
        _logger.info("正在获取论文 …")
        resolved = resolve_paper_spec(config, paper_id)
        _logger.info("论文获取完成: %s", resolved["title"])
        return {
            "paper_id": resolved["paper_id"],
            "paper_title": resolved["title"],
            "paper_year": resolved.get("year", ""),
            "paper_text": resolved["text"],
        }

    def check_duplicate(state: TrainingState) -> dict:
        paper_node_id = _make_paper_id(state["paper_id"])
        with neo4j_client.session() as session:
            result = session.run(
                "MATCH (p:Paper {id: $id}) RETURN p.trained_at AS trained_at",
                id=paper_node_id,
            )
            if result.single():
                _logger.info("论文已训练，跳过: %s", state["paper_id"])
                return {"already_trained": True}
            return {"already_trained": False}

    def route_after_dup(state: TrainingState) -> str:
        return "skip" if state.get("already_trained") else "generate_query"

    def generate_query(state: TrainingState) -> dict:
        _logger.info("正在生成检索查询 …")
        messages = build_query_generation_messages(state.get("paper_text", ""))
        result = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.llm_temperature,
            parser=parse_query_text,
        )
        _logger.info("检索查询生成完成")
        return {"query_text": result["query_text"]}

    def retrieve(state: TrainingState) -> dict:
        _logger.info("正在检索知识图谱 …")
        embeddings = client.embed([state.get("query_text", "")])
        nodes = retrieve_with_traversal(neo4j_client, embeddings[0], config)
        _logger.info("知识图谱检索完成，命中 %d 条", len(nodes))
        return {"retrieved_nodes": nodes}

    def llm_a_judge(state: TrainingState) -> dict:
        _logger.info("正在 LLM 分析，判断可提炼性 …")
        practice_summary = build_practice_summary(neo4j_client)
        messages = build_llm_a_judge_messages(
            state.get("paper_text", ""),
            practice_summary,
            state.get("retrieved_nodes", []),
            paper_title=state.get("paper_title", ""),
            paper_year=state.get("paper_year", ""),
        )
        payload = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.llm_temperature,
            parser=parse_llm_a_candidate,
        )
        _logger.info(
            "LLM 分析完成，%s",
            "需要提取新节点" if not payload.can_infer else "可直接推断，无需提取",
        )
        return {
            "llm_a": payload.model_dump(),
            "can_infer": payload.can_infer,
            "inspirations": [node.model_dump() for node in payload.inspiration_nodes],
            "questions": [node.model_dump() for node in payload.question_nodes],
            "edges": [edge.model_dump() for edge in payload.edges],
            "node_updates": [upd.model_dump() for upd in payload.node_updates],
        }

    def route_after_llm_a(state: TrainingState) -> str:
        return "commit_candidates"

    def commit_candidates(state: TrainingState) -> dict:
        can_infer = state.get("can_infer", False)
        raw_paper_id = state["paper_id"]
        paper_id = _make_paper_id(raw_paper_id)
        paper_title = state.get("paper_title", "")
        paper_year = state.get("paper_year", "")
        paper_text = state.get("paper_text", "")

        paper_node = PaperNode(
            id=paper_id,
            title=paper_title,
            year=paper_year or "",
            abstract=paper_text[:2000],
            trained_at=datetime.now(timezone.utc).isoformat(),
        )

        if can_infer:
            _logger.info("可直接推断，仅登记论文")
            with neo4j_client.driver.session(database=config.neo4j_database) as session:
                session.execute_write(create_paper, paper_node)
            return {}

        _logger.info("正在提交候选人到图谱 …")
        inspirations = [
            InspirationNode.model_validate(item)
            for item in state.get("inspirations", [])
        ]
        questions = [
            QuestionNode.model_validate(item) for item in state.get("questions", [])
        ]
        edges = [Edge.model_validate(item) for item in state.get("edges", [])]
        node_updates = [
            NodeUpdate.model_validate(item) for item in state.get("node_updates", [])
        ]

        edges = validate_and_fix_refinement_edges(
            edges, inspirations, state.get("retrieved_nodes", [])
        )

        # 为 LLM 生成的 ID 添加 paper_id 前缀，避免并发训练 ID 冲突
        prefix = f"{raw_paper_id}__"
        old_to_new: dict[str, str] = {}

        for insp in inspirations:
            old_to_new[insp.id] = prefix + insp.id
            insp.id = old_to_new[insp.id]
        for q in questions:
            old_to_new[q.id] = prefix + q.id
            q.id = old_to_new[q.id]
        for edge in edges:
            if edge.from_id in old_to_new:
                edge.from_id = old_to_new[edge.from_id]
            if edge.to_id in old_to_new:
                edge.to_id = old_to_new[edge.to_id]

        # 用真实 embedding 替换向量占位符
        all_descs = [n.核心描述 for n in inspirations] + [q.核心描述 for q in questions]
        if all_descs:
            real_embeddings = client.embed(all_descs)
            insp_count = len(inspirations)
            for i, emb in enumerate(real_embeddings[:insp_count]):
                inspirations[i].向量 = emb
            for j, emb in enumerate(real_embeddings[insp_count:]):
                questions[j].向量 = emb

        with neo4j_client.driver.session(database=config.neo4j_database) as session:

            def _commit(tx) -> None:
                create_paper(tx, paper_node)

                if node_updates:
                    batch_update(tx, node_updates)
                    for upd in node_updates:
                        if upd.node_id:
                            create_paper_contributes_edge(tx, paper_id, upd.node_id)

                if inspirations or questions:
                    batch_write(tx, inspirations, questions, edges)
                    for insp in inspirations:
                        create_paper_contributes_edge(tx, paper_id, insp.id)
                    for q in questions:
                        create_paper_contributes_edge(tx, paper_id, q.id)

            session.execute_write(_commit)

        _logger.info(
            "候选人提交完成: +%d 灵感, +%d 问题, +%d 边, 更新 %d 节点",
            len(inspirations),
            len(questions),
            len(edges),
            len(node_updates),
        )
        return {}

    def skip_training(state: TrainingState) -> dict:
        return {}

    graph = StateGraph(TrainingState)
    graph.add_node("load_paper", load_paper)
    graph.add_node("check_duplicate", check_duplicate)
    graph.add_node("generate_query", generate_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node(
        "llm_a_judge",
        llm_a_judge,
        retry_policy=RetryPolicy(max_attempts=config.max_retries),
    )
    graph.add_node("commit_candidates", commit_candidates)
    graph.add_node("skip", skip_training)

    graph.add_edge(START, "load_paper")
    graph.add_edge("load_paper", "check_duplicate")
    graph.add_conditional_edges(
        "check_duplicate",
        route_after_dup,
        {"skip": "skip", "generate_query": "generate_query"},
    )
    graph.add_edge("skip", END)
    graph.add_edge("generate_query", "retrieve")
    graph.add_edge("retrieve", "llm_a_judge")
    graph.add_conditional_edges(
        "llm_a_judge",
        route_after_llm_a,
        {"commit_candidates": "commit_candidates"},
    )
    graph.add_edge("commit_candidates", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_training(graph, paper_id: str) -> dict:
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id}, config)


def run_training_with_record(graph, paper_id: str, record: dict) -> dict:
    """使用预加载的论文记录训练（跳过 resolve_paper_spec）。"""
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id, "paper_record": record}, config)
