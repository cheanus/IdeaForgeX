"""训练工作流。"""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.types import RetryPolicy

from src.agent.common import build_failure_record, parse_llm_a_candidate, parse_llm_b_candidate, parse_llm_c_evaluation
from src.config import Config
from src.llm.client import ChatClient
from src.llm.prompts import build_llm_a_judge_messages, build_llm_b_generate_messages, build_llm_c_eval_messages
from src.llm.service import call_with_retry
from src.models import Edge, InnovationIdea, InspirationNode, QuestionNode
from src.neo4j.client import Neo4jClient
from src.neo4j.schema import batch_write, write_failure_record
from src.paper.discovery import AMinerClient
from src.paper.extractor import ArxivExtractor


class TrainingState(TypedDict, total=False):
    paper_id: str
    paper_title: str
    paper_text: str
    can_infer: bool
    paper_recorded: bool
    llm_a: dict
    llm_b: dict
    llm_c: dict
    inspirations: list[dict]
    questions: list[dict]
    edges: list[dict]
    candidate_ideas: list[dict]
    best_idea: dict | None
    evaluation: dict | None
    failure_reason: str
    retry_count: int
    messages: Annotated[list, add_messages]


def build_training_graph(config: Config, client: ChatClient, neo4j_client: Neo4jClient):
    def load_paper(state: TrainingState) -> dict:
        paper_text = state.get("paper_text", "")
        if paper_text:
            return {}
        paper_id = state["paper_id"]
        discovery = AMinerClient(config)
        paper = discovery.get_paper_detail(paper_id)
        text = paper.get("abstract_slice") or paper.get("abstract") or paper.get("title", "")
        if len(text) < config.arxiv_short_abstract_threshold:
            arxiv_text = ""
            extractor = ArxivExtractor(config)
            arxiv_id = extractor.find_arxiv_id(paper)
            if arxiv_id:
                arxiv_text = extractor.fetch_full_text(arxiv_id)
            if arxiv_text:
                text = arxiv_text
        return {"paper_title": paper.get("title", state.get("paper_title", "")), "paper_text": text}

    def llm_a_judge(state: TrainingState) -> dict:
        messages = build_llm_a_judge_messages(state["paper_text"], "实践库概要待接入")
        payload = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.temperature_llm_a_judge,
            parser=parse_llm_a_candidate,
        )
        return {
            "llm_a": payload.model_dump(),
            "can_infer": payload.can_infer,
            "inspirations": [node.model_dump() for node in payload.inspiration_nodes],
            "questions": [node.model_dump() for node in payload.question_nodes],
            "edges": [edge.model_dump() for edge in payload.edges],
        }

    def route_after_llm_a(state: TrainingState) -> Literal["record_paper", "generate_nodes"]:
        return "record_paper" if state.get("can_infer") else "generate_nodes"

    def record_paper(state: TrainingState) -> dict:
        with neo4j_client.driver.session(database=config.neo4j_database) as session:
            session.run("CREATE (:PaperRecord {id: $id, title: $title})", id=state["paper_id"], title=state.get("paper_title", ""))
        return {"paper_recorded": True}

    def generate_nodes(state: TrainingState) -> dict:
        messages = build_llm_b_generate_messages(state["paper_text"], "检索上下文待接入")
        payload = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.temperature_llm_b,
            parser=parse_llm_b_candidate,
        )
        return {
            "llm_b": payload.model_dump(),
            "candidate_ideas": [idea.model_dump() for idea in payload.ideas],
        }

    def evaluate_candidate(state: TrainingState) -> dict:
        ideas = state.get("candidate_ideas", [])
        if not ideas:
            return {"evaluation": {"passed": False, "comment": "没有候选创新点", "failure_reason": "无候选创新点"}, "best_idea": None, "failure_reason": "无候选创新点"}
        best = InnovationIdea.model_validate(ideas[0])
        messages = build_llm_c_eval_messages(state.get("paper_title", ""), state["paper_text"], best)
        payload = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.temperature_llm_c,
            parser=parse_llm_c_evaluation,
        )
        return {
            "llm_c": payload.model_dump(),
            "evaluation": payload.model_dump(),
            "best_idea": payload.best_idea.model_dump() if payload.best_idea else best.model_dump(),
            "failure_reason": payload.failure_reason,
        }

    def route_after_eval(state: TrainingState) -> Literal["commit_candidates", "retry_or_fail"]:
        evaluation = state.get("evaluation") or {}
        if evaluation.get("passed"):
            return "commit_candidates"
        return "retry_or_fail"

    def retry_or_fail(state: TrainingState) -> dict:
        retry_count = int(state.get("retry_count", 0)) + 1
        return {"retry_count": retry_count, "failure_reason": state.get("failure_reason", "LLM C 未通过")}

    def route_after_retry(state: TrainingState) -> Literal["generate_nodes", "record_failure"]:
        retry_count = int(state.get("retry_count", 0))
        if retry_count < config.max_reflection_rounds:
            return "generate_nodes"
        return "record_failure"

    def record_failure(state: TrainingState) -> dict:
        record = build_failure_record(
            paper_id=state["paper_id"],
            paper_title=state.get("paper_title", ""),
            failure_reason=state.get("failure_reason", "LLM C 未通过"),
            llm_c_eval=state.get("llm_c") or state.get("evaluation") or {},
            candidate_idea_snapshot=state.get("best_idea") or {},
        )
        with neo4j_client.driver.session(database=config.neo4j_database) as session:
            session.execute_write(write_failure_record, record.model_dump())
        return {"failure_reason": record.failure_reason}

    def commit_candidates(state: TrainingState) -> dict:
        inspirations = [InspirationNode.model_validate(item) for item in state.get("inspirations", [])]
        questions = [QuestionNode.model_validate(item) for item in state.get("questions", [])]
        edges = [Edge.model_validate(item) for item in state.get("edges", [])]
        with neo4j_client.driver.session(database=config.neo4j_database) as session:
            session.execute_write(batch_write, inspirations, questions, edges)
        return {"paper_recorded": True}

    graph = StateGraph(TrainingState)
    graph.add_node("load_paper", load_paper)
    graph.add_node("llm_a_judge", llm_a_judge, retry_policy=RetryPolicy(max_attempts=config.max_retries))
    graph.add_node("record_paper", record_paper)
    graph.add_node("generate_nodes", generate_nodes, retry_policy=RetryPolicy(max_attempts=config.max_retries))
    graph.add_node("evaluate_candidate", evaluate_candidate, retry_policy=RetryPolicy(max_attempts=config.max_retries))
    graph.add_node("retry_or_fail", retry_or_fail)
    graph.add_node("record_failure", record_failure)
    graph.add_node("commit_candidates", commit_candidates)

    graph.add_edge(START, "load_paper")
    graph.add_edge("load_paper", "llm_a_judge")
    graph.add_conditional_edges(
        "llm_a_judge",
        route_after_llm_a,
        {"record_paper": "record_paper", "generate_nodes": "generate_nodes"},
    )
    graph.add_edge("record_paper", END)
    graph.add_edge("generate_nodes", "evaluate_candidate")
    graph.add_conditional_edges(
        "evaluate_candidate",
        route_after_eval,
        {"commit_candidates": "commit_candidates", "retry_or_fail": "retry_or_fail"},
    )
    graph.add_conditional_edges(
        "retry_or_fail",
        route_after_retry,
        {"generate_nodes": "generate_nodes", "record_failure": "record_failure"},
    )
    graph.add_edge("record_failure", END)
    graph.add_edge("commit_candidates", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_training(graph, paper_id: str) -> dict:
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id, "retry_count": 0}, config)
