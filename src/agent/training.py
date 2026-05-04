"""训练工作流。"""

from __future__ import annotations

from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from src.agent.common import parse_llm_a_candidate
from src.config import Config
from src.llm.client import ChatClient
from src.llm.prompts import build_llm_a_judge_messages
from src.llm.service import call_with_retry
from src.models import Edge, InspirationNode, QuestionNode
from src.neo4j.client import Neo4jClient
from src.neo4j.schema import batch_write
from src.paper.resolver import build_practice_summary, load_paper_record


class TrainingState(TypedDict, total=False):
    paper_id: str
    paper_title: str
    paper_text: str
    can_infer: bool
    llm_a: dict
    inspirations: list[dict]
    questions: list[dict]
    edges: list[dict]
    messages: Annotated[list, add_messages]


def build_training_graph(config: Config, client: ChatClient, neo4j_client: Neo4jClient):
    def load_paper(state: TrainingState) -> dict:
        paper_text = state.get("paper_text", "")
        if paper_text:
            return {}
        paper_id = state["paper_id"]
        record = load_paper_record(config, paper_id)
        return {"paper_title": record["title"], "paper_text": record["text"]}

    def llm_a_judge(state: TrainingState) -> dict:
        practice_summary = build_practice_summary(neo4j_client)
        messages = build_llm_a_judge_messages(state["paper_text"], practice_summary)
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

    def route_after_llm_a(state: TrainingState) -> str:
        return "record_paper" if state.get("can_infer") else "commit_candidates"

    def record_paper(state: TrainingState) -> dict:
        with neo4j_client.driver.session(database=config.neo4j_database) as session:
            session.run(
                "CREATE (:PaperRecord {id: $id, title: $title})",
                id=state["paper_id"],
                title=state.get("paper_title", ""),
            )
        return {}

    def commit_candidates(state: TrainingState) -> dict:
        inspirations = [
            InspirationNode.model_validate(item)
            for item in state.get("inspirations", [])
        ]
        questions = [
            QuestionNode.model_validate(item) for item in state.get("questions", [])
        ]
        edges = [Edge.model_validate(item) for item in state.get("edges", [])]
        with neo4j_client.driver.session(database=config.neo4j_database) as session:
            session.execute_write(batch_write, inspirations, questions, edges)
        return {}

    graph = StateGraph(TrainingState)
    graph.add_node("load_paper", load_paper)
    graph.add_node(
        "llm_a_judge",
        llm_a_judge,
        retry_policy=RetryPolicy(max_attempts=config.max_retries),
    )
    graph.add_node("record_paper", record_paper)
    graph.add_node("commit_candidates", commit_candidates)

    graph.add_edge(START, "load_paper")
    graph.add_edge("load_paper", "llm_a_judge")
    graph.add_conditional_edges(
        "llm_a_judge",
        route_after_llm_a,
        {"record_paper": "record_paper", "commit_candidates": "commit_candidates"},
    )
    graph.add_edge("record_paper", END)
    graph.add_edge("commit_candidates", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_training(graph, paper_id: str) -> dict:
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id}, config)
