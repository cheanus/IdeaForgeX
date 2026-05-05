"""推理工作流。"""

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
from src.llm.prompts import build_inference_messages
from src.llm.service import call_with_retry
from src.neo4j.client import Neo4jClient
from src.neo4j.retrieval import retrieve_with_traversal
from src.paper.resolver import load_paper_record


class InferenceState(TypedDict, total=False):
    paper_id: str
    paper_title: str
    paper_text: str
    query_text: str
    retrieved_nodes: list[dict]
    llm_a: dict | None
    messages: Annotated[list, add_messages]


def build_inference_graph(
    config: Config, client: ChatClient, neo4j_client: Neo4jClient
):
    def load_target_paper(state: InferenceState) -> dict:
        record = load_paper_record(config, state["paper_id"])  # type: ignore[reportTypedDictNotRequiredAccess]
        text = record.get("text", "")
        return {
            "paper_title": record.get("title", ""),
            "paper_text": text,
            "query_text": text,
        }

    def retrieve(state: InferenceState) -> dict:
        embeddings = client.embed([state["query_text"]])  # type: ignore[reportTypedDictNotRequiredAccess]
        nodes = retrieve_with_traversal(neo4j_client, embeddings[0], config)
        return {"retrieved_nodes": nodes}

    def apply_llm_a(state: InferenceState) -> dict:
        messages = build_inference_messages(
            state["paper_text"],  # type: ignore[reportTypedDictNotRequiredAccess]
            state.get("retrieved_nodes", []),
        )
        payload = call_with_retry(
            client,
            messages,
            max_retries=config.max_retries,
            temperature=config.temperature_llm_a_judge,
            parser=parse_llm_a_candidate,
        )
        return {"llm_a": payload.model_dump()}

    graph = StateGraph(InferenceState)
    graph.add_node("load_target_paper", load_target_paper)
    graph.add_node("retrieve", retrieve)
    graph.add_node(
        "apply_llm_a",
        apply_llm_a,
        retry_policy=RetryPolicy(max_attempts=config.max_retries),
    )

    graph.add_edge(START, "load_target_paper")
    graph.add_edge("load_target_paper", "retrieve")
    graph.add_edge("retrieve", "apply_llm_a")
    graph.add_edge("apply_llm_a", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_inference(graph, paper_id: str) -> dict:
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id}, config)
