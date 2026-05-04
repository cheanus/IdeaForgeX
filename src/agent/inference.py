"""推理工作流。"""

from __future__ import annotations

from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.config import Config
from src.llm.client import ChatClient
from src.llm.prompts import build_llm_b_generate_messages, build_llm_c_eval_messages
from src.llm.service import call_chat_with_retry, call_with_retry
from src.models import InnovationIdea
from src.neo4j.client import Neo4jClient
from src.neo4j.retrieval import retrieve_with_traversal
from src.paper.discovery import AMinerClient


class InferenceState(TypedDict, total=False):
    paper_id: str
    paper_title: str
    paper_text: str
    query_text: str
    retrieved_nodes: list[dict]
    candidate_ideas: list[dict]
    best_idea: dict | None
    messages: Annotated[list, add_messages]


def build_inference_graph(config: Config, client: ChatClient, neo4j_client: Neo4jClient):
    def load_target_paper(state: InferenceState) -> dict:
        discovery = AMinerClient(config)
        paper = discovery.get_paper_detail(state["paper_id"])
        text = paper.get("abstract_slice") or paper.get("abstract") or paper.get("title", "")
        return {"paper_title": paper.get("title", ""), "paper_text": text, "query_text": text}

    def retrieve(state: InferenceState) -> dict:
        embeddings = client.embed([state["query_text"]])
        nodes = retrieve_with_traversal(neo4j_client, embeddings[0], config)
        return {"retrieved_nodes": nodes}

    def generate(state: InferenceState) -> dict:
        messages = build_llm_b_generate_messages(state["paper_text"], str(state.get("retrieved_nodes", [])))
        raw = call_chat_with_retry(client, messages, max_retries=config.max_retries, temperature=config.temperature_llm_b)
        return {"candidate_ideas": [{"title": raw[:40], "description": raw, "feasibility_score": 0.5, "novelty_score": 0.5, "value_score": 0.5}]}

    def evaluate(state: InferenceState) -> dict:
        ideas = state.get("candidate_ideas", [])
        if not ideas:
            return {"best_idea": None}
        best = InnovationIdea.model_validate(ideas[0])
        messages = build_llm_c_eval_messages(state.get("paper_title", ""), state["paper_text"], best)
        payload = call_with_retry(client, messages, max_retries=config.max_retries, temperature=config.temperature_llm_c)
        best_idea = payload.best_idea.model_dump() if getattr(payload, "best_idea", None) else best.model_dump()
        return {"best_idea": best_idea}

    graph = StateGraph(InferenceState)
    graph.add_node("load_target_paper", load_target_paper)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "load_target_paper")
    graph.add_edge("load_target_paper", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_inference(graph, paper_id: str) -> dict:
    config = {"configurable": {"thread_id": paper_id}}
    return graph.invoke({"paper_id": paper_id}, config)
