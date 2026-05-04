from __future__ import annotations

from types import SimpleNamespace

from src.agent.common import parse_llm_a_candidate, parse_llm_b_candidate, parse_llm_c_evaluation
from src.agent.inference import build_inference_graph
from src.agent.training import build_training_graph
from src.config import Config
from src.models import Edge, InspirationNode, InnovationIdea, LLMACandidate, LLMBCandidate, LLMCEvaluation, QuestionNode


ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_ABSTRACT = (
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
    "dispensing with recurrence and convolutions."
)


class FakeSession:
    def __init__(self, log: list[tuple[str, tuple, dict]]):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, fn, *args):
        self.log.append((fn.__name__, args, {}))
        tx = SimpleNamespace(run=lambda *run_args, **run_kwargs: self.log.append(("run", run_args, run_kwargs)))
        return fn(tx, *args)

    def run(self, query, **kwargs):
        self.log.append((query, tuple(), kwargs))


class FakeDriver:
    def __init__(self):
        self.log: list[tuple[str, tuple, dict]] = []

    def session(self, database: str):
        self.log.append(("session", (database,), {}))
        return FakeSession(self.log)


class FakeNeo4jClient:
    def __init__(self, config: Config):
        self.config = config
        self.driver = FakeDriver()


class FakeAMinerClient:
    def __init__(self, config: Config):
        self.config = config

    def get_paper_detail(self, paper_id: str):
        return {
            "id": paper_id,
            "title": ATTENTION_TITLE,
            "abstract_slice": "short abstract",
        }


class FakeArxivExtractor:
    def __init__(self, config: Config):
        self.config = config

    def find_arxiv_id(self, paper: dict):
        return "1706.03762"

    def fetch_full_text(self, arxiv_id: str):
        return ATTENTION_ABSTRACT


def test_training_graph_commits_attention_sample(monkeypatch):
    config = Config(
        neo4j_database="neo4j-test",
        arxiv_short_abstract_threshold=200,
        max_reflection_rounds=1,
    )
    neo4j_client = FakeNeo4jClient(config)
    recorded: dict[str, object] = {}

    monkeypatch.setattr("src.agent.training.AMinerClient", FakeAMinerClient)
    monkeypatch.setattr("src.agent.training.ArxivExtractor", FakeArxivExtractor)

    def fake_call_with_retry(client, messages, max_retries=3, temperature=0.3, parser=None):
        if parser is parse_llm_a_candidate:
            return LLMACandidate(
                can_infer=False,
                inspiration_nodes=[
                    InspirationNode(
                        id="insp-1",
                        核心描述="Transformer uses only attention",
                        向量=[0.1, 0.2],
                        粒度=1,
                        前提条件="attention",
                        操作步骤="replace recurrence",
                        已知实例="Attention Is All You Need",
                    )
                ],
                question_nodes=[
                    QuestionNode(
                        id="q-1",
                        核心描述="How to remove recurrence from seq2seq",
                        向量=[0.3, 0.4],
                        问题类型="方法重构",
                        当前现状="RNN dominates",
                        未解决部分="Need an attention-only architecture",
                    )
                ],
                edges=[Edge(from_id="insp-1", to_id="q-1", rel_type="INSP_QUESTION", weight=0.8)],
            )
        if parser is parse_llm_b_candidate:
            return LLMBCandidate(
                ideas=[
                    InnovationIdea(
                        title="Attention-only seq2seq",
                        description="Use attention as the only core mechanism",
                        feasibility_score=0.9,
                        novelty_score=0.8,
                        value_score=0.9,
                    )
                ]
            )
        if parser is parse_llm_c_evaluation:
            return LLMCEvaluation(
                passed=True,
                best_idea=InnovationIdea(
                    title="Attention-only seq2seq",
                    description="Use attention as the only core mechanism",
                    feasibility_score=0.9,
                    novelty_score=0.8,
                    value_score=0.9,
                ),
                comment="pass",
            )
        raise AssertionError("unexpected parser")

    def fake_batch_write(tx, inspirations, questions, edges):
        recorded["batch_write"] = {
            "inspirations": inspirations,
            "questions": questions,
            "edges": edges,
        }

    monkeypatch.setattr("src.agent.training.call_with_retry", fake_call_with_retry)
    monkeypatch.setattr("src.agent.training.batch_write", fake_batch_write)

    graph = build_training_graph(config, SimpleNamespace(), neo4j_client)
    result = graph.invoke({"paper_id": "paper-1706.03762", "retry_count": 0}, {"configurable": {"thread_id": "paper-1706.03762"}})

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["paper_text"] == ATTENTION_ABSTRACT
    assert result["can_infer"] is False
    assert result["best_idea"]["title"] == "Attention-only seq2seq"
    assert len(recorded["batch_write"]["inspirations"]) == 1
    assert len(recorded["batch_write"]["questions"]) == 1
    assert len(recorded["batch_write"]["edges"]) == 1


def test_inference_graph_returns_attention_sample_best_idea(monkeypatch):
    config = Config(neo4j_database="neo4j-test")
    neo4j_client = SimpleNamespace(config=config, driver=SimpleNamespace())
    retrieved_nodes = [
        {"node": {"id": "insp-1", "type": "Inspiration", "核心描述": "attention-only", "向量": [0.1, 0.2]}, "score": 0.99},
        {"node": {"id": "q-1", "type": "Question", "核心描述": "gap to fill", "向量": [0.3, 0.4]}, "score": 0.88},
    ]

    class FakePaperClient:
        def __init__(self, config: Config):
            self.config = config

        def get_paper_detail(self, paper_id: str):
            return {
                "id": paper_id,
                "title": ATTENTION_TITLE,
                "abstract_slice": ATTENTION_ABSTRACT,
            }

    monkeypatch.setattr("src.agent.inference.AMinerClient", FakePaperClient)
    monkeypatch.setattr("src.agent.inference.retrieve_with_traversal", lambda client, embedding, cfg: retrieved_nodes)
    monkeypatch.setattr("src.agent.inference.call_chat_with_retry", lambda client, messages, max_retries=3, temperature=0.7: "attention-driven idea")
    monkeypatch.setattr(
        "src.agent.inference.call_with_retry",
        lambda client, messages, max_retries=3, temperature=0.3, parser=None: LLMCEvaluation(
            passed=True,
            best_idea=InnovationIdea(
                title="Attention-driven idea",
                description="Keep the attention core",
                feasibility_score=0.9,
                novelty_score=0.85,
                value_score=0.95,
            ),
            comment="pass",
        ),
    )

    class FakeInferenceClient:
        def __init__(self, config: Config):
            self.config = config

        def embed(self, texts: list[str]):
            return [[0.1, 0.2, 0.3]]

    graph = build_inference_graph(config, FakeInferenceClient(config), neo4j_client)
    result = graph.invoke({"paper_id": "paper-1706.03762"}, {"configurable": {"thread_id": "paper-1706.03762"}})

    assert result["paper_title"] == ATTENTION_TITLE
    assert result["retrieved_nodes"] == retrieved_nodes
    assert result["best_idea"]["title"] == "Attention-driven idea"
