from types import SimpleNamespace

from src.config import Config
from src.llm.client import ChatClient


class DummyCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class DummyEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])])


class DummyOpenAI:
    def __init__(self, *, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=DummyCompletions())
        self.embeddings = DummyEmbeddings()


def test_chat_client_uses_distinct_clients_for_llm_and_embedding(monkeypatch):
    created_clients = []

    def factory(*, base_url: str, api_key: str):
        client = DummyOpenAI(base_url=base_url, api_key=api_key)
        created_clients.append(client)
        return client

    monkeypatch.setattr("src.llm.client.OpenAI", factory)

    config = Config(
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-key",
        llm_model_name="llm-model",
        embedding_base_url="https://embed.example/v1",
        embedding_api_key="embed-key",
        embedding_model_name="embed-model",
    )
    client = ChatClient(config)

    assert len(created_clients) == 2
    assert created_clients[0].base_url == "https://llm.example/v1"
    assert created_clients[0].api_key == "llm-key"
    assert created_clients[1].base_url == "https://embed.example/v1"
    assert created_clients[1].api_key == "embed-key"

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert client.embed(["abc"]) == [[1.0, 2.0, 3.0]]

    assert created_clients[0].chat.completions.calls[0]["model"] == "llm-model"
    assert created_clients[1].embeddings.calls[0]["model"] == "embed-model"


def test_chat_client_exposes_new_model_field_names(monkeypatch):
    monkeypatch.setattr("src.llm.client.OpenAI", DummyOpenAI)

    config = Config(
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-key",
        llm_model_name="llm-model",
        embedding_base_url="https://embed.example/v1",
        embedding_api_key="embed-key",
        embedding_model_name="embed-model",
    )
    client = ChatClient(config)

    assert client.llm_model_name == "llm-model"
    assert client.embedding_model_name == "embed-model"
