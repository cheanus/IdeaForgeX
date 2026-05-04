"""OpenAI 兼容 LLM 客户端。"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.config import Config


class ChatClient:
    def __init__(self, config: Config):
        self.config = config
        self.chat_client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
        self.embedding_client = OpenAI(base_url=config.embedding_base_url, api_key=config.embedding_api_key)
        self.llm_model_name = config.llm_model_name
        self.embedding_model_name = config.embedding_model_name

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.7,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.llm_model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.chat_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict[str, Any]:
        raw = self.chat(messages, response_format={"type": "json_object"}, temperature=temperature)
        return json.loads(raw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.embedding_client.embeddings.create(model=self.embedding_model_name, input=texts)
        return [list(item.embedding) for item in response.data]
