"""LLM 调用辅助服务。"""

from __future__ import annotations

import json
from typing import Any, Callable

import openai

from src.llm.client import ChatClient


def call_with_retry(
    client: ChatClient,
    messages: list[dict[str, str]],
    max_retries: int = 3,
    temperature: float = 0.1,
    parser: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            payload = client.chat_json(messages, temperature=temperature)
            return parser(payload) if parser is not None else payload
        except (json.JSONDecodeError, KeyError, ValueError, openai.APIError) as exc:
            last_error = exc
    raise last_error or RuntimeError("LLM 调用失败")
