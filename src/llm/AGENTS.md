# src/llm — LLM 调用与 Prompt 管理约定

## 客户端封装

```python
from openai import OpenAI

class ChatClient:
    def __init__(self, config: Config):
        self.chat_client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
        self.embedding_client = OpenAI(base_url=config.embedding_base_url, api_key=config.embedding_api_key)
        self.llm_model_name = config.llm_model_name
        self.embedding_model_name = config.embedding_model_name

    def chat(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float = 0.7
    ) -> str:
        """调用 chat/completions，返回 content 字符串"""
        kwargs = {"model": self.llm_model_name, "messages": messages, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.chat_client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def chat_json(self, messages: list[dict], temperature: float = 0.3) -> dict:
        """JSON mode 调用，返回解析后的 dict"""
        raw = self.chat(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature
        )
        return json.loads(raw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding"""
        resp = self.embedding_client.embeddings.create(
            model=self.embedding_model_name,
            input=texts
        )
        return [d.embedding for d in resp.data]
```

所有 LLM 调用通过 `ChatClient`，禁止在业务代码中直接 `import openai`。

## Prompt 管理

`prompts.py` 仅为 LLM A 定义 `build_*_messages` 函数：

```python
def build_llm_a_judge_messages(paper_text: str, paradigm_list: str, practice_summary: str) -> list[dict]:
    system = """你是..."""
    user = f"""论文：{paper_text}\n范式库：{paradigm_list}\n实践库概要：{practice_summary}\n..."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
```

函数命名：`build_{role}_{task}_messages`。当前仅保留 LLM A 的判断函数。

## System Prompt 编写规则

- 用中文。角色描述清晰。
- 输出格式明确指定（JSON schema 示例）。
- 包含失败处理指引：
  > 如果无法生成有效 JSON，返回 `{"error": "原因"}`。
- 对 LLM A：在 prompt 中嵌入问题组合边的判断标准（交集定义新缺口）和权重评分指引。

## JSON 解析与重试

```python
from src.llm.client import ChatClient

def call_with_retry(client: ChatClient, messages: list[dict], max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            return client.chat_json(messages)
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON 解析失败 (attempt {attempt+1}): {e}")
    raise RuntimeError("unreachable")
```

RetryPolicy 在 LangGraph 层也配置了。`call_with_retry` 是函数级兜底。

## 温度

| 场景 | temperature |
|---|---|
| LLM A 判断 | 0.1（确定性） |
| LLM A 生成节点/边 | 0.5（需创造性） |

所有值通过 `config.py` 可配置，不在 prompt 里硬编码。

## 文件职责

| 文件 | 职责 |
|---|---|
| `client.py` | `ChatClient` 封装：chat / chat_json / embed |
| `prompts.py` | 所有 system prompt 模板 + `build_*_messages` 函数 |
