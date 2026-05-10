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

`prompts.py` 为 LLM A 定义 3 种 `build_*_messages` 函数：

| 函数 | 用途 | 输出格式 | 调用方 parser |
|---|---|---|---|
| `build_query_generation_messages` | 训练：从论文提炼检索查询 | `{"query_text": "..."}` | `parse_query_text` |
| `build_llm_a_judge_messages` | 训练：判断 + 生成节点/边/更新 | `LLMACandidate` (含 `node_updates`) | `parse_llm_a_candidate` |
| `build_inference_messages` | 推理：生成创新点候选 | `LLMACandidate` | `parse_llm_a_candidate` |

```python
# 训练：检索查询提炼
def build_query_generation_messages(paper_text: str) -> list[dict]:
    ...

# 训练：判断 + 生成 + 更新
def build_llm_a_judge_messages(
    paper_text: str, practice_summary: str, retrieved_nodes: list[dict]
) -> list[dict]:
    ...

# 推理：创新点生成
def build_inference_messages(
    paper_text: str, retrieved_nodes: list[dict]
) -> list[dict]:
    ...
```

## System Prompt 编写规则

- 用中文。角色描述清晰。
- 输出格式明确指定（JSON schema 示例）。
- 包含失败处理指引：
  > 如果无法生成有效 JSON，返回 `{"error": "原因"}`。
- `build_llm_a_judge_messages`：嵌入检索结果 + `node_updates` 字段说明 + 更新 vs 新增规则
- `build_inference_messages`：嵌入检索结果 + 范式库，引导 LLM 基于已有知识生成新连接
- `rel_type` 取值在 system prompt 中直接约束为枚举值，不依赖代码层兜底映射

## JSON 解析与重试

```python
from src.llm.client import ChatClient

def call_with_retry(client: ChatClient, messages: list[dict], max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            return client.chat_json(messages)
        except (json.JSONDecodeError, KeyError, ValueError, openai.APIError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON 解析失败 (attempt {attempt+1}): {e}")
    raise RuntimeError("unreachable")
```

RetryPolicy 在 LangGraph 层也配置了。`call_with_retry` 是函数级兜底。

## 温度

| 场景 | temperature | 配置键 |
|---|---|---|
| LLM A 检索查询提炼 | 0.1（确定性） | `temperature_llm_a_judge` |
| LLM A 判断/生成 | 0.1（确定性） | `temperature_llm_a_judge` |

所有值通过 `config.py` 可配置，不在 prompt 里硬编码。

## 文件职责

| 文件 | 职责 |
|---|---|
| `client.py` | `ChatClient` 封装：chat / chat_json / embed |
| `prompts.py` | 3 种 system prompt 模板 + `build_*_messages` 函数 |
| `service.py` | `call_with_retry`：JSON mode 调用 + parser 回调 + 自动重试 |
