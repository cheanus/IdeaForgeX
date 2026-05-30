# src/llm — LLM 调用与 Prompt 管理约定

## 客户端

`ChatClient` 在 `client.py` 中封装 OpenAI 兼容 API，提供三个方法：

| 方法 | 用途 | 返回 |
|---|---|---|
| `chat(messages, response_format?, temperature)` | 通用 chat/completions | `str` |
| `chat_json(messages, temperature)` | JSON mode 调用 | `dict` |
| `embed(texts)` | 批量 embedding | `list[list[float]]` |

所有 LLM 调用通过 `ChatClient`，禁止在业务代码中直接 `import openai`。

## Prompt 管理

`prompts.py` 为 LLM A 定义 2 种 `build_*_messages` 函数：

| 函数 | 用途 | 输出格式 | 调用方 parser |
|---|---|---|---|
| `build_query_generation_messages` | 训练：从论文提炼检索查询 | `{"query_text": "..."}` | `parse_query_text` |
| `build_llm_a_judge_messages` | 训练：判断 + 生成节点/边/更新 | `LLMACandidate` (含 `node_updates`) | `parse_llm_a_candidate` |

## System Prompt 编写规则

- 用中文。角色描述清晰。
- 输出格式明确指定（JSON schema 示例）。
- 包含失败处理指引：`如果无法生成有效 JSON，返回 {"error": "原因"}`。
- `build_llm_a_judge_messages`：嵌入检索结果 + `node_updates` 字段说明 + 更新 vs 新增规则。
- `rel_type` 取值在 system prompt 中直接约束为枚举值，不依赖代码层兜底映射。

## JSON 解析与重试

```python
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
| LLM A 检索查询提炼 | 1.0 | `llm_temperature` |
| LLM A 判断/生成 | 1.0 | `llm_temperature` |

所有值通过 `config.py` 可配置，不在 prompt 里硬编码。

## 文件职责

| 文件 | 职责 |
|---|---|
| `client.py` | `ChatClient` 封装：chat / chat_json / embed |
| `prompts.py` | 2 种 system prompt 模板 + `build_*_messages` 函数 |
| `service.py` | `call_with_retry`：JSON mode 调用 + parser 回调 + 自动重试 |
