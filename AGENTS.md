# IdeaForgeX

LLM 论文创新点生成系统。设计文档见 `doc/`：

| 文件 | 内容 |
|---|---|
| `doc/design.md` | 完整设计：节点、边、遍历、训练/推理流程 |
| `doc/architecture.md` | 技术架构：模块职责、LangGraph 工作流、检索算法伪代码 |
| `doc/data-model.md` | Neo4j 图模型：约束、索引、可执行 Cypher |

## 全局编码约定

### 命名

| 域 | 约定 |
|---|---|
| Neo4j 标签 | 英文 UpperCamelCase（`Inspiration`, `Question`） |
| Neo4j 关系 | 英文 UPPER_SNAKE（`INSP_COMBINES`, `INSP_REFINES`） |
| Python 类 | 英文 UpperCamelCase |
| Python 函数/变量 | 英文 snake_case |
| 文档/注释 | 中文 |
| 用户可见 log | 中文 |

### 数据流

LLM 返回 JSON → `models.py` 的 Pydantic 验证 → 写入 Neo4j。验证失败触发 RetryPolicy。

### 错误处理

- LLM JSON 解析失败：重试，超过 `max_retries` 记录失败库。
- Neo4j 连接失败：抛异常退出，不静默。
- AMiner API 失败：重试 3 次，仍失败跳过该论文。
- arXiv 全文获取失败：降级为仅用摘要。
- 向量索引不存在：`schema.py` 幂等创建。

### 测试

```bash
uv format
pyright
uv run pytest -v
```

### 配置

所有参数来自 `src/config.py` 的 `Config`（`pydantic-settings`，读取一个统一的 yaml 配置文件）。不要在代码中硬编码值。

### 日志与调试

日志统一使用 `logging.getLogger("ideaforgex")`，在 `src/main.py` 入口处通过 `logging.basicConfig` 初始化。

| 配置项 | 说明 |
|---|---|
| `config.yaml` → `log_level` | 全局日志级别，默认 `WARNING` |
| 环境变量 `LOG_LEVEL` | 运行时覆盖，优先级高于 yaml |

```bash
# 开发调试：显示详细日志
LOG_LEVEL=DEBUG uv run python main.py train 1706.03762

# 仅看关键信息
LOG_LEVEL=INFO uv run python main.py bootstrap

# 生产默认（无状态消息）
uv run python main.py train 1706.03762
```

级别对照：`DEBUG`（所有细节）→ `INFO`（状态消息）→ `WARNING`（异常降级等）→ `ERROR`（仅致命错误）。

注意：`print()` 仅用于 train/infer 的产品输出（stdout），状态消息和异常通过 logger 输出到 stderr。

### 依赖管理

用 `uv`，不用 pip。新增依赖：`uv add <pkg>`，确保 `pyproject.toml` 同步。

## 编码规范

- 良好的函数/类以及模块划分，单一职责原则。
- 完善又精炼的 docstring 和注释，尤其是复杂逻辑。

## 文档维护

各核心子目录有独立 `AGENTS.md`：

- `src/agent/AGENTS.md` — LangGraph StateGraph 编写规范
- `src/neo4j/AGENTS.md` — Neo4j 操作与事务约定
- `src/llm/AGENTS.md` — LLM 调用与 prompt 管理
- `src/paper/AGENTS.md` — AMiner + arXiv 论文 API 约定

`docs/` 只包含设计与架构文档，不包含实现细节。实现细节在各子模块的 `AGENTS.md` 中定义。

全局编码规范在本文件中定义，子模块 `AGENTS.md` 可以根据需要补充特定约定，但必须遵守全局规范。

每次提交前检查相关 `AGENTS.md` 和 `docs/` 是否需要更新，确保文档与代码同步。
