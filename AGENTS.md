# IdeaForgeX

AI 论文创新点知识图谱系统。训练时通过 LLM 从论文中提炼方法灵感和研究问题，写入 Neo4j 知识图谱；推理时外部 agent 通过 CLI 检索图数据，自行编排创新点生成。

设计文档见 `docs/superpowers/specs/`：

| 文件 | 内容 |
|---|---|
| `2025-05-30-system-design.md` | 系统设计：数据模型、检索遍历、训练/推理流程、技术架构 |
| `2025-05-30-cli-spec.md` | CLI 命令规范：`retrieve` / `inspect` 输入输出格式 |
| `../use_cli.md` | CLI 使用指南：面向用户的命令参考 |

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

- LLM JSON 解析失败：重试，超过 `max_retries` 终止。
- Neo4j 连接失败：抛异常退出，不静默。
- OpenAlex API 失败：重试 3 次（指数退避），仍失败跳过该论文。
- arXiv 全文获取失败：降级为仅用摘要。
- 向量索引不存在：`schema.py` 幂等创建。

### 测试

```bash
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

级别对照：`DEBUG`（所有细节）→ `INFO`（状态消息）→ `WARNING`（异常降级等）→ `ERROR`（仅致命错误）。

`print()` 仅用于 train/retrieve/inspect 的产品输出（stdout），状态消息和异常通过 logger 输出到 stderr。

### 依赖管理

用 `uv`，不用 pip。新增依赖：`uv add <pkg>`，确保 `pyproject.toml` 同步。

## 行为规范

- 良好的函数/类以及模块划分，单一职责原则。
- 完善又精炼的 docstring 和注释，尤其是复杂逻辑。
- 允许自主发起 git 提交，但须合理拆分（每个提交一个逻辑变更），提交信息用中文并简洁准确。涉及机密信息（API key、密码等）的文件不提交。
- 目前为开发版本，旧代码可直接删除，不必兼容。
- 遇到奇怪现象、任务难以解决、决策不确定、两难困境时，直接向用户报告，而非盲目尝试。

## 文档维护

各核心子目录有独立 `CONVENTIONS.md`：

- `src/agent/CONVENTIONS.md` — LangGraph StateGraph 编写规范
- `src/neo4j/CONVENTIONS.md` — Neo4j 操作与事务约定
- `src/llm/CONVENTIONS.md` — LLM 调用与 prompt 管理
- `src/paper/CONVENTIONS.md` — OpenAlex + arXiv 论文 API 约定
- `src/cli/CONVENTIONS.md` — CLI 查询命令约定

`docs/superpowers/specs/` 包含设计与架构文档，不包含实现细节。实现细节在各子模块的 `CONVENTIONS.md` 中定义。

全局编码规范在本文件中定义，子模块 `CONVENTIONS.md` 可以根据需要补充特定约定，但必须遵守全局规范。

每次提交前检查相关 `CONVENTIONS.md`、`README.md`（包括多语种版本）和 `docs/` 是否需要更新，确保文档与代码同步。

## 多 agent 开发协作

- 主 agent: 规划、方案制定、文档编写、功能开发
- 子 agent: 功能测试、bug 修复、代码探索
