# Server 编写规范

## 架构

FastAPI 应用通过 `create_app(config)` 工厂函数构建，避免模块级全局状态。

`lifespan` 上下文管理 Neo4jClient 和 ChatClient 的创建与销毁，所有端点通过闭包访问。

## 端点设计

| 端点 | 对应 CLI 命令 | 方法 |
|---|---|---|
| `/retrieve` | `retrieve` | POST（查询文本长，不适合 query string） |
| `/inspect/{node_ids}` | `inspect` | GET |
| `/stats` | `stats` | GET |
| `/random` | `random` | GET |
| `/relate/{from_id}/{to_id}` | `relate` | GET |

所有端点直接调用 `src/cli/queries.py` 中的 `cmd_*` 函数，零逻辑重复。参数名、类型、默认值与 CLI 严格一致。

## 错误处理

- 节点不存在等业务异常：cmd_* 函数已返回包含 `error` 字段的 dict，直接透传
- 服务未就绪（客户端未初始化）：返回 `{"error": "服务未就绪"}`
- Neo4j 连接失败：lifespan 阶段抛异常，FastAPI 自动返回 500

## 测试

使用 `fastapi.testclient.TestClient` + `unittest.mock` 模拟 Neo4jClient/ChatClient，无需真实数据库。

## 配置

server 相关配置平铺在 `Config` 中：
- `server_host`: 绑定地址，默认 `0.0.0.0`
- `server_port`: 端口，默认 `2048`
- `server_cors_origins`: CORS 允许来源列表，默认 `["*"]`

通过 `src/config.py` 的 pydantic-settings 统一管理，环境变量前缀 `IDEAFORGEX_`。