"""FastAPI 只读服务器，对外暴露知识图谱查询接口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cli.queries import cmd_inspect, cmd_random, cmd_relate, cmd_retrieve, cmd_stats
from src.config import Config
from src.llm.client import ChatClient
from src.neo4j.client import Neo4jClient, create_client

_logger = logging.getLogger("ideaforgex")


class RetrieveRequest(BaseModel):
    query: str = Field(description="查询文本（论文摘要 / 一句话想法 / 关键词）")
    top_k: int | None = Field(default=None, description="向量命中数")
    expand_hops: int | None = Field(default=None, description="非精化边最大扩展深度")
    max_per_node: int | None = Field(default=None, description="每节点扩展上限")
    decay: float | None = Field(default=None, description="分数衰减因子")
    final_limit: int | None = Field(default=None, description="最终截断数")


def create_app(config: Config) -> FastAPI:
    """构建 FastAPI 应用，通过 lifespan 管理 Neo4j/Chat 客户端生命周期。"""

    neo4j_client: Neo4jClient | None = None
    llm_client: ChatClient | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal neo4j_client, llm_client
        _logger.info("启动服务器，连接 Neo4j …")
        neo4j_client = create_client(config)
        llm_client = ChatClient(config)
        _logger.info("服务器已就绪，端口 %d", config.server_port)
        yield
        _logger.info("关闭服务器 …")
        if neo4j_client:
            neo4j_client.close()

    app = FastAPI(
        title="IdeaForgeX 知识图谱只读服务",
        version="0.1.0",
        lifespan=lifespan,
        root_path=config.server_root_path,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "IdeaForgeX"}

    @app.post("/retrieve")
    async def api_retrieve(req: RetrieveRequest) -> dict[str, Any]:
        if neo4j_client is None or llm_client is None:
            return {"error": "服务未就绪"}
        return cmd_retrieve(
            config,
            llm_client,
            neo4j_client,
            req.query,
            top_k=req.top_k,
            expand_hops=req.expand_hops,
            max_per_node=req.max_per_node,
            decay=req.decay,
            final_limit=req.final_limit,
        )

    @app.get("/inspect/{node_ids}")
    async def api_inspect(
        node_ids: str,
        expand_edges: bool = Query(default=True, description="是否展开边目标节点详情"),
    ) -> list[dict[str, Any]]:
        if neo4j_client is None:
            return [{"error": "服务未就绪"}]
        return cmd_inspect(neo4j_client, node_ids, expand_edges=expand_edges)

    @app.get("/stats")
    async def api_stats() -> dict[str, Any]:
        if neo4j_client is None:
            return {"error": "服务未就绪"}
        return cmd_stats(neo4j_client)

    @app.get("/random")
    async def api_random(
        count: int = Query(default=5, description="返回节点数"),
        query: str | None = Query(default=None, description="主题过滤"),
    ) -> dict[str, Any]:
        if neo4j_client is None or llm_client is None:
            return {"error": "服务未就绪"}
        return cmd_random(
            config, llm_client, neo4j_client, count=count, query_text=query
        )

    @app.get("/relate/{from_id}/{to_id}")
    async def api_relate(
        from_id: str,
        to_id: str,
        max_len: int = Query(default=6, description="最长路径跳数"),
    ) -> dict[str, Any]:
        if neo4j_client is None:
            return {"error": "服务未就绪"}
        return cmd_relate(neo4j_client, from_id, to_id, max_len=max_len)

    return app
