"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging

from src.agent.training import build_training_graph, run_training
from src.cli.queries import cmd_inspect, cmd_random, cmd_relate, cmd_retrieve
from src.config import load_config
from src.llm.client import ChatClient
from src.neo4j.client import create_client
from src.neo4j.schema import ensure_schema, reset_practice_graph

_logger = logging.getLogger("ideaforgex")


def _json_print(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ideaforgex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap")
    subparsers.add_parser("reset")

    train = subparsers.add_parser("train")
    train.add_argument(
        "paper", help="论文 ID（arXiv ID / AMiner ID）或标题，支持多级降级解析"
    )

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("query", help="查询文本（论文摘要 / 一句话想法 / 关键词）")
    retrieve.add_argument("--top_k", type=int, default=None, help="向量命中数")
    retrieve.add_argument(
        "--expand_hops", type=int, default=None, help="非精化边最大扩展深度"
    )
    retrieve.add_argument(
        "--max_per_node", type=int, default=None, help="每节点扩展上限"
    )
    retrieve.add_argument("--decay", type=float, default=None, help="分数衰减因子")
    retrieve.add_argument("--final_limit", type=int, default=None, help="最终截断数")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("id", help="节点 ID（支持逗号分隔多个）")
    inspect.add_argument(
        "--expand-edges",
        dest="expand_edges",
        action="store_true",
        default=True,
        help="是否展开边目标节点详情",
    )
    inspect.add_argument(
        "--no-expand-edges",
        dest="expand_edges",
        action="store_false",
        help="不展开边目标节点详情",
    )

    subparsers.add_parser("stats")

    random_ = subparsers.add_parser("random")
    random_.add_argument("--count", type=int, default=5, help="返回节点数")
    random_.add_argument(
        "--query", type=str, default=None, help="主题过滤，有则在相关范围内随机"
    )

    relate = subparsers.add_parser("relate")
    relate.add_argument("id_a", help="起始节点 ID")
    relate.add_argument("id_b", help="目标节点 ID")
    relate.add_argument("--max_len", type=int, default=6, help="最长路径跳数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.WARNING),
        format="%(levelname)-7s  %(message)s",
    )

    neo4j_client = create_client(config)
    llm_client = ChatClient(config)

    try:
        if args.command == "bootstrap":
            _logger.info("开始初始化 schema …")
            ensure_schema(neo4j_client)
            _logger.info("已完成 schema 初始化")
            return
        if args.command == "reset":
            _logger.info("开始清空实践库 …")
            reset_practice_graph(neo4j_client)
            _logger.info("已清空实践库")
            return
        if args.command == "train":
            _logger.info("开始训练，论文 ID: %s", args.paper)
            graph = build_training_graph(config, llm_client, neo4j_client)
            result = run_training(graph, args.paper)
            _logger.info("训练完成，已生成训练图谱")
            return
        if args.command == "retrieve":
            _logger.info("开始检索: %s", args.query)
            result = cmd_retrieve(
                config,
                llm_client,
                neo4j_client,
                args.query,
                top_k=args.top_k,
                expand_hops=args.expand_hops,
                max_per_node=args.max_per_node,
                decay=args.decay,
                final_limit=args.final_limit,
            )
            _logger.info("检索完成，返回 %d 条", result["meta"]["total_hits"])
            _json_print(result)
            return
        if args.command == "random":
            _logger.info("随机探索，模式=%s", "主题加权" if args.query else "纯随机")
            result = cmd_random(
                config,
                llm_client,
                neo4j_client,
                count=args.count,
                query_text=args.query,
            )
            _logger.info("随机探索完成，返回 %d 个节点", len(result["nodes"]))
            _json_print(result)
            return
        if args.command == "relate":
            _logger.info("路径查询: %s ↔ %s", args.id_a, args.id_b)
            result = cmd_relate(
                neo4j_client,
                args.id_a,
                args.id_b,
                max_len=args.max_len,
            )
            _logger.info(
                "路径查询完成，%s",
                "已连通" if result["connected"] else "未连通",
            )
            _json_print(result)
            return
        if args.command == "stats":
            _logger.warning("stats 功能待接入")
    finally:
        neo4j_client.close()


if __name__ == "__main__":
    main()
