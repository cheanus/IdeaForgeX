"""命令行入口。"""

from __future__ import annotations

import argparse

from src.agent.inference import build_inference_graph, run_inference
from src.agent.training import build_training_graph, run_training
from src.config import load_config
from src.llm.client import ChatClient
from src.neo4j.client import create_client
from src.neo4j.schema import ensure_schema, reset_practice_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ideaforgex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap")
    subparsers.add_parser("reset")

    train = subparsers.add_parser("train")
    train.add_argument(
        "paper", help="论文 ID（arXiv ID / AMiner ID）或标题，支持多级降级解析"
    )

    infer = subparsers.add_parser("infer")
    infer.add_argument(
        "paper", help="论文 ID（arXiv ID / AMiner ID）或标题，支持多级降级解析"
    )

    subparsers.add_parser("stats")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()
    neo4j_client = create_client(config)
    llm_client = ChatClient(config)

    try:
        if args.command == "bootstrap":
            ensure_schema(neo4j_client)
            print("已完成 schema 初始化")
            return
        if args.command == "reset":
            reset_practice_graph(neo4j_client)
            print("已清空实践库")
            return
        if args.command == "train":
            graph = build_training_graph(config, llm_client, neo4j_client)
            result = run_training(graph, args.paper)
            print(result)
            return
        if args.command == "infer":
            graph = build_inference_graph(config, llm_client, neo4j_client)
            result = run_inference(graph, args.paper)
            print(result)
            return
        if args.command == "stats":
            print("stats 功能待接入")
    finally:
        neo4j_client.close()


if __name__ == "__main__":
    main()
