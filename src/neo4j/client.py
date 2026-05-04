"""Neo4j 连接管理。"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import GraphDatabase

from src.config import Config


@dataclass
class Neo4jClient:
    config: Config

    def __post_init__(self) -> None:
        auth = None if not self.config.neo4j_user or not self.config.neo4j_password else (self.config.neo4j_user, self.config.neo4j_password)
        self.driver = GraphDatabase.driver(self.config.neo4j_uri, auth=auth)

    def close(self) -> None:
        self.driver.close()


def create_client(config: Config) -> Neo4jClient:
    return Neo4jClient(config)
