"""论文库 — 基于 SQLite 的已训练论文去重索引。"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

_logger = logging.getLogger("ideaforgex")


class PaperLibrary:
    """已训练论文的轻量去重库。

    WAL 模式支持并发读，写操作串行化队列等待。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title    TEXT NOT NULL,
                    year     TEXT NOT NULL DEFAULT '',
                    trained_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )

    def is_trained(self, paper_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)
            ).fetchone()
            return row is not None

    def try_reserve(self, paper_id: str, title: str, year: str = "") -> bool:
        """原子操作：尝试预留论文记录。返回 True 表示新记录已写入，可继续训练。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO papers (paper_id, title, year) VALUES (?, ?, ?)",
                (paper_id, title, year or ""),
            )
            reserved = cursor.rowcount > 0
        if reserved:
            _logger.info("标记论文已训练: %s", paper_id)
        return reserved
