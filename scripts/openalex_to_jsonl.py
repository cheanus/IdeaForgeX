#!/usr/bin/env python3
"""将 OpenAlex CLI 下载的论文元数据目录转换为 JSONL，供 batch-train --jsonl 使用。

用法:
    python scripts/openalex_to_jsonl.py <input_dir> [-o output.jsonl]
    python scripts/openalex_to_jsonl.py <input_dir> | uv run python main.py batch-train --jsonl -
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def decode_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """将 OpenAlex 的 abstract_inverted_index 还原为纯文本。"""
    if not inverted_index:
        return ""
    pos_words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_words[pos] = word
    return " ".join(pos_words[i] for i in sorted(pos_words))


def extract_work_id(full_id: str) -> str:
    """从 OpenAlex 完整 URL 提取 work ID（如 W3138516171）。"""
    return full_id.rstrip("/").rsplit("/", 1)[-1] if full_id else ""


def convert_file(filepath: Path) -> dict | None:
    """将单个 OpenAlex JSON 文件转为 JSONL 记录，非论文文件返回 None。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict):
        return None

    full_id = data.get("id") or ""
    work_id = extract_work_id(full_id)
    if not work_id:
        return None

    title = data.get("title") or data.get("display_name") or ""
    if not title:
        return None

    abstract = decode_abstract(data.get("abstract_inverted_index"))
    if not abstract:
        return None

    record: dict = {
        "id": f"openalex:{work_id}",
        "title": title,
        "abstract": abstract,
    }
    year = data.get("publication_year")
    if year is not None:
        record["year"] = str(year)
    return record


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python scripts/openalex_to_jsonl.py <input_dir> [-o output.jsonl]", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.is_dir():
        print(f"错误: 目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path: str | None = None
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    jsonl_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix == ".json" and f.name != ".openalex-checkpoint.json"
    )

    if not jsonl_files:
        print(f"警告: {input_dir} 中没有找到 .json 文件", file=sys.stderr)
        return

    out_fp = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout
    converted = 0
    skipped = 0

    try:
        for fp in jsonl_files:
            record = convert_file(fp)
            if record:
                out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                converted += 1
            else:
                skipped += 1
    finally:
        if output_path and out_fp is not sys.stdout:
            out_fp.close()

    print(f"完成: {converted} 篇转换, {skipped} 篇跳过", file=sys.stderr)


if __name__ == "__main__":
    main()
