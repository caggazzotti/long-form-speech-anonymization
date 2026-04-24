"""
Create a retry JSONL containing only prompt rows whose responses failed.

Usage:
  python scripts/retry_failed_batch_rows.py \
    --prompts data/paraphrase_gpt4omini_prompts.jsonl \
    --responses data/paraphrased_gpt4omini_errors.jsonl \
    --output data/paraphrase_gpt4omini_failed_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract failed rows for batch retry.")
    parser.add_argument("--prompts", required=True, help="Master prompt JSONL (with custom_id)")
    parser.add_argument(
        "--responses",
        nargs="+",
        required=True,
        help="One or more response/error JSONL files to scan",
    )
    parser.add_argument("--output", required=True, help="Output JSONL with failed prompt rows")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also retry custom_ids from responses missing status_code",
    )
    args = parser.parse_args()

    failed_ids: set[str] = set()
    for response_file in args.responses:
        rows = read_jsonl(response_file)
        for row in rows:
            custom_id = str(row.get("custom_id", ""))
            if not custom_id:
                continue
            status_code = row.get("response", {}).get("status_code")
            if status_code is None:
                if args.include_missing:
                    failed_ids.add(custom_id)
            elif int(status_code) != 200:
                failed_ids.add(custom_id)

    prompts = read_jsonl(args.prompts)
    retry_rows = [row for row in prompts if str(row.get("custom_id", "")) in failed_ids]
    write_jsonl(args.output, retry_rows)
    print(f"Collected {len(failed_ids)} failed IDs; wrote {len(retry_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
