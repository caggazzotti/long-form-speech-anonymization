"""
Run OpenAI-style Batch API for paraphrasing prompts.

This script uploads an input JSONL file, creates a batch job, polls status,
and downloads output and/or error JSONL files when complete.

Required environment:
  OPENAI_API_KEY

Optional environment (for Azure OpenAI-compatible endpoints):
  OPENAI_BASE_URL
"""

from __future__ import annotations

import argparse
import os
import time

from openai import OpenAI


TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def write_bytes(path: str, content: bytes) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and monitor a paraphrasing batch job.")
    parser.add_argument("--prompts", required=True, help="Input prompts JSONL path")
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="Batch endpoint (OpenAI: /v1/chat/completions, Azure often /chat/completions)",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window (usually 24h)",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--responses-out", default=None, help="Write output file content here")
    parser.add_argument("--errors-out", default=None, help="Write error file content here")
    parser.add_argument("--batch-id-out", default=None, help="Optional path to save batch ID")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit batch and exit immediately (still prints batch_id)",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    uploaded = client.files.create(file=open(args.prompts, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
    )
    print(f"Created batch: {batch.id}")

    if args.batch_id_out:
        with open(args.batch_id_out, "w", encoding="utf-8") as f:
            f.write(batch.id + "\n")

    if args.no_wait:
        return

    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"Batch {batch.id} status: {batch.status}")
        if batch.status in TERMINAL_STATES:
            break
        time.sleep(args.poll_seconds)

    output_file_id = batch.output_file_id
    error_file_id = batch.error_file_id

    if output_file_id and args.responses_out:
        output_content = client.files.content(output_file_id).content
        write_bytes(args.responses_out, output_content)
        print(f"Wrote responses -> {args.responses_out}")
    elif output_file_id:
        print(f"Output file available: {output_file_id} (set --responses-out to download)")

    if error_file_id and args.errors_out:
        error_content = client.files.content(error_file_id).content
        write_bytes(args.errors_out, error_content)
        print(f"Wrote errors -> {args.errors_out}")
    elif error_file_id:
        print(f"Error file available: {error_file_id} (set --errors-out to download)")

    if batch.status != "completed":
        raise SystemExit(f"Batch ended with status: {batch.status}")


if __name__ == "__main__":
    main()
