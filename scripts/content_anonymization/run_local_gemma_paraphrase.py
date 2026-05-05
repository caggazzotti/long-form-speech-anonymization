"""
Run local Gemma paraphrasing over prompt JSONL and emit response JSONL.

The output format mirrors batch-style responses expected by
paraphrase_responses_to_utterances.py:
{
  "custom_id": "...",
  "response": {
    "status_code": 200,
    "body": {"choices": [{"message": {"content": "..."}}]}
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def extract_messages(prompt_row: dict[str, Any]) -> list[dict[str, str]]:
    body = prompt_row.get("body", {})
    messages = body.get("messages", [])
    extracted: list[dict[str, str]] = []
    for m in messages:
        extracted.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
    return extracted


def run_model(
    model_id: str,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    outputs: list[dict[str, Any]] = []
    do_sample = temperature > 0
    for row in prompts:
        custom_id = row.get("custom_id", "")
        messages = extract_messages(row)
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                )
            new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(
                {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": content}}]},
                    },
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            outputs.append(
                {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 500,
                        "error": {"message": str(exc)},
                    },
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Gemma paraphrasing from prompt JSONL.")
    parser.add_argument("--prompts", required=True, help="Input prompt JSONL")
    parser.add_argument("--output", required=True, help="Output response JSONL")
    parser.add_argument("--model-id", default="google/gemma-3-4b-it", help="HF model ID")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    responses = run_model(
        model_id=args.model_id,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    write_jsonl(args.output, responses)
    print(f"Wrote {len(responses)} responses -> {args.output}")


if __name__ == "__main__":
    main()
