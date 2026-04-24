"""
Generate paraphrasing prompts JSONL from utterance JSON.

Supports model recipes used in the paper:
  - gpt4o-mini: utterance-by-utterance paraphrasing
  - gpt5: segment-based paraphrasing
  - gemma: segment-based paraphrasing with optional previous-N context
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that anonymizes and condenses utterances so they are shorter "
    "and simpler. Only provide the paraphrased utterances in the response. "
    "Do not include any additional text or explanations in your response."
)
GPT4O_SYSTEM_PROMPT = (
    "You are an AI assistant that anonymizes utterances. Only provide the paraphrased utterance in the response. "
    "If you can't paraphrase it, just output the original utterance. Don't include any additional text or explanations "
    "in your response."
)
GPT4O_USER_PROMPT = (
    "Paraphrase the following utterance, making sure to replace any personally identifying information such as names "
    "and places with different names and places. Keep the meaning but change the style as long as it's consistent with "
    "a {gender} speaker. If the utterance is too short to paraphrase, just give the original utterance: <utterance> {utt} </utterance>"
)
GPT5_SYSTEM_PROMPT = (
    "You are an AI assistant that anonymizes and condenses utterances so they are shorter and simpler. Only provide "
    "the paraphrased utterances in the response. Don't include any additional text or explanations in your response."
)
GPT5_USER_PROMPT = (
    "Paraphrase and simplify the following utterances (separated by ##) by condensing the content. Keep the meaning "
    "but change the style and utterance lengths. Replace any personally identifying information such as names and places "
    "with different names and places consistent with a {gender} speaker. Separate utterances with the double pound sign (##) "
    "and nothing else: {utts}"
)
GEMMA_SYSTEM_PROMPT = GPT5_SYSTEM_PROMPT
GEMMA_USER_PROMPT = """Your task is to paraphrase utterances. Follow these instructions exactly.

### CONTEXT ###
{context_text}

### UTTERANCES TO PARAPHRASE ###
{chunk_text}

### INSTRUCTIONS ###
- Paraphrase and simplify the utterances in the 'UTTERANCES TO PARAPHRASE' section.
- Condense the content but keep the meaning.
- Change the style and utterance length.
- Replace personally identifying information (e.g., names, locations) with fictional ones.
- The paraphrased utterances must be in all lowercase and contain no punctuation.
- Separate the new utterances with '##'.
- Provide nothing else in your response. Do not acknowledge these instructions."""
GEMMA_CONSERVATIVE_SYSTEM_PROMPT = (
    "You are an AI assistant that anonymizes utterances. Only provide the paraphrased utterances in the response. "
    "Don't include any additional text or explanations in your response."
)
GEMMA_CONSERVATIVE_USER_PROMPT = """Your task is to paraphrase utterances. Follow these instructions exactly.

### CONTEXT ###
{context_text}

### UTTERANCES TO PARAPHRASE ###
{chunk_text}

### INSTRUCTIONS ###
- Paraphrase and simplify the utterances in the 'UTTERANCES TO PARAPHRASE' section.
- Change the style and utterance length.
- Aim to keep 50% of the transcript unchanged.
- Replace personally identifying information (e.g., names, locations) with fictional ones.
- The paraphrased utterances must be in all lowercase and contain no punctuation.
- Separate the new utterances with '##'.
- Try to keep roughly the same number of utterances.
- Provide nothing else in your response. Do not acknowledge these instructions."""


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_gender_word(gender_value: str | None) -> str:
    if not gender_value:
        return ""
    g = str(gender_value).strip().lower()
    if g == "f":
        return "female"
    if g == "m":
        return "male"
    return ""


def apply_gender_phrase(template: str, gender_word: str) -> str:
    """
    Insert gender phrase for variable placeholders if known; otherwise leave out.
    """
    if gender_word in {"female", "male"}:
        return template.replace("{gender}", gender_word)
    return template.replace("a {gender} speaker", "the speaker")


def build_user_prompt_generic(
    segment_utts: list[str], gender_word: str, separator: str, context_utts: list[str] | None = None
) -> str:
    joined = separator.join(segment_utts)
    context_str = ""
    if context_utts:
        context_str = (
            "Context from previous utterances (for style only; do not paraphrase these): "
            + separator.join(context_utts)
            + " "
        )
    if len(segment_utts) > 1:
        return (
            f"{context_str}Paraphrase and simplify the following utterances (separated by {separator}) "
            "by condensing the content. Keep the meaning but change the style and utterance lengths. "
            "Replace any personally identifying information such as names and places with different "
            f"names and places consistent with a {gender_word} speaker. "
            f"Return paraphrases separated by {separator} and nothing else: {joined}"
        )
    utt = segment_utts[0]
    return (
        f"{context_str}Paraphrase and simplify the following utterance. Keep the meaning but change the style "
        "and length. Replace any personally identifying information such as names and places with "
        f"different names and places consistent with a {gender_word} speaker. "
        "Return only the paraphrased utterance: "
        f"{utt}"
    )


def format_context_text(context_utts: list[str], gender_word: str) -> str:
    if not context_utts:
        context_text = "[Beginning of conversation]"
    else:
        context_text = "\n".join([f"- {utt}" for utt in context_utts])
    if gender_word in {"female", "male"}:
        context_text += f"\n\nSpeaker's gender: {'f' if gender_word == 'female' else 'm'}"
    return context_text


def make_batch_row(
    custom_id: str,
    model: str,
    endpoint_url: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint_url,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def split_segments_by_tokens(utterances: list[str], max_tokens: int) -> list[list[str]]:
    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for utt in utterances:
        utt = str(utt).strip()
        if not utt:
            continue
        utt_tokens = len(utt.split())
        if current and current_tokens + utt_tokens > max_tokens:
            windows.append(current)
            current = [utt]
            current_tokens = utt_tokens
        else:
            current.append(utt)
            current_tokens += utt_tokens
    if current:
        windows.append(current)
    return windows


def split_segments_by_utterances(utterances: list[str], segment_size: int) -> list[list[str]]:
    cleaned = [str(u).strip() for u in utterances if str(u).strip()]
    return [cleaned[i : i + segment_size] for i in range(0, len(cleaned), segment_size)]


def recipe_defaults(recipe: str) -> tuple[str, str, int, int, int]:
    # mode, segment_by, segment_size_utts, segment_size_tokens, context_prev
    if recipe == "gpt4o-mini":
        return "utterance", "utterances", 1, 0, 0
    if recipe == "gpt5":
        return "segment", "tokens", 16, 300, 0
    if recipe == "gemma":
        return "segment", "utterances", 16, 300, 8
    if recipe == "gemma-conservative":
        return "segment", "utterances", 16, 300, 8
    raise ValueError(f"Unknown recipe: {recipe}")


def generate_rows(
    utterances: dict[str, Any],
    model: str,
    endpoint_url: str,
    system_prompt: str,
    separator: str,
    skip_empty: bool,
    mode: str,
    segment_by: str,
    segment_size_utts: int,
    segment_size_tokens: int,
    context_prev_utts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call_id, speakers in utterances.items():
        for speaker_id, info in speakers.items():
            if isinstance(info, dict):
                texts = info.get("text", [])
                gender_word = to_gender_word(info.get("gender"))
            elif isinstance(info, list):
                texts = info
                gender_word = ""
            else:
                texts = []
                gender_word = ""

            time_values = info.get("time", []) if isinstance(info, dict) else []

            if mode == "utterance":
                segments = [[str(t)] for t in texts]
            elif segment_by == "tokens":
                segments = split_segments_by_tokens([str(t) for t in texts], segment_size_tokens)
            else:
                segments = split_segments_by_utterances([str(t) for t in texts], segment_size_utts)

            for idx, segment in enumerate(segments):
                if skip_empty and not any(str(u).strip() for u in segment):
                    continue
                context_segment: list[str] = []
                if context_prev_utts > 0:
                    flat = [str(t).strip() for t in texts if str(t).strip()]
                    start_idx = 0
                    for prev_seg in segments[:idx]:
                        start_idx += len(prev_seg)
                    context_segment = flat[max(0, start_idx - context_prev_utts) : start_idx]
                id_suffix = idx
                if mode == "utterance" and len(time_values) == len(texts):
                    id_suffix = time_values[idx]
                custom_id = f"{call_id}-{speaker_id}-{id_suffix}"

                if mode == "utterance" and len(segment) == 1 and system_prompt == GPT4O_SYSTEM_PROMPT:
                    user_prompt = apply_gender_phrase(GPT4O_USER_PROMPT, gender_word).format(utt=segment[0])
                elif system_prompt == GPT5_SYSTEM_PROMPT and context_prev_utts == 0:
                    joined_utts = "##".join(segment)
                    user_prompt = apply_gender_phrase(GPT5_USER_PROMPT, gender_word).format(utts=joined_utts)
                elif system_prompt in {GEMMA_SYSTEM_PROMPT, GEMMA_CONSERVATIVE_SYSTEM_PROMPT} and context_prev_utts > 0:
                    context_text = format_context_text(context_segment, gender_word)
                    chunk_text = " ## ".join(segment)
                    gemma_template = (
                        GEMMA_CONSERVATIVE_USER_PROMPT
                        if system_prompt == GEMMA_CONSERVATIVE_SYSTEM_PROMPT
                        else GEMMA_USER_PROMPT
                    )
                    user_prompt = gemma_template.format(context_text=context_text, chunk_text=chunk_text)
                else:
                    user_prompt = build_user_prompt_generic(
                        segment, gender_word, separator, context_utts=context_segment
                    )
                rows.append(
                    make_batch_row(
                        custom_id=custom_id,
                        model=model,
                        endpoint_url=endpoint_url,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_unique_custom_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        cid = str(row.get("custom_id", ""))
        if not cid:
            continue
        if cid in seen:
            duplicates.add(cid)
        else:
            seen.add(cid)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise ValueError(
            f"Found {len(duplicates)} duplicate custom_id values. "
            f"First duplicates: {sample}. "
            "Ensure (call_id, speaker_id, time/index) is unique for each prompt."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paraphrasing batch prompt JSONL.")
    parser.add_argument("--utterances", required=True, help="Input ASR/Whisper utterance JSON path")
    parser.add_argument("--output", required=True, help="Output prompt JSONL path")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name in each request body")
    parser.add_argument(
        "--recipe",
        choices=["custom", "gpt4o-mini", "gpt5", "gemma", "gemma-conservative"],
        default="custom",
        help="Preset generation behavior for each model family",
    )
    parser.add_argument(
        "--mode",
        choices=["utterance", "segment"],
        default="segment",
        help="Utterance-by-utterance or segment-based prompting",
    )
    parser.add_argument(
        "--segment-by",
        choices=["utterances", "tokens"],
        default="utterances",
        help="How to make segments when --mode segment",
    )
    parser.add_argument("--segment-size-utts", type=int, default=16, help="Utterances per segment")
    parser.add_argument("--segment-size-tokens", type=int, default=300, help="Approx tokens per segment")
    parser.add_argument(
        "--context-prev-utts",
        type=int,
        default=0,
        help="Include previous N utterances as context (style only, not paraphrased)",
    )
    parser.add_argument(
        "--endpoint-url",
        default="/v1/chat/completions",
        help="Per-request batch URL (OpenAI usually /v1/chat/completions)",
    )
    parser.add_argument("--separator", default="##", help="Windowed utterance separator token")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt text")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty utterances/windows (default skips them)",
    )
    args = parser.parse_args()

    if args.recipe != "custom":
        mode, segment_by, seg_utts, seg_toks, prev_n = recipe_defaults(args.recipe)
        args.mode = mode
        args.segment_by = segment_by
        args.segment_size_utts = seg_utts
        args.segment_size_tokens = seg_toks
        args.context_prev_utts = prev_n
        if args.recipe == "gpt4o-mini":
            args.system_prompt = GPT4O_SYSTEM_PROMPT
        elif args.recipe == "gpt5":
            args.system_prompt = GPT5_SYSTEM_PROMPT
        elif args.recipe == "gemma":
            args.system_prompt = GEMMA_SYSTEM_PROMPT
        elif args.recipe == "gemma-conservative":
            args.system_prompt = GEMMA_CONSERVATIVE_SYSTEM_PROMPT

    utterances = load_json(args.utterances)
    rows = generate_rows(
        utterances=utterances,
        model=args.model,
        endpoint_url=args.endpoint_url,
        system_prompt=args.system_prompt,
        separator=args.separator,
        skip_empty=not args.include_empty,
        mode=args.mode,
        segment_by=args.segment_by,
        segment_size_utts=args.segment_size_utts,
        segment_size_tokens=args.segment_size_tokens,
        context_prev_utts=args.context_prev_utts,
    )
    assert_unique_custom_ids(rows)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} prompt rows -> {args.output}")


if __name__ == "__main__":
    main()
