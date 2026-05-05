"""
Convert paraphrasing API responses (e.g. OpenAI/Azure batch JSONL) into utterance JSON.

Expects response format: each line is a JSON object with at least:
  - custom_id: "callId-speakerId-timeOrIndex"
  - response.body.choices[0].message.content (or equivalent)

Output: {call_id: {speaker_id: {'text': [str, ...]}}} saved as JSON.

Usage:
  python scripts/content_anonymization/paraphrase_responses_to_utterances.py --responses data/paraphrased_gpt4omini_responses.jsonl --output data/paraphrased_gpt4omini_test_trials_utts.json [--normalize]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict


def clean_utterance(text: str) -> str:
    """Remove prompt/template artifacts seen in some model outputs."""
    if "Paraphrased utterance:" in text:
        text = re.sub(r".*Paraphrased utterance:\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"Original utterance:\s*<utterance>(.*?)</utterance>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"Original utterance:\s*", "", text)
    text = re.sub(r"speaker'?s gender:? [mf]\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:speaker's\s+)?utterances(?:\s+to paraphrase)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def normalize_text(s: str) -> str:
    """Normalize text for trial/embedding pipeline (lowercase, strip punctuation)."""
    s = s.replace("\u2014", " ").replace("\u2019", "'").replace("\u2013", "-")
    s = s.lower()
    s = re.sub(r"[^\w\s\-']|[_]", "", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


def load_responses_jsonl(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_content(obj: dict) -> str:
    """Extract message content from OpenAI/Azure-style response."""
    try:
        return obj["response"]["body"]["choices"][0]["message"]["content"]
    except KeyError:
        pass
    try:
        return obj["choices"][0]["message"]["content"]
    except KeyError:
        pass
    return obj.get("content", "")


def split_paraphrase_content(text: str, separator: str | None) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if separator:
        parts = [p.strip() for p in text.split(separator)]
        parts = [p for p in parts if p]
        if parts:
            return parts
    if "\n" in text:
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="JSONL file of API responses (custom_id + response)")
    ap.add_argument("--output", required=True, help="Output utterance JSON path")
    ap.add_argument("--normalize", action="store_true", help="Normalize text (lowercase, strip punctuation)")
    ap.add_argument(
        "--segment-separator",
        default=None,
        help="Split model outputs by this separator (e.g., ##) for segment-based paraphrasing",
    )
    args = ap.parse_args()

    data = load_responses_jsonl(args.responses)
    combined = []
    for r in data:
        cid = r.get("custom_id", "")
        content = extract_content(r)
        cleaned = clean_utterance(content) if content else ""
        segments = split_paraphrase_content(cleaned, args.segment_separator)
        if args.normalize:
            segments = [normalize_text(s) for s in segments]
        combined.append((cid, segments))

    combined.sort(key=lambda x: (int(x[0].split("-")[0]), int(x[0].split("-")[1]), float(x[0].split("-")[2]) if x[0].count("-") >= 2 else 0))

    utts = defaultdict(lambda: defaultdict(lambda: {"text": []}))
    for cid, segment_texts in combined:
        parts = cid.split("-", 2)
        call_id, spk_id = parts[0], parts[1]
        if segment_texts:
            utts[call_id][spk_id]["text"].extend(segment_texts)
        else:
            utts[call_id][spk_id]["text"].append("")

    utts = json.loads(json.dumps(utts))
    with open(args.output, "w") as f:
        json.dump(utts, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
