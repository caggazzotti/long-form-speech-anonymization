# Shared helpers for content anonymization pipeline.
from __future__ import annotations

import json
import os
import re


def normalize_text(s: str) -> str:
    """Normalize text for trial/embedding pipeline (lowercase, strip punctuation)."""
    text = s.replace("\u2014", " ")
    text = text.replace("\u2019", "'")
    text = text.replace("\u2013", "-")
    text = re.sub(r"speaker'?s gender:? [mf]\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"(?:speaker's\s+)?utterances(?:\s+to paraphrase)?", "", text, flags=re.IGNORECASE)
    lower_text = text.lower()
    cleaned = re.sub(r"[^\w\s\-']|[_]", "", lower_text)
    no_extra_spaces = re.sub(" +", " ", cleaned)
    return no_extra_spaces.strip()


def load_utterances(path: str) -> dict:
    """
    Load utterance JSON.
    Expected: {call_id: {speaker_id: {'text': [str, ...]}}} or {call_id: {speaker_id: [str, ...]}}.
    Returns dict with structure {call_id: {speaker_id: {'text': [...]}}}.
    """
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for call_id, speakers in data.items():
        cid = str(call_id)
        out[cid] = {}
        for spk_id, val in speakers.items():
            sid = str(spk_id)
            if isinstance(val, dict) and "text" in val:
                out[cid][sid] = val
            elif isinstance(val, list):
                out[cid][sid] = {"text": val}
            else:
                out[cid][sid] = {"text": val.get("text", []) if isinstance(val, dict) else []}
    return out


def get_speaker_lines(utts: dict, call_id: str, speaker_id: str) -> list[str] | None:
    """Return list of utterance strings for (call_id, speaker_id); IDs as strings."""
    cid, sid = str(call_id), str(speaker_id)
    if cid not in utts or sid not in utts[cid]:
        return None
    val = utts[cid][sid]
    if isinstance(val, dict):
        lines = val.get("text", [])
    else:
        lines = val if isinstance(val, list) else []
    return lines if lines else None
