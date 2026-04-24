"""
Build Whisper-side utterance JSON for the content pipeline (call 1 / non-anonymized text).

Writes one file per dataset, e.g. data/whisper_medium_test_trials_utts.json, shaped as:
  { call_id: { speaker_pin: { "text": [str, ...] } } }.


Usage:
  python scripts/whisper_transcribe.py config.yaml
  python scripts/whisper_transcribe.py config.yaml --system whisper_medium --utterances-per-side 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from match_trials import _resolve_trials_info_dir, difficulty_to_trial_types


def transcribe_placeholder(call_id: str, pin: str, utterances_per_side: int) -> list[str]:
    """
    PLACEHOLDER: replace with real transcription of Fisher audio for this call/side.

    Should return one string per utterance segment (same granularity you use downstream).
    """
    return [
        f"[PLACEHOLDER ASR] call {call_id} speaker {pin} segment {i + 1}"
        for i in range(utterances_per_side)
    ]


def _collect_call_pins_for_dataset(
    trials_info_dir: str, dataset: str, difficulties: list[str]
) -> dict[str, set[str]]:
    """call_id -> set of PIN strings for this dataset across the given difficulties."""
    by_call: dict[str, set[str]] = {}

    for difficulty in difficulties:
        pos_type, neg_type = difficulty_to_trial_types(difficulty)
        pos_path = os.path.join(trials_info_dir, f"{dataset}_{pos_type}_trials_info_final.json")
        neg_path = os.path.join(trials_info_dir, f"{dataset}_{neg_type}_trials_info_final.json")
        if not (os.path.isfile(pos_path) and os.path.isfile(neg_path)):
            continue

        with open(pos_path) as f:
            pos_info = json.load(f)
        for trial in pos_info:
            pin = str(trial["PIN"])
            call1 = str(trial["call 1"][1])
            call2 = str(trial["call 2"][1])
            by_call.setdefault(call1, set()).add(pin)
            by_call.setdefault(call2, set()).add(pin)

        with open(neg_path) as f:
            neg_info = json.load(f)
        for trial in neg_info:
            pin1, call1 = str(trial[0][0]), str(trial[0][2])
            pin2, call2 = str(trial[1][0]), str(trial[1][2])
            by_call.setdefault(call1, set()).add(pin1)
            by_call.setdefault(call2, set()).add(pin2)

    return by_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Build whisper_*_trials_utts.json (ASR placeholder).")
    parser.add_argument("config", help="Path to config.yaml (needs trials_info_dir or speech_attribution_dir)")
    parser.add_argument("--system", default="whisper_medium", help="Prefix for output filename under data/")
    parser.add_argument(
        "--utterances-per-side",
        type=int,
        default=2,
        help="How many placeholder strings to emit per (call, pin)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write a single JSON to this path (ignores datasets loop)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    data_dir = os.path.join(work_dir, "data")
    datasets = cfg.get("datasets", ["test"])
    difficulties = cfg.get("difficulties", ["hard"])

    try:
        trials_info_dir = _resolve_trials_info_dir(cfg)
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(data_dir, exist_ok=True)

    def build_utts_dict(by_call: dict[str, set[str]]) -> dict:
        out: dict = {}
        for call_id in sorted(by_call.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            out[call_id] = {}
            for pin in sorted(by_call[call_id], key=lambda x: int(x) if str(x).isdigit() else str(x)):
                text = transcribe_placeholder(call_id, pin, args.utterances_per_side)
                out[call_id][pin] = {"text": text}
        return out

    if args.output:
        by_call: dict[str, set[str]] = {}
        for dataset in datasets:
            part = _collect_call_pins_for_dataset(trials_info_dir, dataset, difficulties)
            for cid, pins in part.items():
                by_call.setdefault(cid, set()).update(pins)
        if not by_call:
            print(
                f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
                file=sys.stderr,
            )
            sys.exit(1)
        utts = build_utts_dict(by_call)
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(utts, f, indent=2)
        print(f"Wrote {out_path} ({len(utts)} calls)")
        return

    any_written = False
    for dataset in datasets:
        by_call = _collect_call_pins_for_dataset(trials_info_dir, dataset, difficulties)
        if not by_call:
            print(f"Skip dataset {dataset}: no trial-info JSONs for requested difficulties.", file=sys.stderr)
            continue
        utts = build_utts_dict(by_call)
        out_path = os.path.join(data_dir, f"{args.system}_{dataset}_trials_utts.json")
        with open(out_path, "w") as f:
            json.dump(utts, f, indent=2)
        print(f"Wrote {out_path} ({len(utts)} calls)")
        any_written = True

    if not any_written:
        print(
            f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
