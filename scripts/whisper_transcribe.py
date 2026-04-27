"""
Build Whisper-side transcription outputs for Fisher calls.

Default output is the utterance JSON used by the content / matched-trial pipeline:
  { call_id: { speaker_pin: { "text": [str, ...], "gender": "m"|"f" } } }

This script can also emit XTTS-style `filename.wav|transcript` rows for the voice
anonymization pipeline.

Usage:
  python scripts/whisper_transcribe.py config.yaml
  python scripts/whisper_transcribe.py config.yaml --system whisper_medium --utterances-per-side 3
  python scripts/whisper_transcribe.py config.yaml --output-format xtts_manifest --output data/voiceanon_inputs.txt
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
from paraphrase_responses_to_utterances import normalize_text as normalize_utterance_line


def transcribe_placeholder(call_id: str, pin: str, gender: str, utterances_per_side: int) -> list[str]:
    """
    PLACEHOLDER: replace with real transcription of Fisher audio for this call/side.

    Should return one string per utterance segment (same granularity you use downstream).
    `gender` is \"m\", \"f\", or \"\" (from trial metadata).
    """
    _ = gender  # available for real ASR / prompt conditioning
    return [
        f"[PLACEHOLDER ASR] call {call_id} speaker {pin} segment {i + 1}"
        for i in range(utterances_per_side)
    ]


def _normalize_gender(raw: object) -> str:
    if raw is None:
        return ""
    g = str(raw).strip().lower()
    if g in ("m", "male"):
        return "m"
    if g in ("f", "female"):
        return "f"
    return ""


def _merge_pair_gender(dst: dict[tuple[str, str], str], call_id: str, pin: str, gender: str) -> None:
    key = (call_id, pin)
    if not gender:
        if key not in dst:
            dst[key] = ""
        return
    if key not in dst or not dst[key]:
        dst[key] = gender
        return
    if dst[key] != gender:
        print(f"Warning: conflicting gender for call {call_id} pin {pin}: {dst[key]!r} vs {gender!r}", file=sys.stderr)


def _collect_pair_genders_for_dataset(
    trials_info_dir: str, dataset: str, difficulties: list[str]
) -> dict[tuple[str, str], str]:
    """(call_id, pin) -> \"m\"|\"f\"|\"\" from pos/neg trial-info JSONs for this dataset."""
    out: dict[tuple[str, str], str] = {}

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
            row1, row2 = trial["call 1"], trial["call 2"]
            g1, call1 = _normalize_gender(row1[0]), str(row1[1])
            g2, call2 = _normalize_gender(row2[0]), str(row2[1])
            _merge_pair_gender(out, call1, pin, g1)
            _merge_pair_gender(out, call2, pin, g2)

        with open(neg_path) as f:
            neg_info = json.load(f)
        for trial in neg_info:
            pin1 = str(trial[0][0])
            g1 = _normalize_gender(trial[0][1])
            call1 = str(trial[0][2])
            pin2 = str(trial[1][0])
            g2 = _normalize_gender(trial[1][1])
            call2 = str(trial[1][2])
            _merge_pair_gender(out, call1, pin1, g1)
            _merge_pair_gender(out, call2, pin2, g2)

    return out


def _sort_call_id(cid: str) -> tuple[int, str]:
    return (int(cid), cid) if str(cid).isdigit() else (10**18, cid)


def _sort_pin(pid: str) -> tuple[int, str]:
    return (int(pid), pid) if str(pid).isdigit() else (10**18, pid)


def build_utts_dict(
    pair_gender: dict[tuple[str, str], str], utterances_per_side: int, *, normalize: bool
) -> dict:
    """Pipeline utterance JSON: call -> pin -> {text, gender?}."""
    by_call: dict[str, dict] = {}
    for (call_id, pin) in sorted(pair_gender.keys(), key=lambda t: (_sort_call_id(t[0]), _sort_pin(t[1]))):
        gender = pair_gender[(call_id, pin)]
        raw = transcribe_placeholder(call_id, pin, gender, utterances_per_side)
        text = [normalize_utterance_line(s) for s in raw] if normalize else raw
        speaker: dict = {"text": text}
        if gender:
            speaker["gender"] = gender
        by_call.setdefault(call_id, {})[pin] = speaker
    return by_call


def build_xtts_manifest_lines(
    pair_gender: dict[tuple[str, str], str], utterances_per_side: int, *, normalize: bool
) -> list[str]:
    """
    XTTS manifest rows:
      fe_{call_id}_{speaker_id}_{segment_index}.wav|transcript

    This keeps a deterministic filename convention for downstream anonymization.
    """
    lines: list[str] = []
    for (call_id, pin) in sorted(pair_gender.keys(), key=lambda t: (_sort_call_id(t[0]), _sort_pin(t[1]))):
        gender = pair_gender[(call_id, pin)]
        raw = transcribe_placeholder(call_id, pin, gender, utterances_per_side)
        text_items = [normalize_utterance_line(s) for s in raw] if normalize else raw
        for index, transcript in enumerate(text_items, start=1):
            filename = f"fe_{call_id}_{pin}_{index}.wav"
            lines.append(f"{filename}|{transcript}")
    return lines


def _merge_pair_genders_across_datasets(
    trials_info_dir: str, datasets: list[str], difficulties: list[str]
) -> dict[tuple[str, str], str]:
    merged: dict[tuple[str, str], str] = {}
    for dataset in datasets:
        part = _collect_pair_genders_for_dataset(trials_info_dir, dataset, difficulties)
        for key, gender in part.items():
            if key not in merged:
                merged[key] = gender
            elif gender and not merged[key]:
                merged[key] = gender
            elif gender and merged[key] and merged[key] != gender:
                print(
                    f"Warning: conflicting gender for {key} across datasets: {merged[key]!r} vs {gender!r}",
                    file=sys.stderr,
                )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Whisper transcription outputs (ASR placeholder).")
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
        help="If set, write a single output file to this path (ignores dataset-based default naming).",
    )
    parser.add_argument(
        "--output-format",
        choices=("utterance_json", "xtts_manifest"),
        default="utterance_json",
        help="Output format: content-pipeline JSON or XTTS `filename|transcript` manifest.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same per-line normalization as paraphrase_responses_to_utterances.py (default: on)",
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

    if args.output_format == "xtts_manifest":
        merged = _merge_pair_genders_across_datasets(trials_info_dir, datasets, difficulties)
        if not merged:
            print(
                f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
                file=sys.stderr,
            )
            sys.exit(1)
        lines = build_xtts_manifest_lines(merged, args.utterances_per_side, normalize=args.normalize)
        out_path = os.path.abspath(args.output) if args.output else os.path.join(data_dir, f"{args.system}_voiceanon_inputs.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for line in lines:
                f.write(f"{line}\n")
        print(f"Wrote {out_path} ({len(lines)} utterances)")
        return

    if args.output:
        merged = _merge_pair_genders_across_datasets(trials_info_dir, datasets, difficulties)
        if not merged:
            print(
                f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
                file=sys.stderr,
            )
            sys.exit(1)
        utts = build_utts_dict(merged, args.utterances_per_side, normalize=args.normalize)
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(utts, f, indent=2)
        print(f"Wrote {out_path} ({len(utts)} calls)")
        return

    any_written = False
    for dataset in datasets:
        pair_gender = _collect_pair_genders_for_dataset(trials_info_dir, dataset, difficulties)
        if not pair_gender:
            print(f"Skip dataset {dataset}: no trial-info JSONs for requested difficulties.", file=sys.stderr)
            continue
        utts = build_utts_dict(pair_gender, args.utterances_per_side, normalize=args.normalize)
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
