"""
Build Whisper-side transcription outputs for Fisher calls.

Expected audio input is a directory of pre-segmented speaker utterance wavs, grouped
by call ID, with filenames that encode call ID, channel, utterance index, and speaker
ID. Example:
  {utterance_audio_dir}/03780/fe_03_03780_A_29_49704.wav

Default output is the utterance JSON used by the content / matched-trial pipeline:
  { call_id: { speaker_pin: { "text": [str, ...], "gender": "m"|"f" } } }

This script can also emit XTTS-style `filename.wav|transcript` rows for the voice
anonymization pipeline.

Usage:
  python scripts/whisper_transcribe.py config.yaml
  python scripts/whisper_transcribe.py config.yaml --system whisper_medium
  python scripts/whisper_transcribe.py config.yaml --output-format xtts_manifest --output data/voiceanon_inputs.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
import whisper
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from match_trials import _resolve_trials_info_dir, difficulty_to_trial_types
from paraphrase_responses_to_utterances import normalize_text as normalize_utterance_line


_UTT_WAV_RE = re.compile(
    r"^(?P<stem>.+)_(?P<call_id>\d+)_(?P<channel>[AB])_(?P<utt_index>\d+)_(?P<speaker_id>\d+)\.wav$"
)


def _parse_utt_wav_name(path: str) -> dict[str, object] | None:
    match = _UTT_WAV_RE.match(os.path.basename(path))
    if not match:
        return None
    parsed = match.groupdict()
    parsed["utt_index"] = int(parsed["utt_index"])
    return parsed


def _normalize_gender(raw: object) -> str:
    if raw is None:
        return ""
    g = str(raw).strip().lower()
    if g in ("m", "male"):
        return "m"
    if g in ("f", "female"):
        return "f"
    return ""


def _merge_pair_meta(
    dst: dict[tuple[str, str], dict[str, str]],
    call_id: str,
    pin: str,
    *,
    gender: str,
    channel: str,
) -> None:
    key = (call_id, pin)
    if key not in dst:
        dst[key] = {"gender": gender or "", "channel": channel or ""}
        return

    if gender and not dst[key]["gender"]:
        dst[key]["gender"] = gender
    elif gender and dst[key]["gender"] and dst[key]["gender"] != gender:
        print(
            f"Warning: conflicting gender for call {call_id} pin {pin}: {dst[key]['gender']!r} vs {gender!r}",
            file=sys.stderr,
        )

    if channel and not dst[key]["channel"]:
        dst[key]["channel"] = channel
    elif channel and dst[key]["channel"] and dst[key]["channel"] != channel:
        print(
            f"Warning: conflicting channel for call {call_id} pin {pin}: {dst[key]['channel']!r} vs {channel!r}",
            file=sys.stderr,
        )


def _normalize_channel(raw: object) -> str:
    if raw is None:
        return ""
    channel = str(raw).strip().upper()
    return channel if channel in {"A", "B"} else ""


def _collect_pair_meta_for_dataset(
    trials_info_dir: str, dataset: str, difficulties: list[str]
) -> dict[tuple[str, str], dict[str, str]]:
    """(call_id, pin) -> {'gender': 'm'|'f'|'', 'channel': 'A'|'B'|''}."""
    out: dict[tuple[str, str], dict[str, str]] = {}

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
            c1 = _normalize_channel(row1[2]) if len(row1) > 2 else ""
            c2 = _normalize_channel(row2[2]) if len(row2) > 2 else ""
            _merge_pair_meta(out, call1, pin, gender=g1, channel=c1)
            _merge_pair_meta(out, call2, pin, gender=g2, channel=c2)

        with open(neg_path) as f:
            neg_info = json.load(f)
        for trial in neg_info:
            pin1 = str(trial[0][0])
            g1 = _normalize_gender(trial[0][1])
            call1 = str(trial[0][2])
            c1 = _normalize_channel(trial[0][3]) if len(trial[0]) > 3 else ""
            pin2 = str(trial[1][0])
            g2 = _normalize_gender(trial[1][1])
            call2 = str(trial[1][2])
            c2 = _normalize_channel(trial[1][3]) if len(trial[1]) > 3 else ""
            _merge_pair_meta(out, call1, pin1, gender=g1, channel=c1)
            _merge_pair_meta(out, call2, pin2, gender=g2, channel=c2)

    return out


def _sort_call_id(cid: str) -> tuple[int, str]:
    return (int(cid), cid) if str(cid).isdigit() else (10**18, cid)


def _sort_pin(pid: str) -> tuple[int, str]:
    return (int(pid), pid) if str(pid).isdigit() else (10**18, pid)


def _candidate_call_dirs(utterance_audio_dir: str, call_id: str) -> list[str]:
    candidates = []
    for candidate in {call_id, call_id.zfill(5)}:
        path = os.path.join(utterance_audio_dir, candidate)
        if os.path.isdir(path):
            candidates.append(path)
    return candidates


def _resolve_audio_paths(utterance_audio_dir: str, call_id: str, pin: str, channel: str) -> list[str]:
    if not channel:
        raise ValueError(
            f"Missing channel for call {call_id} speaker {pin}. Trial-info JSON must provide 'A' or 'B'."
        )

    matches: list[tuple[int, str]] = []
    for call_dir in _candidate_call_dirs(utterance_audio_dir, call_id):
        for wav_path in glob.glob(os.path.join(call_dir, "*.wav")):
            parsed = _parse_utt_wav_name(wav_path)
            if not parsed:
                continue
            if str(parsed["call_id"]) != str(call_id):
                continue
            if str(parsed["speaker_id"]) != str(pin):
                continue
            if str(parsed["channel"]) != str(channel):
                continue
            matches.append((int(parsed["utt_index"]), wav_path))

    if not matches:
        raise FileNotFoundError(
            f"No utterance wavs found for call {call_id} speaker {pin} channel {channel} under {utterance_audio_dir}"
        )

    matches.sort(key=lambda item: (item[0], item[1]))
    return [path for _, path in matches]


def _transcribe_audio_paths(model, audio_paths: list[str]) -> list[str]:
    texts: list[str] = []
    for audio_path in audio_paths:
        result = model.transcribe(audio_path)
        text = str(result.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def build_utts_dict(
    pair_meta: dict[tuple[str, str], dict[str, str]],
    model,
    utterance_audio_dir: str,
    *,
    normalize: bool,
) -> dict:
    """Pipeline utterance JSON: call -> pin -> {text, gender?}."""
    by_call: dict[str, dict] = {}
    for (call_id, pin) in sorted(pair_meta.keys(), key=lambda t: (_sort_call_id(t[0]), _sort_pin(t[1]))):
        meta = pair_meta[(call_id, pin)]
        gender = meta["gender"]
        channel = meta["channel"]
        audio_paths = _resolve_audio_paths(utterance_audio_dir, call_id, pin, channel)
        raw = _transcribe_audio_paths(model, audio_paths)
        text = [normalize_utterance_line(s) for s in raw] if normalize else raw
        speaker: dict = {"text": text}
        if gender:
            speaker["gender"] = gender
        by_call.setdefault(call_id, {})[pin] = speaker
    return by_call


def build_xtts_manifest_lines(
    pair_meta: dict[tuple[str, str], dict[str, str]],
    model,
    utterance_audio_dir: str,
    *,
    normalize: bool,
) -> list[str]:
    """
    XTTS manifest rows:
      fe_{call_id}_{speaker_id}_{segment_index}.wav|transcript

    This keeps a deterministic filename convention for downstream anonymization.
    """
    lines: list[str] = []
    for (call_id, pin) in sorted(pair_meta.keys(), key=lambda t: (_sort_call_id(t[0]), _sort_pin(t[1]))):
        meta = pair_meta[(call_id, pin)]
        audio_paths = _resolve_audio_paths(utterance_audio_dir, call_id, pin, meta["channel"])
        raw = _transcribe_audio_paths(model, audio_paths)
        text_items = [normalize_utterance_line(s) for s in raw] if normalize else raw
        for index, transcript in enumerate(text_items, start=1):
            filename = f"fe_{call_id}_{pin}_{index}.wav"
            lines.append(f"{filename}|{transcript}")
    return lines


def _merge_pair_meta_across_datasets(
    trials_info_dir: str, datasets: list[str], difficulties: list[str]
) -> dict[tuple[str, str], dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in datasets:
        part = _collect_pair_meta_for_dataset(trials_info_dir, dataset, difficulties)
        for key, meta in part.items():
            if key not in merged:
                merged[key] = dict(meta)
                continue

            if meta["gender"] and not merged[key]["gender"]:
                merged[key]["gender"] = meta["gender"]
            elif meta["gender"] and merged[key]["gender"] and merged[key]["gender"] != meta["gender"]:
                print(
                    f"Warning: conflicting gender for {key} across datasets: {merged[key]['gender']!r} vs {meta['gender']!r}",
                    file=sys.stderr,
                )
            if meta["channel"] and not merged[key]["channel"]:
                merged[key]["channel"] = meta["channel"]
            elif meta["channel"] and merged[key]["channel"] and merged[key]["channel"] != meta["channel"]:
                print(
                    f"Warning: conflicting channel for {key} across datasets: {merged[key]['channel']!r} vs {meta['channel']!r}",
                    file=sys.stderr,
                )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Whisper transcription outputs from pre-segmented speaker audio.")
    parser.add_argument("config", help="Path to config.yaml (needs trials_info_dir or speech_attribution_dir)")
    parser.add_argument("--system", default="whisper_medium", help="Prefix for output filename under data/")
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
    parser.add_argument("--model-name", default="medium.en", help="Whisper model name to load.")
    parser.add_argument("--device", default=None, help="Whisper device override (default: auto-detect).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    data_dir = os.path.join(work_dir, "data")
    datasets = cfg.get("datasets", ["test"])
    difficulties = cfg.get("difficulties", ["hard"])
    utterance_audio_dir_cfg = (
        cfg.get("utterance_audio_dir")
        or cfg.get("speaker_utterance_audio_dir")
        or cfg.get("utterance_wav_dir")
    )
    if not utterance_audio_dir_cfg:
        print(
            "Config error: set utterance_audio_dir (or speaker_utterance_audio_dir / utterance_wav_dir) "
            "to a directory of pre-segmented speaker utterance wavs.",
            file=sys.stderr,
        )
        sys.exit(1)
    utterance_audio_dir = os.path.abspath(utterance_audio_dir_cfg)
    if not os.path.isdir(utterance_audio_dir):
        print(f"Config error: utterance audio directory not found: {utterance_audio_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        trials_info_dir = _resolve_trials_info_dir(cfg)
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(data_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(args.model_name, device=device)

    if args.output_format == "xtts_manifest":
        merged = _merge_pair_meta_across_datasets(trials_info_dir, datasets, difficulties)
        if not merged:
            print(
                f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
                file=sys.stderr,
            )
            sys.exit(1)
        lines = build_xtts_manifest_lines(merged, model, utterance_audio_dir, normalize=args.normalize)
        out_path = os.path.abspath(args.output) if args.output else os.path.join(data_dir, f"{args.system}_voiceanon_inputs.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            for line in lines:
                f.write(f"{line}\n")
        print(f"Wrote {out_path} ({len(lines)} utterances)")
        return

    if args.output:
        merged = _merge_pair_meta_across_datasets(trials_info_dir, datasets, difficulties)
        if not merged:
            print(
                f"No trial-info JSONs found under {trials_info_dir} for datasets={datasets} difficulties={difficulties}.",
                file=sys.stderr,
            )
            sys.exit(1)
        utts = build_utts_dict(merged, model, utterance_audio_dir, normalize=args.normalize)
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(utts, f, indent=2)
        print(f"Wrote {out_path} ({len(utts)} calls)")
        return

    any_written = False
    for dataset in datasets:
        pair_meta = _collect_pair_meta_for_dataset(trials_info_dir, dataset, difficulties)
        if not pair_meta:
            print(f"Skip dataset {dataset}: no trial-info JSONs for requested difficulties.", file=sys.stderr)
            continue
        utts = build_utts_dict(pair_meta, model, utterance_audio_dir, normalize=args.normalize)
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
