"""
Build trial .npy files from utterance JSON and speech-attribution trial info.

Utterance JSON format: {call_id: {speaker_id: {'text': [str, ...]}}}
Trial info: speech-attribution trials_data/{dataset}_basepos_trials_info_final.json etc.

Usage:
  python scripts/content_anonymization/build_trials_from_utterances.py config.yaml --system whisper_medium
  python scripts/content_anonymization/build_trials_from_utterances.py config.yaml --system paraphrased_gpt4omini --utterances data/paraphrased_gpt4omini_test_trials_utts.json
"""

import argparse
import json
import os
import sys
import numpy as np
import yaml

# Allow running from repo root: python scripts/content_anonymization/build_trials_from_utterances.py ...
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from utils import load_utterances, get_speaker_lines, normalize_text


def get_pos_transcripts(pos_trials_info_file: str, utts: dict):
    """Build positive trials from trial info and utterance dict."""
    with open(pos_trials_info_file, "r") as f:
        pos_trials_info = json.load(f)
    pos_transcripts = []
    for trial in pos_trials_info:
        pin = trial["PIN"]
        call_ID1 = trial["call 1"][1]
        call_ID2 = trial["call 2"][1]
        transcripts = []
        for call_id in (call_ID1, call_ID2):
            speaker_lines = get_speaker_lines(utts, str(call_id), str(pin))
            if speaker_lines:
                speaker_lines = [normalize_text(t) for t in speaker_lines]
                transcripts.append(speaker_lines)
        if len(transcripts) == 2:
            pos_transcripts.append({"label": 1, "call 1": transcripts[0], "call 2": transcripts[1]})
    return pos_transcripts


def get_neg_transcripts(neg_trials_info_file: str, utts: dict):
    """Build negative trials from trial info and utterance dict."""
    with open(neg_trials_info_file, "r") as f:
        neg_trials_info = json.load(f)
    neg_transcripts = []
    for trial in neg_trials_info:
        pin1, call_ID1 = trial[0][0], trial[0][2]
        pin2, call_ID2 = trial[1][0], trial[1][2]
        transcripts = []
        for (call_id, pin) in ((call_ID1, pin1), (call_ID2, pin2)):
            speaker_lines = get_speaker_lines(utts, str(call_id), str(pin))
            if speaker_lines is not None:
                speaker_lines = [normalize_text(t) for t in speaker_lines]
                transcripts.append(speaker_lines)
            else:
                transcripts.append([])
        if len(transcripts) == 2:
            neg_transcripts.append({"label": 0, "call 1": transcripts[0], "call 2": transcripts[1]})
    return neg_transcripts


def difficulty_to_trial_types(difficulty: str):
    if difficulty == "base":
        return "basepos", "baseneg"
    if difficulty == "hard":
        return "hardpos", "hardneg"
    if difficulty == "harder":
        return "hardpos", "harderneg"
    raise ValueError(f"Unknown difficulty: {difficulty}")


def main():
    parser = argparse.ArgumentParser(description="Build trial .npy from utterance JSON and trial info")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--system", required=True, help="System name (e.g. whisper_medium, paraphrased_gpt4omini)")
    parser.add_argument("--utterances", default=None, help="Path to utterance JSON (default: data/{system}_test_trials_utts.json)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override config datasets")
    parser.add_argument("--difficulties", nargs="+", default=None, help="Override config difficulties")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    data_dir = os.path.join(work_dir, "data")
    out_dir = os.path.join(work_dir, "trials")
    trials_info_dir = cfg.get("trials_info_dir")
    if not trials_info_dir:
        speech_attr = cfg.get("speech_attribution_dir")
        if speech_attr:
            trials_info_dir = os.path.join(os.path.abspath(speech_attr), "trials_data")
        else:
            print("Set trials_info_dir or speech_attribution_dir in config", file=sys.stderr)
            sys.exit(1)
    trials_info_dir = os.path.abspath(trials_info_dir)

    datasets = args.datasets or cfg.get("datasets", ["test"])
    difficulties = args.difficulties or cfg.get("difficulties", ["hard"])

    if args.utterances:
        utts_path = os.path.abspath(args.utterances)
    else:
        utts_path = os.path.join(data_dir, f"{args.system}_test_trials_utts.json")
    if not os.path.isfile(utts_path):
        print(f"Utterance file not found: {utts_path}", file=sys.stderr)
        sys.exit(1)

    utts = load_utterances(utts_path)
    os.makedirs(out_dir, exist_ok=True)

    for dataset in datasets:
        for difficulty in difficulties:
            pos_type, neg_type = difficulty_to_trial_types(difficulty)

            pos_file = os.path.join(trials_info_dir, f"{dataset}_{pos_type}_trials_info_final.json")
            neg_file = os.path.join(trials_info_dir, f"{dataset}_{neg_type}_trials_info_final.json")

            pos_trials = get_pos_transcripts(pos_file, utts)
            neg_trials = get_neg_transcripts(neg_file, utts)
            combined = pos_trials + neg_trials
            out_path = os.path.join(out_dir, f"{args.system}_{dataset}_{difficulty}_trials.npy")
            np.save(out_path, combined, allow_pickle=True)
            print(f"Wrote {len(combined)} trials -> {out_path}")


if __name__ == "__main__":
    main()
