"""
Build matched (non-anon vs anon) text trial files directly from:
  - speech-attribution trial info JSONs, and
  - utterance JSONs for Whisper + anonymized/paraphrased systems.

Pairs trials so that call 1 is from the non-anonymized system and call 2 from
the anonymized/paraphrased system, by trial index.

Usage:
  python scripts/content_anonymization/match_trials.py config.yaml
"""
from __future__ import annotations

import os
import re
import sys
import argparse
import json
import numpy as np
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from utils import load_utterances, get_speaker_lines, normalize_text


def difficulty_to_trial_types(difficulty: str) -> tuple[str, str]:
    if difficulty == "base":
        return "basepos", "baseneg"
    if difficulty == "hard":
        return "hardpos", "hardneg"
    if difficulty == "harder":
        return "hardpos", "harderneg"
    raise ValueError(f"Unknown difficulty: {difficulty}")


def system_to_short_name(system: str) -> str:
    """Short name for matched output filenames (e.g. gpt4omini_paraphrased)."""
    if "whisper" in system:
        return "whisper"
    if "gpt5" in system:
        if "paraphrased" in system:
            return "gpt5"
        if "voiceanonpara" in system:
            return "voiceanongpt5"
    if "voiceanonymized" in system:
        return "voiceanon"
    if "gpt4omini" in system:
        if "paraphrased" in system:
            return "gpt4omini"
        if "voiceanonpara" in system:
            return "voiceanongpt4omini"
    if "gemma" in system:
        name = re.split("_", system)[1]
        if "paraphrased" in system:
            return name
        if "voiceanonpara" in system:
            return "voiceanon" + name
    return system


def match_entry_to_anon_system(entry: str) -> str:
    """
    Accept either:
      - anon system name (e.g. paraphrased_gpt4omini), or
      - matched label (e.g. whisper-gpt4omini)
    and return anon system name used in trials filenames.
    """
    if entry.startswith("whisper-"):
        target = entry.split("-", 1)[1]
        mapping = {
            "voiceanon": "voiceanonymized",
            "gpt4omini": "paraphrased_gpt4omini",
            "gpt5": "paraphrased_gpt5",
            "gemma3-4b": "paraphrased_gemma3-4b",
            "gemma3-4bc": "paraphrased_gemma3-4bc",
            "voiceanongpt4omini": "voiceanonpara_gpt4omini",
            "voiceanongpt5": "voiceanonpara_gpt5",
            "voiceanongemma3-4b": "voiceanonpara_gemma3-4b",
            "voiceanongemma3-4bc": "voiceanonpara_gemma3-4bc",
        }
        return mapping.get(target, target)
    return entry


def _resolve_trials_info_dir(cfg: dict) -> str:
    trials_info_dir = cfg.get("trials_info_dir")
    if trials_info_dir:
        return os.path.abspath(trials_info_dir)
    speech_attr = cfg.get("speech_attribution_dir")
    if speech_attr:
        return os.path.join(os.path.abspath(speech_attr), "trials_data")
    raise ValueError("Set trials_info_dir or speech_attribution_dir in config")


def _resolve_utts_path(work_dir: str, system: str) -> str:
    return os.path.join(work_dir, "data", f"{system}_test_trials_utts.json")


def _build_matched_trials_from_info(
    pos_info_file: str,
    neg_info_file: str,
    noanon_utts: dict,
    anon_utts: dict,
    num_utts: int | None = None,
) -> list:
    with open(pos_info_file, "r") as f:
        pos_info = json.load(f)
    with open(neg_info_file, "r") as f:
        neg_info = json.load(f)

    matched: list[dict] = []

    # Positive: same PIN appears on both sides.
    for trial in pos_info:
        pin = str(trial["PIN"])
        call1 = str(trial["call 1"][1])
        call2 = str(trial["call 2"][1])
        c1 = get_speaker_lines(noanon_utts, call1, pin)
        c2 = get_speaker_lines(anon_utts, call2, pin)
        if c1 is None or c2 is None:
            continue
        c1 = [normalize_text(t) for t in c1]
        c2 = [normalize_text(t) for t in c2]
        if num_utts is not None:
            c1, c2 = c1[:num_utts], c2[:num_utts]
        matched.append({"label": 1, "call 1": c1, "call 2": c2})

    # Negative: different PIN per side.
    for trial in neg_info:
        pin1, call1 = str(trial[0][0]), str(trial[0][2])
        pin2, call2 = str(trial[1][0]), str(trial[1][2])
        c1 = get_speaker_lines(noanon_utts, call1, pin1)
        c2 = get_speaker_lines(anon_utts, call2, pin2)
        if c1 is None or c2 is None:
            continue
        c1 = [normalize_text(t) for t in c1]
        c2 = [normalize_text(t) for t in c2]
        if num_utts is not None:
            c1, c2 = c1[:num_utts], c2[:num_utts]
        matched.append({"label": 0, "call 1": c1, "call 2": c2})

    return matched


def _resolve_trial_info_files(trials_info_dir: str, dataset: str, difficulty: str) -> tuple[str, str] | tuple[None, None]:
    pos_type, neg_type = difficulty_to_trial_types(difficulty)
    pos_file = os.path.join(trials_info_dir, f"{dataset}_{pos_type}_trials_info_final.json")
    neg_file = os.path.join(trials_info_dir, f"{dataset}_{neg_type}_trials_info_final.json")
    if os.path.isfile(pos_file) and os.path.isfile(neg_file):
        return pos_file, neg_file
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    trials_dir = os.path.join(work_dir, "trials")
    matched_dir = os.path.join(trials_dir, "matched")
    trials_info_dir = _resolve_trials_info_dir(cfg)
    datasets = cfg.get("datasets", ["test"])
    difficulties = cfg.get("difficulties", ["hard"])
    num_utts_options = cfg.get("num_utts_options", [5, 15, 25, 45, 75, "full"])
    matched_systems = cfg.get("matched_systems", [])
    noanon_system = "whisper_medium"

    if not matched_systems:
        print("Set matched_systems in config (e.g. whisper-gpt4omini, whisper-gemma3-4b)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(matched_dir, exist_ok=True)
    noanon_short = system_to_short_name(noanon_system)
    noanon_utts_path = _resolve_utts_path(work_dir, noanon_system)
    if not os.path.isfile(noanon_utts_path):
        print(f"Missing non-anon utterances file: {noanon_utts_path}", file=sys.stderr)
        sys.exit(1)
    noanon_utts = load_utterances(noanon_utts_path)

    for entry in matched_systems:
        anon_system = match_entry_to_anon_system(entry)
        anon_short = system_to_short_name(anon_system)
        anon_utts_path = _resolve_utts_path(work_dir, anon_system)
        if not os.path.isfile(anon_utts_path):
            print(f"Skip {anon_system}: missing utterances file {anon_utts_path}", file=sys.stderr)
            continue
        anon_utts = load_utterances(anon_utts_path)
        for dataset in datasets:
            for difficulty in difficulties:
                pos_file, neg_file = _resolve_trial_info_files(trials_info_dir, dataset, difficulty)
                if pos_file is None or neg_file is None:
                    print(f"Skip {dataset} {difficulty}: missing trial info files", file=sys.stderr)
                    continue
                for num_utts in num_utts_options:
                    if num_utts != "full":
                        out_file = os.path.join(matched_dir, f"{noanon_short}-{anon_short}_utts{num_utts}_{dataset}_{difficulty}_trials.npy")
                        matched = _build_matched_trials_from_info(pos_file, neg_file, noanon_utts, anon_utts, num_utts=num_utts)
                    else:
                        out_file = os.path.join(matched_dir, f"{noanon_short}-{anon_short}_{dataset}_{difficulty}_trials.npy")
                        matched = _build_matched_trials_from_info(pos_file, neg_file, noanon_utts, anon_utts, num_utts=None)
                    np.save(out_file, matched, allow_pickle=True)
                    print(f"Matched {len(matched)} -> {out_file}")


if __name__ == "__main__":
    main()
