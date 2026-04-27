"""
Build one pseudo-speaker XTTS profile per speaker id.

Input:
  - a text file with one speaker id per line
  - a directory of base XTTS profiles

Output:
  pseudo-spks/{speaker_id}/anon_{speaker_id}.pth
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pseudo-speaker XTTS profiles for a list of speakers.")
    parser.add_argument("--speaker-list", required=True, help="Text file with one speaker id per line.")
    parser.add_argument("--profiles-dir", required=True, help="Directory containing base XTTS `.pt` profiles.")
    parser.add_argument("--output-root", required=True, help="Root directory for pseudo-speaker profiles.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for create_xtts_pseudo_profile.py.",
    )
    return parser.parse_args()


def load_speaker_ids(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    speaker_ids = load_speaker_ids(args.speaker_list)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_xtts_pseudo_profile.py")

    for speaker_id in speaker_ids:
        speaker_dir = os.path.join(args.output_root, speaker_id)
        output_path = os.path.join(speaker_dir, f"anon_{speaker_id}.pth")
        if os.path.isfile(output_path):
            print(f"Skipping existing pseudo profile for speaker {speaker_id}")
            continue

        os.makedirs(speaker_dir, exist_ok=True)
        print(f"Building pseudo profile for speaker {speaker_id}")
        subprocess.run(
            [
                args.python_bin,
                script_path,
                "--profiles-dir",
                args.profiles_dir,
                "--target-profile",
                f"anon_{speaker_id}",
                "--save-dir",
                speaker_dir,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
