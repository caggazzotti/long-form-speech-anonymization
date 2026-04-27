"""
Batch voice anonymization with XTTS-v2 from `fname|transcript` input lines.

Input format:
  fe_11393_90835_114.wav|yeah

The script infers the speaker id from the filename (`90835` in the example
above), loads the corresponding precomputed XTTS profile, synthesizes the
transcript, and writes one anonymized wav per input line.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torchaudio
from TTS.api import TTS


DEFAULT_PROFILE_PATTERN = "pseudo-spks/{speaker_id}/anon_{speaker_id}.pth"


@dataclass
class SynthItem:
    filename: str
    transcript: str
    speaker_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch XTTS-based voice anonymization.")
    parser.add_argument("--stm-file", required=True, help="Path to input file with `filename|transcript` rows.")
    parser.add_argument(
        "--profile-dir",
        required=True,
        help="Base directory containing saved XTTS speaker profiles.",
    )
    parser.add_argument("--save-dir", required=True, help="Output directory for anonymized wav files.")
    parser.add_argument(
        "--profile-pattern",
        default=DEFAULT_PROFILE_PATTERN,
        help=(
            "Relative path under --profile-dir for each speaker profile. "
            "Available placeholder: {speaker_id}. "
            f"Default: {DEFAULT_PROFILE_PATTERN}"
        ),
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. `cuda` or `cpu`.")
    parser.add_argument(
        "--xtts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Model id passed to Coqui TTS.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to XTTS inference.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for saved audio.",
    )
    return parser.parse_args()


def parse_input_line(line: str, line_number: int) -> SynthItem | None:
    row = line.strip()
    if not row:
        return None

    parts = row.split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"Line {line_number}: expected `filename|transcript`, got {row!r}")

    filename, transcript = parts[0].strip(), parts[1].strip()
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem_parts = stem.split("_")
    if len(stem_parts) < 3:
        raise ValueError(
            f"Line {line_number}: expected filename like `fe_11393_90835_114.wav`, got {filename!r}"
        )

    speaker_id = stem_parts[2]
    if not transcript:
        raise ValueError(f"Line {line_number}: empty transcript for {filename!r}")

    return SynthItem(filename=filename, transcript=transcript, speaker_id=speaker_id)


def resolve_profile_path(profile_dir: str, profile_pattern: str, speaker_id: str) -> str:
    relative = profile_pattern.format(speaker_id=speaker_id)
    return os.path.join(profile_dir, relative)


def synthesize(
    xtts_model,
    tts_config,
    text: str,
    speaker_embedding: torch.Tensor,
    conditioning_latent: torch.Tensor,
    language: str,
) -> list[float]:
    outputs = xtts_model.inference(
        text,
        language,
        conditioning_latent,
        speaker_embedding,
        temperature=tts_config.temperature,
        length_penalty=tts_config.length_penalty,
        repetition_penalty=tts_config.repetition_penalty,
        top_k=tts_config.top_k,
        top_p=tts_config.top_p,
        do_sample=True,
    )
    return outputs["wav"]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tts = TTS(args.xtts_model).to(device)
    os.makedirs(args.save_dir, exist_ok=True)

    processed = 0
    skipped_existing = 0
    skipped_missing_profile = 0
    failed = 0

    with open(args.stm_file) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = parse_input_line(line, line_number)
                if item is None:
                    continue

                output_dir = os.path.join(args.save_dir, item.speaker_id)
                output_path = os.path.join(output_dir, item.filename)
                if os.path.exists(output_path):
                    print(f"Skipping existing output: {output_path}")
                    skipped_existing += 1
                    continue

                profile_path = resolve_profile_path(args.profile_dir, args.profile_pattern, item.speaker_id)
                if not os.path.exists(profile_path):
                    print(f"Missing profile for speaker {item.speaker_id}: {profile_path}")
                    skipped_missing_profile += 1
                    continue

                profile = torch.load(profile_path, map_location=device)
                speaker_embedding = profile["embed"].to(device)
                conditioning_latent = profile["cond"].to(device)
                audio = synthesize(
                    tts.synthesizer.tts_model,
                    tts.synthesizer.tts_config,
                    item.transcript,
                    speaker_embedding,
                    conditioning_latent,
                    args.language,
                )

                os.makedirs(output_dir, exist_ok=True)
                waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).detach().cpu()
                torchaudio.save(output_path, waveform, args.sample_rate)
                print(f"Wrote {output_path}")
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"Error on line {line_number}: {exc!r}")

    print(
        "Done: "
        f"processed={processed} skipped_existing={skipped_existing} "
        f"skipped_missing_profile={skipped_missing_profile} failed={failed}"
    )


if __name__ == "__main__":
    main()
