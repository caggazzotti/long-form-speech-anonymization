"""
Prepare per-utterance speaker wavs from Fisher call audio and transcripts.

This script converts Fisher `.sph` call audio into channel-specific wavs, aligns them
with Fisher transcript timestamps, and writes one wav per utterance under:

  {output_dir}/{call_id}/{conversation_stem}_{channel}_{utterance_index}_{speaker_id}.wav

Example:
  data/utterance_audio/03780/fe_03_03780_A_29_49704.wav

These outputs are the expected inputs for `scripts/whisper_transcribe.py`.

Usage:
  python scripts/prepare_utterance_audio.py \
    --audio-root /path/to/LDC2004S13/audio \
    --audio-root /path/to/LDC2005S13/audio \
    --transcript-root /path/to/LDC2004T19/fe_03_p1_tran/data/trans \
    --transcript-root /path/to/LDC2005T19/data/trans \
    --speaker-map /path/to/LDC2004T19/fe_03_p1_tran/doc/fe_03_pindata.tbl \
    --speaker-map /path/to/LDC2005T19/doc/fe_03_pindata.tbl \
    --sph2pipe /path/to/sph2pipe \
    --output-dir data/utterance_audio
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile

import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-utterance speaker wavs from Fisher audio.")
    parser.add_argument("--audio-root", action="append", required=True, help="Root containing Fisher .sph audio files.")
    parser.add_argument(
        "--transcript-root",
        action="append",
        required=True,
        help="Root containing Fisher transcript .txt files.",
    )
    parser.add_argument(
        "--speaker-map",
        action="append",
        required=True,
        help="Path to Fisher pindata table mapping (call_id, channel) to speaker id.",
    )
    parser.add_argument("--sph2pipe", required=True, help="Path to the sph2pipe executable.")
    parser.add_argument("--output-dir", required=True, help="Output root for per-utterance wavs.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional TSV metadata output path. Defaults to {output_dir}/metadata.tsv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite utterance wavs even if they already exist.",
    )
    return parser.parse_args()


def load_speaker_map(paths: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    continue
                speaker_id = parts[0]
                sex = parts[1]
                side_blob = parts[-1]
                for entry in side_blob.split(";"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    call_channel, _, dialect = entry.partition("/")
                    if not call_channel or "_" not in call_channel:
                        continue
                    call_id, channel = call_channel.split("_", 1)
                    dialect_code = ""
                    if dialect and "." in dialect:
                        _, dialect_code = dialect.split(".", 1)
                    mapping[(call_id, channel)] = {
                        "pin": speaker_id,
                        "sex": sex,
                        "dialect": dialect_code,
                    }
    return mapping


def index_transcripts(roots: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".txt") and filename not in index:
                    index[filename] = os.path.join(dirpath, filename)
    return index


def iter_sph_files(roots: list[str]) -> list[str]:
    paths: list[str] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".sph"):
                    paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def parse_transcript_lines(path: str) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for utterance_index, raw_line in enumerate(f.readlines()[2:]):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start_time = float(parts[0])
            end_time = float(parts[1])
            channel = parts[2].rstrip(":").upper()
            transcript = " ".join(parts[3:]) if len(parts) > 3 else ""
            rows.append(
                {
                    "utterance_index": utterance_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "channel": channel,
                    "transcript": transcript,
                }
            )
    return rows


def convert_channel_wav(sph2pipe_path: str, sph_path: str, channel_number: int, wav_path: str) -> None:
    with open(wav_path, "wb") as f:
        subprocess.run(
            [sph2pipe_path, "-f", "wav", "-c", str(channel_number), sph_path],
            check=True,
            stdout=f,
        )


def extract_segment(waveform, sample_rate: int, start_time: float, end_time: float):
    start = max(0, int(round(start_time * sample_rate)))
    end = max(start, int(round(end_time * sample_rate)))
    return waveform[:, start:end]


def prepare_call(
    sph_path: str,
    transcript_path: str,
    speaker_map: dict[tuple[str, str], dict[str, str]],
    sph2pipe_path: str,
    output_dir: str,
    metadata_rows: list[dict[str, str]],
    *,
    overwrite: bool,
) -> tuple[int, int]:
    stem = os.path.splitext(os.path.basename(sph_path))[0]
    call_id = stem.split("_")[-1]
    call_output_dir = os.path.join(output_dir, call_id)
    os.makedirs(call_output_dir, exist_ok=True)

    transcript_rows = parse_transcript_lines(transcript_path)
    if not transcript_rows:
        return 0, 0

    written = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="fisher_utt_audio_") as temp_dir:
        wav_a = os.path.join(temp_dir, f"{stem}_A.wav")
        wav_b = os.path.join(temp_dir, f"{stem}_B.wav")
        convert_channel_wav(sph2pipe_path, sph_path, 1, wav_a)
        convert_channel_wav(sph2pipe_path, sph_path, 2, wav_b)
        waveform_a, sample_rate_a = torchaudio.load(wav_a)
        waveform_b, sample_rate_b = torchaudio.load(wav_b)

        if sample_rate_a != sample_rate_b:
            raise ValueError(f"Channel sample-rate mismatch for {sph_path}: {sample_rate_a} vs {sample_rate_b}")

        for row in transcript_rows:
            channel = str(row["channel"])
            speaker_meta = speaker_map.get((call_id, channel), {})
            speaker_id = speaker_meta.get("pin", "unknown")
            waveform = waveform_a if channel == "A" else waveform_b
            segment = extract_segment(
                waveform,
                sample_rate_a,
                float(row["start_time"]),
                float(row["end_time"]),
            )
            if segment.numel() == 0 or segment.shape[-1] == 0:
                skipped += 1
                continue
            out_name = f"{stem}_{channel}_{row['utterance_index']}_{speaker_id}.wav"
            out_path = os.path.join(call_output_dir, out_name)
            if os.path.exists(out_path) and not overwrite:
                metadata_rows.append(
                    {
                        "call_id": call_id,
                        "speaker_id": speaker_id,
                        "channel": channel,
                        "utterance_index": str(row["utterance_index"]),
                        "audio_path": out_path,
                        "transcript": str(row["transcript"]),
                        "start_time": str(row["start_time"]),
                        "end_time": str(row["end_time"]),
                    }
                )
                skipped += 1
                continue
            torchaudio.save(out_path, segment, sample_rate_a)
            metadata_rows.append(
                {
                    "call_id": call_id,
                    "speaker_id": speaker_id,
                    "channel": channel,
                    "utterance_index": str(row["utterance_index"]),
                    "audio_path": out_path,
                    "transcript": str(row["transcript"]),
                    "start_time": str(row["start_time"]),
                    "end_time": str(row["end_time"]),
                }
            )
            written += 1
    return written, skipped


def write_metadata(path: str, rows: list[dict[str, str]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "call_id",
        "speaker_id",
        "channel",
        "utterance_index",
        "audio_path",
        "transcript",
        "start_time",
        "end_time",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    speaker_map = load_speaker_map(args.speaker_map)
    transcript_index = index_transcripts(args.transcript_root)
    sph_paths = iter_sph_files(args.audio_root)
    metadata_rows: list[dict[str, str]] = []
    total_written = 0
    total_skipped = 0
    missing_transcripts: list[str] = []

    for sph_path in sph_paths:
        stem = os.path.splitext(os.path.basename(sph_path))[0]
        transcript_path = transcript_index.get(f"{stem}.txt")
        if transcript_path is None:
            missing_transcripts.append(sph_path)
            continue
        written, skipped = prepare_call(
            sph_path,
            transcript_path,
            speaker_map,
            args.sph2pipe,
            args.output_dir,
            metadata_rows,
            overwrite=args.overwrite,
        )
        total_written += written
        total_skipped += skipped

    metadata_path = args.metadata or os.path.join(args.output_dir, "metadata.tsv")
    write_metadata(metadata_path, metadata_rows)

    print(f"Wrote/recorded {len(metadata_rows)} utterances to {args.output_dir}")
    print(f"New wavs written: {total_written}")
    print(f"Skipped existing/empty utterances: {total_skipped}")
    print(f"Metadata: {metadata_path}")
    if missing_transcripts:
        print(f"Missing transcript files for {len(missing_transcripts)} calls", flush=True)


if __name__ == "__main__":
    main()
