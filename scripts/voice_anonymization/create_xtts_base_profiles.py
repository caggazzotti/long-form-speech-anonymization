"""
Create base XTTS speaker profiles from reference wav paths.

Each output file is a `.pt` containing:
  - `embed`: XTTS speaker embedding
  - `cond`: XTTS GPT conditioning latent

Input manifest format:
  speaker_id<TAB>audio_path

Example:
  id00012\t/path/to/ref.wav


"""

from __future__ import annotations

import argparse
import os

import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.models.xtts import load_audio, wav_to_mel_cloning


DEFAULT_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create base XTTS profiles from reference audio.")
    parser.add_argument(
        "--input-list",
        required=True,
        help="Tab-separated file with `speaker_id<TAB>audio_path` per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for base profile `.pt` files.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. `cuda` or `cpu`.",
    )
    parser.add_argument(
        "--xtts-model",
        default=DEFAULT_XTTS_MODEL,
        help="Model id passed to Coqui TTS.",
    )
    parser.add_argument(
        "--output-pattern",
        default="speaker_{speaker_id}_index_{index}.pt",
        help="Filename pattern for saved base profiles.",
    )
    return parser.parse_args()


def get_gpt_cond_latents(xtts_model, audio, sr, length: int = 30, chunk_length: int = 6):
    if sr != 22050:
        audio = torchaudio.functional.resample(audio, sr, 22050)
    if length > 0:
        audio = audio[:, : 22050 * length]
    if xtts_model.args.gpt_use_perceiver_resampler:
        style_embs = []
        for i in range(0, audio.shape[1], 22050 * chunk_length):
            audio_chunk = audio[:, i : i + 22050 * chunk_length]
            if audio_chunk.size(-1) < 22050 * 0.33:
                continue
            mel_chunk = wav_to_mel_cloning(
                audio_chunk,
                mel_norms=xtts_model.mel_stats.cpu(),
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            style_emb = xtts_model.gpt.get_style_emb(mel_chunk.to(xtts_model.device), None)
            style_embs.append(style_emb)
        if not style_embs:
            raise ValueError("No valid audio chunks found for GPT conditioning.")
        cond_latent = torch.stack(style_embs).mean(dim=0)
    else:
        mel = wav_to_mel_cloning(
            audio,
            mel_norms=xtts_model.mel_stats.cpu(),
            n_fft=4096,
            hop_length=1024,
            win_length=4096,
            power=2,
            normalized=False,
            sample_rate=22050,
            f_min=0,
            f_max=8000,
            n_mels=80,
        )
        cond_latent = xtts_model.gpt.get_style_emb(mel.to(xtts_model.device))
    return cond_latent.transpose(1, 2)


@torch.inference_mode()
def get_conditioning_latents(
    xtts_model,
    audio_path: str,
    max_ref_length=30,
    gpt_cond_len=6,
    gpt_cond_chunk_len=6,
    load_sr=22050,
):
    audio = load_audio(audio_path, load_sr)
    audio = audio[:, : load_sr * max_ref_length].to(xtts_model.device)
    speaker_embedding = xtts_model.get_speaker_embedding(audio, load_sr)
    gpt_cond_latent = get_gpt_cond_latents(
        xtts_model,
        audio,
        load_sr,
        length=gpt_cond_len,
        chunk_length=gpt_cond_chunk_len,
    )
    return gpt_cond_latent, speaker_embedding


def load_manifest(path: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Line {line_number}: expected `speaker_id<TAB>audio_path`.")
            speaker_id, audio_path = parts[0].strip(), parts[-1].strip()
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Line {line_number}: audio file not found: {audio_path}")
            items.append((speaker_id, audio_path))
    return items


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tts = TTS(args.xtts_model).to(device)
    cfg = tts.synthesizer.tts_config
    os.makedirs(args.output_dir, exist_ok=True)

    items = load_manifest(args.input_list)
    for index, (speaker_id, audio_path) in enumerate(items, start=1):
        cond, embed = get_conditioning_latents(
            tts.synthesizer.tts_model,
            audio_path=audio_path,
            gpt_cond_len=cfg.gpt_cond_len,
            gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
            max_ref_length=cfg.max_ref_len,
        )
        output_name = args.output_pattern.format(speaker_id=speaker_id, index=index)
        output_path = os.path.join(args.output_dir, output_name)
        torch.save({"embed": embed.detach().cpu(), "cond": cond.detach().cpu()}, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
