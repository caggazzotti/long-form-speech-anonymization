"""
Create one weighted pseudo-speaker XTTS profile from previously saved base profiles.

Each input base profile must contain:
  - `embed`
  - `cond`

The output file is a `.pth` with the same keys and can be consumed by
`scripts/anonymize_batch_xtts.py`.

This is a cleaned, pushable copy of the local research script `create_avg.py`.
"""

from __future__ import annotations

import argparse
import os
import random

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one weighted pseudo-speaker XTTS profile.")
    parser.add_argument(
        "--profiles-dir",
        required=True,
        help="Directory containing base XTTS `.pt` profile files.",
    )
    parser.add_argument(
        "--target-profile",
        required=True,
        help="Output profile stem, e.g. `anon_90835`.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory for the final `.pth` file.",
    )
    parser.add_argument(
        "--num-profiles-min",
        type=int,
        default=5,
        help="Minimum number of base profiles to sample.",
    )
    parser.add_argument(
        "--num-profiles-max",
        type=int,
        default=6,
        help="Maximum number of base profiles to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible profile selection.",
    )
    parser.add_argument(
        "--profile-glob-suffix",
        default=".pt",
        help="Suffix used to collect base profiles from --profiles-dir.",
    )
    return parser.parse_args()


def create_weight_tensor(size: int) -> torch.Tensor:
    weights = torch.rand(size)
    weights /= weights.sum()
    return weights


def choose_profiles(profile_paths: list[str], count: int, rng: random.Random) -> list[str]:
    if count > len(profile_paths):
        raise ValueError(f"Requested {count} profiles, but only found {len(profile_paths)} base profiles.")
    return rng.sample(profile_paths, count)


def weighted_average_profiles(profile_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speaker_embeddings = []
    gpt_cond_latents = []
    for path in profile_paths:
        profile = torch.load(path, map_location="cpu")
        speaker_embeddings.append(profile["embed"].float())
        gpt_cond_latents.append(profile["cond"].float())

    weights = create_weight_tensor(len(profile_paths)).cpu()
    reshape = weights.view(-1, 1, 1, 1)
    speaker_embedding = (reshape * torch.stack(speaker_embeddings)).sum(dim=0)
    gpt_cond_latent = (reshape * torch.stack(gpt_cond_latents)).sum(dim=0)
    return gpt_cond_latent, speaker_embedding, weights


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.num_profiles_min <= 0 or args.num_profiles_max < args.num_profiles_min:
        raise ValueError("Profile-count bounds must satisfy 1 <= min <= max.")

    profile_paths = sorted(
        os.path.join(args.profiles_dir, name)
        for name in os.listdir(args.profiles_dir)
        if name.endswith(args.profile_glob_suffix)
    )
    if not profile_paths:
        raise FileNotFoundError(f"No base profiles found under {args.profiles_dir}")

    rng = random.Random(args.seed)
    count = rng.randint(args.num_profiles_min, args.num_profiles_max)
    chosen_profiles = choose_profiles(profile_paths, count, rng)

    info_path = os.path.join(args.save_dir, "profile_info.txt")
    with open(info_path, "w") as f:
        for path in chosen_profiles:
            f.write(f"{path}\n")

    cond, embed, weights = weighted_average_profiles(chosen_profiles)
    save_dict = {"embed": embed.detach().cpu(), "cond": cond.detach().cpu()}
    output_path = os.path.join(args.save_dir, f"{args.target_profile}.pth")
    torch.save(save_dict, output_path)
    print(f"Wrote {output_path}")
    print(f"Selected {len(chosen_profiles)} base profiles -> {info_path}")
    print(f"Weights: {weights.tolist()}")


if __name__ == "__main__":
    main()
