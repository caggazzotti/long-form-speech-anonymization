"""
Embed trials with SLUAR (content-attack model) via Hugging Face checkpoint.

SLUAR checkpoint: https://huggingface.co/noandrews/sluar

Requires: transformers, torch. Loading uses trust_remote_code=True.

Usage:
  python scripts/content_anonymization/embed_trials_sluar.py config.yaml [--system whisper_medium] [--varyutts]
  python scripts/content_anonymization/embed_trials_sluar.py config.yaml --matched
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import yaml

def _load_model(model_id: str, token: str | None):
    from transformers import AutoModel, AutoTokenizer

    tok_kw = {"token": token, "use_fast": False, "trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    model = AutoModel.from_pretrained(model_id, token=token, trust_remote_code=True)
    return tokenizer, model


def embed_utterances(speaker_utterances: list[str], model, tokenizer, max_length: int = 512):
    """Get SLUAR embedding for one speaker's utterance sequence."""
    batch_size = 1
    tokenized = tokenizer(
        speaker_utterances,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized = tokenized.to("cuda")
    tokenized["input_ids"] = tokenized["input_ids"].reshape(batch_size, len(speaker_utterances), max_length)
    tokenized["attention_mask"] = tokenized["attention_mask"].reshape(batch_size, len(speaker_utterances), max_length)
    tokenized.pop("token_type_ids", None)
    with torch.inference_mode():
        out = model(**tokenized)
        if isinstance(out, tuple):
            embed_tensor, _ = out
        else:
            embed_tensor = out
        embedding = torch.nn.functional.normalize(embed_tensor, p=2, dim=-1)
    if embedding.dtype == torch.bfloat16:
        embedding = embedding.float()
    return embedding.cpu().numpy()


def embed_trials(trials_file: str, tokenizer, model, out_file: str, max_length: int = 512):
    """Embed each trial in the .npy trials file."""
    trials = np.load(trials_file, allow_pickle=True)
    embeddings = []
    for trial in trials:
        e1 = embed_utterances(trial["call 1"], model, tokenizer, max_length)
        e2 = embed_utterances(trial["call 2"], model, tokenizer, max_length)
        if e1.size > 0 and e2.size > 0:
            embeddings.append({"label": trial["label"], "call 1": e1, "call 2": e2})
    np.save(out_file, embeddings, allow_pickle=True)
    print(f"Embedded {len(embeddings)} trials -> {out_file}")


def embed_trials_varyutts(trials_file: str, tokenizer, model, num_utts: int, out_file: str, max_length: int = 512):
    """Embed trials using only first num_utts utterances per call."""
    trials = np.load(trials_file, allow_pickle=True)
    embeddings = []
    for trial in trials:
        e1 = embed_utterances(trial["call 1"][:num_utts], model, tokenizer, max_length)
        e2 = embed_utterances(trial["call 2"][:num_utts], model, tokenizer, max_length)
        if e1.size > 0 and e2.size > 0:
            embeddings.append({"label": trial["label"], "call 1": e1, "call 2": e2})
    np.save(out_file, embeddings, allow_pickle=True)
    print(f"Embedded {len(embeddings)} trials (utts={num_utts}) -> {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--system", default=None, help="Single system to embed (default: all in config)")
    parser.add_argument("--varyutts", action="store_true", help="Also embed vary-utterances versions")
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Embed trials/matched/*.npy text trials (non-SLUAR_*) into SLUAR_*.npy",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    trials_dir = os.path.join(work_dir, "trials")
    trials_vary_dir = os.path.join(work_dir, "trials", "varyutts")
    model_id = cfg.get("sluar_model_id", "noandrews/sluar")
    datasets = cfg.get("datasets", ["test"])
    difficulties = cfg.get("difficulties", ["hard"])
    varyutts = args.varyutts or (cfg.get("varyutts") == "yes")
    num_utts_options = cfg.get("num_utts_options", [5, 15, 25, 45, 75, "full"])
    systems = [args.system] if args.system else cfg.get("systems", [])
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    print("Loading SLUAR model", model_id)
    tokenizer, model = _load_model(model_id, token)
    model.eval()
    model.to("cuda")

    if args.matched:
        matched_dir = os.path.join(trials_dir, "matched")
        os.makedirs(matched_dir, exist_ok=True)
        in_files = sorted(
            f for f in os.listdir(matched_dir)
            if f.endswith("_trials.npy") and not f.startswith("SLUAR_")
        )
        if not in_files:
            print(f"No matched text trials found in {matched_dir}", file=sys.stderr)
            return
        for fname in in_files:
            trials_file = os.path.join(matched_dir, fname)
            out_file = os.path.join(matched_dir, f"SLUAR_{fname}")
            t0 = time.perf_counter()
            embed_trials(trials_file, tokenizer, model, out_file)
            print(f"  {(time.perf_counter() - t0) / 60:.2f} min")
        return

    if not systems:
        print("No systems to embed; set systems in config or --system", file=sys.stderr)
        sys.exit(1)

    for system in systems:
        for dataset in datasets:
            for difficulty in difficulties:
                trials_file = os.path.join(trials_dir, f"{system}_{dataset}_{difficulty}_trials.npy")
                if not os.path.isfile(trials_file):
                    ldc_dir = cfg.get("ldc_trials_dir") or (os.path.join(cfg.get("speech_attribution_dir", ""), "trials_data") if cfg.get("speech_attribution_dir") else None)
                    if system == "ldc" and ldc_dir:
                        trials_file = os.path.join(os.path.abspath(ldc_dir), f"ldc_{dataset}_{difficulty}_trials.npy")
                    else:
                        print(f"Skip {system} {dataset} {difficulty}: missing {trials_file}", file=sys.stderr)
                        continue
                if not os.path.isfile(trials_file):
                    continue
                out_file = os.path.join(trials_dir, f"SLUAR_{system}_{dataset}_{difficulty}_trials.npy")
                t0 = time.perf_counter()
                embed_trials(trials_file, tokenizer, model, out_file)
                print(f"  {(time.perf_counter() - t0) / 60:.2f} min")

                if varyutts:
                    os.makedirs(trials_vary_dir, exist_ok=True)
                    for num_utts in num_utts_options:
                        if num_utts == "full":
                            continue
                        out_v = os.path.join(trials_vary_dir, f"SLUAR_{system}_utts{num_utts}_{dataset}_{difficulty}_trials.npy")
                        t0 = time.perf_counter()
                        embed_trials_varyutts(trials_file, tokenizer, model, num_utts, out_v)
                        print(f"  utts{num_utts} {(time.perf_counter() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
