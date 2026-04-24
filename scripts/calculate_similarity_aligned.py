"""
Compute aligned similarity (greedy + DTW) between original and paraphrased utterances per call.

Uses sentence embeddings (all-MiniLM-L6-v2) and optional normalization.
Output: per-LLM scores and optional combined txt file.

Usage:
  python scripts/calculate_similarity_aligned.py config.yaml --original data/whisper_medium_test_trials_utts.json --paraphrased data/paraphrased_gpt4omini_test_trials_utts.json [--output output/aligned_sim_scores_gpt4omini.txt]
"""

import argparse
import json
import os
import sys
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dtw import dtw
from tqdm import tqdm


def get_embeddings(original_data, paraphrased_data, batch_size=32, device="cuda"):
    original_texts = {
        cid: [u for spkr in call.values() for u in (spkr.get("text", spkr) if isinstance(spkr, dict) else spkr)]
        for cid, call in original_data.items()
    }
    paraphrased_texts = {
        cid: [u for spkr in call.values() for u in (spkr.get("text", spkr) if isinstance(spkr, dict) else spkr)]
        for cid, call in paraphrased_data.items()
    }
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    orig_emb = {cid: model.encode(texts, batch_size=batch_size) for cid, texts in tqdm(original_texts.items(), desc="Original")}
    para_emb = {cid: model.encode(texts, batch_size=batch_size) for cid, texts in tqdm(paraphrased_texts.items(), desc="Paraphrased")}
    return orig_emb, para_emb, original_texts, paraphrased_texts


def greedy_alignment_scorer(orig_emb, para_emb, orig_texts, para_texts):
    scores = []
    for cid in tqdm(orig_emb, desc="Greedy"):
        if cid not in para_emb:
            continue
        sim = cosine_similarity(orig_emb[cid], para_emb[cid])
        n_orig, n_para = len(orig_texts[cid]), len(para_texts[cid])
        if n_orig > n_para:
            sim = sim.T
            short_len, long_len = n_para, n_orig
        else:
            short_len, long_len = n_orig, n_para
        pairs = []
        used = set()
        for i in range(short_len):
            best_j = -1
            best_s = -1
            for j in range(long_len):
                if j not in used and sim[i, j] > best_s:
                    best_s = sim[i, j]
                    best_j = j
            if best_j >= 0:
                pairs.append(best_s)
                used.add(best_j)
        scores.append(np.mean(pairs) if pairs else 0.0)
    return np.mean(scores) if scores else 0.0


def dtw_similarity_scorer(orig_emb, para_emb):
    scores = []
    for cid in tqdm(orig_emb, desc="DTW"):
        if cid not in para_emb:
            continue
        cost = 1 - cosine_similarity(orig_emb[cid], para_emb[cid])
        cost = cost.astype(np.double)
        align = dtw(cost, keep_internals=True)
        scores.append(1 - align.normalizedDistance)
    return np.mean(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to config.yaml")
    ap.add_argument("--original", required=True, help="Original utterance JSON")
    ap.add_argument("--paraphrased", required=True, help="Paraphrased utterance JSON")
    ap.add_argument("--output", default=None, help="Output txt path")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    out_path = args.output or os.path.join(work_dir, "output", "aligned_sim_scores.txt")

    with open(args.original) as f:
        original_data = json.load(f)
    with open(args.paraphrased) as f:
        paraphrased_data = json.load(f)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    orig_emb, para_emb, orig_texts, para_texts = get_embeddings(original_data, paraphrased_data, args.batch_size, device)
    greedy = greedy_alignment_scorer(orig_emb, para_emb, orig_texts, para_texts)
    dtw_s = dtw_similarity_scorer(orig_emb, para_emb)
    print("Greedy alignment:", greedy)
    print("DTW similarity:", dtw_s)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Original: {args.original}\nParaphrased: {args.paraphrased}\n")
        f.write(f"Greedy Alignment Score: {greedy}\nDTW Similarity Score: {dtw_s}\n")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
