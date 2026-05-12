"""
Evaluate LDC (full-transcript, no anonymization) SLUAR-embedded trials: AUC and EER.

Use this for the baseline where both sides of the trial are the original LDC transcripts.

Usage:
  python scripts/content_anonymization/evaluate_ldc_sluar.py config.yaml
"""

import os
import argparse
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity


def eval_cos_sim(trials: np.ndarray):
    sims, labels = [], []
    for trial in trials:
        c1, c2 = trial["call 1"], trial["call 2"]
        if c1.ndim > 1:
            c1, c2 = c1.ravel(), c2.ravel()
        sim = cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0, 0]
        sims.append(float(sim))
        labels.append(trial["label"])
    return np.array(labels), np.array(sims)


def auc_eer(y_true, y_pred):
    auc = roc_auc_score(y_true.ravel(), y_pred.ravel())
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return auc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    work_dir = os.path.abspath(cfg.get("work_dir", "."))
    trials_dir = os.path.join(work_dir, "trials")
    trials_vary_dir = os.path.join(trials_dir, "varyutts")
    out_dir = os.path.join(work_dir, "output")
    datasets = cfg.get("datasets", ["test"])
    difficulties = cfg.get("difficulties", ["hard"])
    num_utts_options = cfg.get("num_utts_options", [5, 15, 25, 45, 75, "full"])
    eval_type = cfg.get("eval_type", "test")

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for difficulty in difficulties:
        for num_utts in num_utts_options:
            if num_utts == "full":
                path = os.path.join(trials_dir, f"SLUAR_ldc_{eval_type}_{difficulty}_trials.npy")
            else:
                path = os.path.join(trials_vary_dir, f"SLUAR_ldc_utts{num_utts}_{eval_type}_{difficulty}_trials.npy")
            if not os.path.isfile(path):
                continue
            trials = np.load(path, allow_pickle=True)
            y_true, y_pred = eval_cos_sim(trials)
            auc, eer = auc_eer(y_true, y_pred)
            results.append({"difficulty": difficulty, "num utts": num_utts, "auc": auc, "eer": eer})

    if not results:
        print("No LDC trial files found. Run embed_trials_sluar.py with system=ldc first.", file=__import__("sys").stderr)
        return
    out_file = os.path.join(out_dir, f"SLUAR_ldc_varyuttsall_{eval_type}_results.txt")
    with open(out_file, "w") as f:
        f.write(f"SLUAR VARY UTTERANCES Evaluation Results ({eval_type} set)\n")
        f.write("Match: ldc (no anonymization)\n")
        f.write("---------------------------\n\n")
        for r in results:
            f.write(f"Difficulty: {r['difficulty']}\nNum of utts: {r['num utts']}\nAUC: {r['auc']:.6f}\nEER: {r['eer']:.6f}\n---------------------------\n")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
