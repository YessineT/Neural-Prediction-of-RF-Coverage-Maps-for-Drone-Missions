"""
How many valid cells are at the -150 dB floor ("no signal"), and how much do they weigh in a squared error?
Needs only numpy (and huggingface_hub to download the data). Put it next to baselines.py and run:
    python floor_stats.py                 # splits from Hugging Face (cached)
    python floor_stats.py --data hf_data  # local folder
For every split it prints
    - share of valid cells at the floor
    - share of maps that have any floor cell / more than 10 % of floor cells
    - for a predictor that outputs the mean path gain everywhere: which share of its total squared error comes from
      the floor cells (an indication of how much they weigh in an MSE)
"""
import argparse

import numpy as np

from baselines import HF_REPO, load_split

FLOOR_DB = -150.0


def stats(z, mean_db):
    n_valid = n_floor = 0
    maps_any = maps_10 = 0
    se_floor = se_all = 0.0
    for i in range(len(z["altitude"])):
        valid = z["heights"][z["scene_idx"][i]] < z["altitude"][i] - 0.5
        y = z["target_db"][i][valid].astype(np.float64)
        fl = y <= FLOOR_DB + 1e-3
        n_valid += len(y)
        n_floor += fl.sum()
        maps_any += fl.any()
        maps_10 += fl.mean() > 0.10
        e2 = (y - mean_db) ** 2
        se_all += e2.sum()
        se_floor += e2[fl].sum()
    return dict(maps=len(z["altitude"]), floor_share=n_floor / n_valid, maps_any=maps_any / len(z["altitude"]),
                maps_10=maps_10 / len(z["altitude"]), se_share=se_floor / se_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "ood"])
    args = ap.parse_args()
    train = load_split("train", args)
    s = c = 0.0
    for i in range(len(train["altitude"])):
        valid = train["heights"][train["scene_idx"][i]] < train["altitude"][i] - 0.5
        y = train["target_db"][i][valid].astype(np.float64)
        s += y.sum()
        c += len(y)
    mean_db = s / c
    print(f"mean path gain of the training cells: {mean_db:.1f} dB (floor: {FLOOR_DB:.0f} dB, {mean_db - FLOOR_DB:.0f} dB below the mean)\n")
    print(f"{'split':6s} {'maps':>6s} {'cells at floor':>15s} {'maps with any':>14s} {'maps with >10%':>15s} {'floor share of squared error':>30s}")
    for split in args.splits:
        z = train if split == "train" else load_split(split, args)
        if z is None:
            continue
        r = stats(z, mean_db)
        print(f"{split:6s} {r['maps']:6d} {r['floor_share']:15.2%} {r['maps_any']:14.1%} {r['maps_10']:15.1%} {r['se_share']:30.1%}")


if __name__ == "__main__":
    main()