"""
Qualitative figures: ground truth vs the 4 trained models, plus error maps.

Put this file next to train_pmnet.py and pmnet_slim.py, then run for example:
    python plot_predictions.py --split ood --pmnet-dir ./models
    python plot_predictions.py --split test --altitudes 15 60 --pmnet-dir ./models --data ./npz_folder

Writes <out>/pred_<split>.png (predictions) and <out>/err_<split>.png (error maps, pred - truth in dB).
Leave --pmnet-dir empty if the models were trained with pmnet_slim.py (--width).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_pmnet as T

RUNS = {"MSE": "runs/cmp_mse", "L1": "runs/cmp_l1", "Huber": "runs/cmp_huber", "Huber + Edge": "runs/cmp_huber_edge"}


def load(path, pmnet_dir, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = T.build_model(pmnet_dir, a["variant"], len(ck["channels"]), a["bn_momentum"], a["n_blocks"], a.get("width")).to(device)
    m.load_state_dict(ck["model"])
    return m.eval(), a.get("distance", False)


def pick(ds, altitudes):
    """One map per (scene family, altitude), the first one found."""
    z, seen, idx = ds.z, set(), []
    for i in range(len(ds)):
        fam = str(z["scene_names"][z["scene_idx"][i]]).split("_")[0]
        alt = int(z["altitude"][i])
        if alt in altitudes and (fam, alt) not in seen:
            seen.add((fam, alt))
            idx.append(i)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="ood")
    ap.add_argument("--data", default="")
    ap.add_argument("--pmnet-dir", default="")
    ap.add_argument("--altitudes", type=int, nargs="+", default=[15, 60])
    ap.add_argument("--runs-root", default="", help="prefix put before each run folder")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--vmin", type=float, default=-140)
    ap.add_argument("--vmax", type=float, default=-60)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = dict(data=a.data, repo=T.HF_REPO, cache="")
    models = {n: load(Path(a.runs_root) / r / "best.pt", a.pmnet_dir, dev) for n, r in RUNS.items()}
    ds = T.CoverageMaps(a.split, src, any(d for _, d in models.values()))
    idx = pick(ds, set(a.altitudes))
    print(f"{len(idx)} maps:", idx)

    rows = []
    for i in idx:
        it = ds[i]
        x = torch.from_numpy(it["x"]).unsqueeze(0).to(dev)
        truth, valid = it["y"], it["valid"]
        preds = {}
        with torch.no_grad():
            for n, (m, _) in models.items():
                preds[n] = T.unit_to_db(m(x).float().squeeze().clamp(0, 1)).cpu().numpy()
        rows.append((it, truth, valid, preds))

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#bfbfbf")                       # buildings above the drone plane
    names = list(RUNS)
    Path(a.out).mkdir(exist_ok=True)

    def title(it):
        return f"{str(it['name']).split('_')[0]}, {int(it['altitude'])} m"

    # figure 1: truth and predictions
    fig, ax = plt.subplots(len(rows), 1 + len(names), figsize=(3 * (1 + len(names)), 3 * len(rows)), squeeze=False)
    for r, (it, truth, valid, preds) in enumerate(rows):
        panels = [("Ground truth", truth)] + [(n, preds[n]) for n in names]
        for c, (n, img) in enumerate(panels):
            im = ax[r, c].imshow(np.ma.masked_where(~valid, img), cmap=cmap, vmin=a.vmin, vmax=a.vmax, origin="lower")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(n)
            if c == 0:
                ax[r, c].set_ylabel(title(it))
    fig.colorbar(im, ax=ax, shrink=0.6, label="path gain (dB)")
    fig.savefig(Path(a.out) / f"pred_{a.split}.png", dpi=150, bbox_inches="tight")

    # figure 2: error maps
    fig, ax = plt.subplots(len(rows), len(names), figsize=(3 * len(names), 3 * len(rows)), squeeze=False)
    ecm = plt.get_cmap("RdBu_r").copy(); ecm.set_bad("#bfbfbf")
    for r, (it, truth, valid, preds) in enumerate(rows):
        for c, n in enumerate(names):
            err = preds[n] - truth
            rmse = np.sqrt((err[valid] ** 2).mean())
            im = ax[r, c].imshow(np.ma.masked_where(~valid, err), cmap=ecm, vmin=-30, vmax=30, origin="lower")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{n}: {rmse:.1f} dB" if r else f"{n}\n{rmse:.1f} dB", fontsize=10)
            if c == 0:
                ax[r, c].set_ylabel(title(it))
    fig.colorbar(im, ax=ax, shrink=0.6, label="prediction - truth (dB): red = too strong, blue = too weak")
    fig.savefig(Path(a.out) / f"err_{a.split}.png", dpi=150, bbox_inches="tight")
    print("saved to", a.out)


if __name__ == "__main__":
    main()