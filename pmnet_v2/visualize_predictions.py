"""
Pictures to analyse a trained PMNet: predicted maps against ray-traced maps, error maps, coverage maps, scatter plots.

For every split (default: test and ood) it runs the network on ALL maps of the split, then draws
    pred_examples_<split>.png   rows = maps picked by their error (good, typical, poor, worst), so nothing is cherry-picked.
                                columns = city and station | ray tracing | PMNet | error (PMNet - ray tracing)
                                | coverage check at a threshold (where the covered region is right or wrong)
    pred_summary_<split>.png    predicted against true path gain (all valid cells, sampled), the error on cells with
                                signal and on no-signal cells (-150 dB), and the distribution of the error per map
and prints the error on cells WITH signal and on no-signal cells separately.

Works with the old and the new train_pmnet.py (it reads the model options from the checkpoint).
Usage (from the folder that contains train_pmnet.py and your PMNet clone):
    python visualize_predictions.py --ckpt runs/v3/best.pt --pmnet-dir PMNet/network
    python visualize_predictions.py --ckpt runs/v3/best.pt --pmnet-dir PMNet/network --splits test --out figures --thr -90
Data comes from Hugging Face (cached), or from a local folder with --data.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FLOOR_DB = -150.0
HALF = 64.0
EXTENT = [-HALF, HALF, -HALF, HALF]
GAIN_RANGE = (-120.0, -60.0)       # colour scale of the path gain maps
ERR_RANGE = 20.0                   # colour scale of the error maps: +-20 dB
PICKS = [("good (10th percentile)", 0.10), ("typical (median)", 0.50), ("poor (90th percentile)", 0.90), ("worst", 1.0)]


# ---------------------------------------------------------------- analysis (numpy only)
def valid_masks(z):
    return z["heights"][z["scene_idx"]] < z["altitude"][:, None, None] - 0.5


def per_map_rmse(z, preds, valid):
    err = (preds - z["target_db"]) * valid
    return np.sqrt((err ** 2).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1))


def signal_split(z, preds, valid):
    """Error on cells with signal and on no-signal cells (target at the -150 dB floor), over all valid cells."""
    t = z["target_db"]
    sig = valid & (t > FLOOR_DB + 1e-3)
    flo = valid & ~sig
    err = preds - t
    out = {}
    for name, m in (("with signal", sig), ("no signal", flo)):
        e = err[m]
        out[name] = dict(share=float(m.sum() / valid.sum()), rmse=float(np.sqrt((e ** 2).mean())) if e.size else float("nan"),
                         bias=float(e.mean()) if e.size else float("nan"), se=float((e ** 2).sum()))
    tot = out["with signal"]["se"] + out["no signal"]["se"]
    for k in out:
        out[k]["se_share"] = out[k]["se"] / tot
    return out


def sample_cells(z, preds, valid, per_map=150, seed=0):
    rng = np.random.default_rng(seed)
    t_all, p_all = [], []
    for i in range(len(preds)):
        idx = np.flatnonzero(valid[i].ravel())
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, size=min(per_map, len(idx)), replace=False)
        t_all.append(z["target_db"][i].ravel()[idx])
        p_all.append(preds[i].ravel()[idx])
    return np.concatenate(t_all), np.concatenate(p_all)


# ---------------------------------------------------------------- figures
def plot_examples(split, z, preds, valid, rmse, thr, out_path):
    order = np.argsort(rmse)
    picks = [(label, int(order[int(q * (len(order) - 1))])) for label, q in PICKS]
    fig, axes = plt.subplots(len(picks), 5, figsize=(18, 3.6 * len(picks)), constrained_layout=True, squeeze=False)
    cov_cmap = ListedColormap(["#4d4d4d", "#f2f2f2", "#9ecae1", "#e41a1c", "#ff9f1c"])   # code 0..4
    for r, (label, i) in enumerate(picks):
        s = z["scene_idx"][i]
        alt = float(z["altitude"][i])
        v = valid[i]
        t = np.where(v, z["target_db"][i], np.nan)
        p = np.where(v, preds[i], np.nan)
        sx, sy, _ = z["station_pos"][i]
        axes[r, 0].imshow(z["heights"][s], origin="lower", cmap="magma", extent=EXTENT, vmin=0)
        axes[r, 0].plot(sx, sy, "c*", ms=13, mec="white", mew=0.6)
        axes[r, 0].set_title(f"{z['scene_names'][s]}, drone at {alt:.0f} m\n{label}", fontsize=10)
        im_g = axes[r, 1].imshow(t, origin="lower", cmap="viridis", extent=EXTENT, vmin=GAIN_RANGE[0], vmax=GAIN_RANGE[1])
        axes[r, 2].imshow(p, origin="lower", cmap="viridis", extent=EXTENT, vmin=GAIN_RANGE[0], vmax=GAIN_RANGE[1])
        im_e = axes[r, 3].imshow(p - t, origin="lower", cmap="RdBu_r", extent=EXTENT, vmin=-ERR_RANGE, vmax=ERR_RANGE)
        tc, pc = z["target_db"][i] >= thr, preds[i] >= thr
        code = np.where(~v, 0, np.where(tc & pc, 2, np.where(~tc & ~pc, 1, np.where(pc & ~tc, 3, 4))))
        axes[r, 4].imshow(code, origin="lower", cmap=cov_cmap, vmin=0, vmax=4, extent=EXTENT, interpolation="nearest")
        for c in (1, 2, 3, 4):
            axes[r, c].plot(sx, sy, "w*" if c != 4 else "k*", ms=9, mec="black" if c != 4 else "white", mew=0.5)
        if r == 0:
            axes[r, 1].set_title("ray tracing (target)", fontsize=10)
            axes[r, 2].set_title("PMNet prediction", fontsize=10)
            axes[r, 3].set_title("error = PMNet - ray tracing", fontsize=10)
            axes[r, 4].set_title(f"coverage at {thr:g} dB", fontsize=10)
        axes[r, 3].set_xlabel(f"RMSE of this map: {rmse[i]:.1f} dB", fontsize=10)
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.colorbar(im_g, ax=axes[:, 1:3].ravel().tolist(), shrink=0.55, label="path gain [dB]", location="bottom", pad=0.02)
    fig.colorbar(im_e, ax=axes[:, 3].tolist(), shrink=0.55, label="error [dB] (red: PMNet too strong, blue: too weak)",
                 location="bottom", pad=0.02)
    fig.legend(handles=[Patch(color="#9ecae1", label="covered, predicted correctly"), Patch(color="#f2f2f2", label="not covered, predicted correctly"),
                        Patch(color="#e41a1c", label="false alarm (predicted covered, is not)"),
                        Patch(color="#ff9f1c", label="miss (is covered, not predicted)"), Patch(color="#4d4d4d", label="inside a building")],
               loc="outside lower right", ncol=1, fontsize=8)
    fig.suptitle(f"{split}: PMNet against ray tracing (white areas: inside buildings taller than the drone)", fontsize=13)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_summary(split, z, preds, valid, rmse, split_stats, out_path):
    t, p = sample_cells(z, preds, valid)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    hb = axes[0].hexbin(t, p, gridsize=90, extent=(-150, -30, -150, -30), bins="log", cmap="viridis", mincnt=1)
    axes[0].plot([-150, -30], [-150, -30], "r--", lw=1, label="perfect prediction")
    axes[0].set_xlabel("ray tracing path gain [dB]")
    axes[0].set_ylabel("PMNet path gain [dB]")
    axes[0].set_title("predicted against true (sampled valid cells)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=axes[0], label="number of cells (log)")
    e = p - t
    sig = t > FLOOR_DB + 1e-3
    bins = np.linspace(-60, 60, 121)
    axes[1].hist(e[sig], bins=bins, density=True, alpha=0.65, color="tab:blue",
                 label=f"cells with signal ({split_stats['with signal']['share']:.1%}): RMSE {split_stats['with signal']['rmse']:.1f} dB")
    if (~sig).sum() > 0:
        axes[1].hist(e[~sig], bins=bins, density=True, alpha=0.65, color="tab:red",
                     label=f"no-signal cells ({split_stats['no signal']['share']:.1%}): RMSE {split_stats['no signal']['rmse']:.1f} dB")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("error = PMNet - ray tracing [dB]")
    axes[1].set_ylabel("share of cells")
    axes[1].set_title("distribution of the error")
    axes[1].legend(fontsize=8)
    axes[2].hist(rmse, bins=40, color="tab:green")
    med, p90 = np.median(rmse), np.percentile(rmse, 90)
    axes[2].axvline(med, color="k", ls="--", label=f"median {med:.1f} dB")
    axes[2].axvline(p90, color="tab:red", ls="--", label=f"90th percentile {p90:.1f} dB")
    axes[2].set_xlabel("RMSE of a map [dB]")
    axes[2].set_ylabel("number of maps")
    axes[2].set_title(f"error per map ({len(rmse)} maps)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{split}: summary of the prediction errors", fontsize=13)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def analyse_split(split, z, preds, thr, out_dir):
    valid = valid_masks(z)
    rmse = per_map_rmse(z, preds, valid)
    ss = signal_split(z, preds, valid)
    print(f"\n{split}: {len(rmse)} maps, RMSE per map: mean {rmse.mean():.2f} dB, median {np.median(rmse):.2f}, "
          f"90th percentile {np.percentile(rmse, 90):.2f}, worst {rmse.max():.2f}")
    for k, v in ss.items():
        print(f"  {k:11s} {v['share']:6.2%} of the valid cells, RMSE {v['rmse']:.2f} dB, mean error {v['bias']:+.2f} dB, "
              f"{v['se_share']:.1%} of the total squared error")
    plot_examples(split, z, preds, valid, rmse, thr, out_dir / f"pred_examples_{split}.png")
    plot_summary(split, z, preds, valid, rmse, ss, out_dir / f"pred_summary_{split}.png")
    print(f"  figures: {out_dir / f'pred_examples_{split}.png'}, {out_dir / f'pred_summary_{split}.png'}")


# ---------------------------------------------------------------- network (needs torch, imported only here)
def make_model(build_model, pmnet_dir, ck_args, in_channels):
    """Build the model from the options stored in the checkpoint. Works with both versions of train_pmnet.py: the
    older build_model(pmnet_dir, variant, in_channels, bn_momentum) and the newer one that also takes n_blocks."""
    import inspect
    bn = ck_args.get("bn_momentum", 0.1)
    n_blocks = list(ck_args.get("n_blocks", [3, 3, 27, 3]))
    if "n_blocks" in inspect.signature(build_model).parameters:
        return build_model(pmnet_dir, ck_args["variant"], in_channels, bn, n_blocks)
    if n_blocks != [3, 3, 27, 3]:
        raise SystemExit(f"this checkpoint was trained with n_blocks={n_blocks}, but your train_pmnet.py cannot build it: "
                         f"use the newer train_pmnet.py")
    return build_model(pmnet_dir, ck_args["variant"], in_channels, bn)


def run_network(args, split):
    import torch
    from train_pmnet import CoverageMaps, build_model, unit_to_db

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    src = dict(data=args.data, repo=args.hf_repo, cache=args.cache_dir)
    ds = CoverageMaps(split, src, a.get("distance", False))
    model = make_model(build_model, args.pmnet_dir, a, len(ck["channels"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds = np.empty((len(ds), 128, 128), np.float32)
    k = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device)).float().squeeze(1).clamp(0, 1)
            preds[k:k + len(out)] = unit_to_db(out).cpu().numpy()
            k += len(out)
    return ds.z, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint, for example runs/v3/best.pt")
    ap.add_argument("--pmnet-dir", required=True, help="folder of the PMNet clone that contains pmnet_v3.py")
    ap.add_argument("--splits", nargs="+", default=["test", "ood"])
    ap.add_argument("--thr", type=float, default=-100.0, help="coverage threshold in dB for the coverage picture")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--data", default="", help="local folder with the .npz files (default: Hugging Face)")
    ap.add_argument("--hf-repo", default="TYessine/airwise-rf-coverage-maps")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        print(f"running the network on {split} ...")
        z, preds = run_network(args, split)
        analyse_split(split, z, preds, args.thr, out_dir)


if __name__ == "__main__":
    main()