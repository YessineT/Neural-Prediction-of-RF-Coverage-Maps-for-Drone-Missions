"""
Simple baselines for the AIRWISE coverage maps, evaluated with the same metrics as train_pmnet.py.
A neural network is only useful if it clearly beats these. Needs only numpy (and huggingface_hub to download the data).

Baselines (everything is fitted on the TRAIN split only, on valid cells):
    mean            the same path gain everywhere: the mean of the training maps
    mean_altitude   the mean path gain of the training maps of the same drone altitude
    free_space      free space path loss from the station, 3D distance, 2.4 GHz, no buildings (nothing is fitted)
    free_space_off  free space plus one constant offset (dB) fitted on train
    log_distance    a + b * log10(distance), a and b fitted on train (b = -20 would be free space)
Predictions are clipped to [-150, 0] dB like the network output.

Metrics (valid cells only, in dB), as in train_pmnet.py:
    RMSE per map (mean over maps, median, 90th percentile), pooled RMSE, MAE,
    coverage F1 and IoU at the thresholds, RMSE per altitude and per family.

Usage:
    python baselines.py                          # downloads the splits from Hugging Face (cached)
    python baselines.py --data hf_data           # local folder with train.npz, val.npz, test.npz, ood.npz
Writes baselines.json (or --out).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HF_REPO = "TYessine/airwise-rf-coverage-maps"
FLOOR_DB, CEIL_DB = -150.0, 0.0
HALF, CELL, N = 64.0, 1.0, 128
FREQ = 2.4e9
WAVELENGTH = 299792458.0 / FREQ
_xc = -HALF + CELL * (np.arange(N) + 0.5)
_X, _Y = np.meshgrid(_xc, _xc)                       # [row = y, col = x], metres
BASELINES = ["mean", "mean_altitude", "free_space", "free_space_off", "log_distance"]


# ---------------------------------------------------------------- data
def get_split_file(split, args):
    if args.data:
        f = Path(args.data) / f"{split}.npz"
        return f if f.exists() else None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    print(f"getting {split}.npz from Hugging Face ({args.hf_repo}); downloaded once, then cached")
    return Path(hf_hub_download(args.hf_repo, f"{split}.npz", repo_type="dataset", cache_dir=args.cache_dir or None))


def load_split(split, args):
    f = get_split_file(split, args)
    if f is None:
        return None
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def map_geometry(z, i):
    """Valid mask, target (dB) and 3D distance (m, at least 1) from the station to every cell, for map i."""
    s = z["scene_idx"][i]
    alt = float(z["altitude"][i])
    valid = z["heights"][s] < alt - 0.5
    sx, sy, sz = (float(v) for v in z["station_pos"][i])
    d = np.sqrt((_X - sx) ** 2 + (_Y - sy) ** 2 + (alt - sz) ** 2)
    return valid, z["target_db"][i], np.maximum(d, 1.0), alt


def free_space_db(d):
    return 20.0 * np.log10(WAVELENGTH / (4.0 * np.pi * d))


# ---------------------------------------------------------------- fitting on train
def fit(z):
    """Streaming sums over all valid cells of the training maps (exact least squares, no subsampling)."""
    n = sy = 0.0
    s_alt = {}
    s_off = 0.0
    sx = sxx = sxy = 0.0
    for i in range(len(z["altitude"])):
        valid, y, d, alt = map_geometry(z, i)
        yv = y[valid].astype(np.float64)
        dv = d[valid]
        x = np.log10(dv)
        n += len(yv)
        sy += yv.sum()
        a = s_alt.setdefault(int(alt), [0.0, 0.0])
        a[0] += len(yv)
        a[1] += yv.sum()
        s_off += (yv - free_space_db(dv)).sum()
        sx += x.sum()
        sxx += (x * x).sum()
        sxy += (x * yv).sum()
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return dict(mean=sy / n, mean_altitude={k: v[1] / v[0] for k, v in s_alt.items()}, offset=s_off / n, a=a, b=b)


def predict(name, par, d, alt):
    if name == "mean":
        p = np.full((N, N), par["mean"])
    elif name == "mean_altitude":
        p = np.full((N, N), par["mean_altitude"].get(int(alt), par["mean"]))
    elif name == "free_space":
        p = free_space_db(d)
    elif name == "free_space_off":
        p = free_space_db(d) + par["offset"]
    else:
        p = par["a"] + par["b"] * np.log10(d)
    return np.clip(p, FLOOR_DB, CEIL_DB)


# ---------------------------------------------------------------- evaluation (same definitions as train_pmnet.py)
def evaluate(z, par, thresholds):
    recs = {b: dict(se=[], ae=[], n=[], tp=[], fp=[], fn=[]) for b in BASELINES}
    alts, fams = [], []
    thr = np.array(thresholds).reshape(-1, 1, 1)
    for i in range(len(z["altitude"])):
        valid, y, d, alt = map_geometry(z, i)
        alts.append(alt)
        fams.append(str(z["scene_names"][z["scene_idx"][i]]).split("_")[0])
        tc = (y[None] >= thr) & valid[None]
        for b in BASELINES:
            p = predict(b, par, d, alt)
            err = (p - y) * valid
            r = recs[b]
            r["se"].append((err ** 2).sum())
            r["ae"].append(np.abs(err).sum())
            r["n"].append(valid.sum())
            pc = (p[None] >= thr) & valid[None]
            r["tp"].append((pc & tc).sum((1, 2)))
            r["fp"].append((pc & ~tc).sum((1, 2)))
            r["fn"].append((~pc & tc).sum((1, 2)))
    alts, fams = np.array(alts), np.array(fams)
    return {b: summarize({k: np.array(v) for k, v in r.items()}, alts, fams, thresholds) for b, r in recs.items()}


def summarize(rec, alts, fams, thresholds):
    n = np.maximum(rec["n"], 1)
    rmse_map = np.sqrt(rec["se"] / n)
    s = dict(maps=int(len(n)), rmse_db=float(rmse_map.mean()), rmse_pooled_db=float(np.sqrt(rec["se"].sum() / n.sum())),
             mae_db=float(rec["ae"].sum() / n.sum()), rmse_median_db=float(np.median(rmse_map)),
             rmse_p90_db=float(np.percentile(rmse_map, 90)))
    for i, t in enumerate(thresholds):
        tp, fp, fn = rec["tp"][:, i].sum(), rec["fp"][:, i].sum(), rec["fn"][:, i].sum()
        s[f"f1@{t:g}"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        s[f"iou@{t:g}"] = float(tp / max(tp + fp + fn, 1))
    s["by_altitude"] = {int(a): float(np.sqrt(rec["se"][alts == a].sum() / max(rec["n"][alts == a].sum(), 1)))
                        for a in np.unique(alts)}
    s["by_family"] = {f: float(np.sqrt(rec["se"][fams == f].sum() / max(rec["n"][fams == f].sum(), 1)))
                      for f in np.unique(fams)}
    return s


def print_table(split, res, thresholds):
    print(f"\n{split}: {next(iter(res.values()))['maps']} maps (errors in dB over valid cells)")
    head = f"  {'baseline':15s} {'RMSE':>6s} {'median':>7s} {'p90':>6s} {'MAE':>6s}  " + "  ".join(f"F1@{t:g}" for t in thresholds)
    print(head)
    for b, s in res.items():
        cov = "  ".join(f"{s[f'f1@{t:g}']:>{len(f'F1@{t:g}')}.3f}" for t in thresholds)
        print(f"  {b:15s} {s['rmse_db']:6.2f} {s['rmse_median_db']:7.2f} {s['rmse_p90_db']:6.2f} {s['mae_db']:6.2f}  {cov}")
    best = min(res, key=lambda b: res[b]["rmse_db"])
    print(f"  best baseline: {best}; its RMSE by altitude: "
          + ", ".join(f"{a} m: {v:.2f}" for a, v in res[best]["by_altitude"].items())
          + " | by family: " + ", ".join(f"{f}: {v:.2f}" for f, v in res[best]["by_family"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="", help="local folder with the .npz files (default: Hugging Face)")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[-110.0, -100.0, -90.0])
    ap.add_argument("--splits", nargs="+", default=["val", "test", "ood"])
    ap.add_argument("--out", default="baselines.json")
    args = ap.parse_args()

    train = load_split("train", args)
    if train is None:
        sys.exit("train.npz not found")
    print(f"fitting on {len(train['altitude'])} training maps")
    par = fit(train)
    del train
    print(f"  mean path gain {par['mean']:.1f} dB; by altitude " + ", ".join(f"{a} m: {m:.1f}" for a, m in sorted(par["mean_altitude"].items())))
    print(f"  free space offset {par['offset']:+.1f} dB (negative: buildings cost signal)")
    print(f"  log-distance fit: gain = {par['a']:.1f} + {par['b']:.1f} * log10(d), path loss exponent {-par['b'] / 10:.2f} (free space: 2.00)")

    results = {}
    for split in args.splits:
        z = load_split(split, args)
        if z is None:
            continue
        results[split] = evaluate(z, par, args.thresholds)
        print_table(split, results[split], args.thresholds)
    par_json = dict(par, mean_altitude={str(k): v for k, v in par["mean_altitude"].items()})
    Path(args.out).write_text(json.dumps(dict(fitted=par_json, results=results), indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()