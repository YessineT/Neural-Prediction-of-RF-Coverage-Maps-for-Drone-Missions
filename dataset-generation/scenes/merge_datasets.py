"""
Merges the datasets made by make_dataset.py (one folder per person) into one dataset, and inspects it.

Each input folder has train.npz, val.npz, test.npz, ood.npz and raw/<scene>.npz (make_dataset.py output).
The merged folder has the same files and keys, so loaders written for one dataset work on the merged one, plus
    scene_source   (S,)   which input folder each scene comes from

Before merging it checks
    - the ray tracing settings stored in the raw files are identical in every folder (samples, max_depth,
      diffraction, diffuse, altitudes, stations, ...). Mixed settings would put different physics in the targets.
    - no scene name and no scene hash (height map) appears twice, so nobody's scenes are duplicated.
    - heights have the same shape everywhere and are never above any altitude you did not expect.
Then it prints statistics of the merged data and saves dataset_inspection.png and example_maps.png.

Usage (from dataset-generation/):
    python merge_datasets.py --datasets dataset_yessine dataset_mahdi dataset_me --out dataset_all
    python merge_datasets.py --datasets dataset_all --inspect-only          # just look at one dataset
    python merge_datasets.py --datasets dataset_yessine dataset_mahdi --out dataset_all --ignore-config
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ["train", "val", "test", "ood"]
PER_SCENE = ("heights", "scene_names", "scene_hash")            # arrays with one entry per scene
PER_MAP = ("scene_idx", "station_pos", "altitude", "target_db", "rt_time")   # one entry per map
FLOOR_DB = -150.0


# ---------------------------------------------------------------- reading
def load_split(folder, split):
    f = Path(folder) / f"{split}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def raw_configs(folder):
    """Set of ray tracing configs (json strings) found in the raw files of a folder."""
    cfgs = {}
    for f in sorted((Path(folder) / "raw").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        with np.load(f) as r:
            if "config" in r.files:
                cfgs.setdefault(str(r["config"]), []).append(f.stem)
    return cfgs


def check_configs(folders, ignore):
    """All folders must have used exactly one and the same configuration."""
    all_cfg = {}
    for d in folders:
        c = raw_configs(d)
        if not c:
            print(f"WARNING: no raw files with a stored config in {d}/raw, cannot verify its settings")
        if len(c) > 1:
            print(f"PROBLEM: {d} has maps traced with {len(c)} different configs (retrace with --overwrite):")
            for k, v in c.items():
                print(f"    {len(v)} scenes: {k}")
            if not ignore:
                sys.exit(1)
        for k in c:
            all_cfg.setdefault(k, []).append(str(d))
    if len(all_cfg) > 1:
        print("PROBLEM: the folders were traced with different settings:")
        parsed = {k: json.loads(k) for k in all_cfg}
        keys = sorted({kk for p in parsed.values() for kk in p})
        for key in keys:
            vals = {json.dumps(p.get(key)) for p in parsed.values()}
            if len(vals) > 1:
                for k, p in parsed.items():
                    print(f"    {key:12s} = {json.dumps(p.get(key)):12s} in {', '.join(all_cfg[k])}")
        if not ignore:
            print("Not merging. Retrace the odd folder with the same settings, or use --ignore-config.")
            sys.exit(1)
        print("--ignore-config given: merging anyway")
    return json.loads(next(iter(all_cfg))) if all_cfg else {}


# ---------------------------------------------------------------- merging
def merge(folders, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_names, seen_hash = {}, {}
    info = {}
    for split in SPLITS:
        parts = [(d, load_split(d, split)) for d in folders]
        parts = [(d, p) for d, p in parts if p is not None]
        if not parts:
            continue
        merged = {k: [] for k in PER_SCENE + PER_MAP}
        sources, offset = [], 0
        for d, p in parts:
            for name, h in zip(p["scene_names"], p["scene_hash"]):
                name, h = str(name), str(h)
                if name in seen_names:
                    sys.exit(f"scene name {name} is in both {seen_names[name]} and {d}")
                if h in seen_hash:
                    sys.exit(f"scene {name} ({d}) has the same height map as {seen_hash[h][1]} ({seen_hash[h][0]}): duplicate")
                seen_names[name] = str(d)
                seen_hash[h] = (str(d), name)
            for k in PER_SCENE + PER_MAP:
                merged[k].append(p[k] + offset if k == "scene_idx" else p[k])
            sources += [Path(d).name] * len(p["scene_names"])
            offset += len(p["scene_names"])
        shapes = {tuple(p["heights"].shape[1:]) for _, p in parts}
        if len(shapes) > 1:
            sys.exit(f"{split}: height maps have different shapes {shapes}")
        out = {k: np.concatenate(v) for k, v in merged.items()}
        out["scene_source"] = np.array(sources)
        np.savez_compressed(out_dir / f"{split}.npz", **out)
        info[split] = dict(scenes=len(out["scene_names"]), maps=len(out["altitude"]),
                           per_source={Path(d).name: len(p["scene_names"]) for d, p in parts})
        print(f"{split}: {info[split]['scenes']} scenes, {info[split]['maps']} maps  {info[split]['per_source']}")
    return info


# ---------------------------------------------------------------- inspection
def valid_mask(z):
    return z["heights"][z["scene_idx"]] < z["altitude"][:, None, None] - 0.5


def inspect(folder, plots=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    data = {s: load_split(folder, s) for s in SPLITS}
    data = {s: z for s, z in data.items() if z is not None}
    print(f"\n=== inspection of {folder} ===")
    print(f"{'split':6s} {'scenes':>7s} {'maps':>6s} {'valid cells':>12s} {'no signal':>10s} {'mean dB':>8s} "
          f"{'dB 1%/99%':>14s} {'rt time':>8s}")
    rng = np.random.default_rng(0)
    samples, floor_share, valid_by_alt, mean_by_alt = {}, {}, {}, {}
    for s, z in data.items():
        v = valid_mask(z)
        t = z["target_db"]
        nof = (t <= FLOOR_DB + 1e-3) & v
        floor_share[s] = nof.sum((1, 2)) / np.maximum(v.sum((1, 2)), 1)
        tv = t[v]
        samples[s] = rng.choice(tv, size=min(300_000, len(tv)), replace=False)
        lo, hi = np.percentile(samples[s], [1, 99])
        print(f"{s:6s} {len(z['scene_names']):7d} {len(z['altitude']):6d} {v.mean():12.3f} "
              f"{nof.sum() / v.sum():10.3f} {tv.mean():8.1f} {lo:7.0f}/{hi:<6.0f} {z['rt_time'].mean():7.2f}s")
        for a in np.unique(z["altitude"]):
            m = z["altitude"] == a
            valid_by_alt.setdefault(a, []).append(v[m].mean())
            mean_by_alt.setdefault(a, []).append(t[m][v[m]].mean())
        # sanity
        if not np.isfinite(t).all():
            print(f"  PROBLEM: {s} has NaN or inf in target_db")
        if t.min() < FLOOR_DB - 1e-3:
            print(f"  PROBLEM: {s} has values below the floor {FLOOR_DB} dB")
        if len(np.unique(z["scene_hash"])) != len(z["scene_hash"]):
            print(f"  PROBLEM: {s} has duplicate scenes")
    if "scene_source" in next(iter(data.values())):
        for s, z in data.items():
            src, n = np.unique(z["scene_source"], return_counts=True)
            print(f"  {s}: scenes per source {dict(zip(src.tolist(), n.tolist()))}")
    alts = sorted(valid_by_alt)
    print("by altitude (all splits): " + ", ".join(
        f"{a:.0f} m: valid {np.mean(valid_by_alt[a]):.2f}, mean {np.mean(mean_by_alt[a]):.1f} dB" for a in alts))
    print("An altitude with a low valid share has many cells inside buildings; those cells are not predicted.")

    if not plots:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    for s in data:
        axes[0].hist(samples[s], bins=80, histtype="step", density=True, label=s, lw=1.5)
        axes[1].hist(floor_share[s], bins=20, histtype="step", label=s, lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("path gain on valid cells [dB]")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("share of valid cells with no signal, per map")
    axes[2].bar([f"{a:.0f}" for a in alts], [np.mean(valid_by_alt[a]) for a in alts], color="tab:green")
    axes[2].set_xlabel("altitude [m]")
    axes[2].set_ylabel("share of valid cells")
    fig.savefig(folder / "dataset_inspection.png", dpi=120)
    plt.close(fig)

    # example maps: first scene of the first split, all its maps
    s0 = next(iter(data))
    z = data[s0]
    ids = np.flatnonzero(z["scene_idx"] == 0)[:8]
    n = len(ids)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.3 * (n + 1), 6.4), constrained_layout=True, squeeze=False)
    h0 = z["heights"][0]
    for r in range(2):
        axes[r, 0].imshow(h0, origin="lower", cmap="magma")
        axes[r, 0].set_title(f"{z['scene_names'][0]} surface height" if r == 0 else "stations")
        axes[r, 0].axis("off")
    for k, i in enumerate(ids):
        ax = axes[0, k + 1]
        v = z["heights"][0] < z["altitude"][i] - 0.5
        ax.imshow(np.where(v, z["target_db"][i], np.nan), origin="lower", cmap="viridis", vmin=-120, vmax=-60)
        ax.set_title(f"{z['altitude'][i]:.0f} m", fontsize=9)
        ax.axis("off")
        axes[1, k + 1].imshow(v, origin="lower", cmap="gray")
        axes[1, k + 1].set_title("valid cells", fontsize=9)
        axes[1, k + 1].axis("off")
    fig.savefig(folder / "example_maps.png", dpi=110)
    plt.close(fig)
    print(f"figures: {folder / 'dataset_inspection.png'}, {folder / 'example_maps.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset folders (output of make_dataset.py)")
    ap.add_argument("--out", help="merged folder (omit with --inspect-only)")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--ignore-config", action="store_true", help="merge even if the ray tracing settings differ")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    if a.inspect_only:
        for d in a.datasets:
            inspect(d, plots=not a.no_plots)
        return
    if not a.out:
        sys.exit("give --out, or --inspect-only")
    folders = [Path(d) for d in a.datasets]
    for d in folders:
        if not d.is_dir():
            sys.exit(f"folder not found: {d}")
    out = Path(a.out)
    if out.resolve() in [d.resolve() for d in folders]:
        sys.exit("--out must be a new folder")
    cfg = check_configs(folders, a.ignore_config)
    info = merge(folders, out)
    (out / "merge_info.json").write_text(json.dumps(dict(config=cfg, sources=[str(d) for d in folders],
                                                         splits=info), indent=1))
    inspect(out, plots=not a.no_plots)


if __name__ == "__main__":
    main()