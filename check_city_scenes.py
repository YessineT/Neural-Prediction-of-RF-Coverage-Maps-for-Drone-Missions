"""
Checks the scenes made by generate_city_scenes.py and the dataset made by make_dataset.py.

Scene checks (no Sionna needed)
    1. all files of every scene exist
    2. height maps are valid (shape, finite, surface >= terrain, plausible heights)
    3. building meshes are inside the area and have outward pointing faces
    4. stations are inside the area, away from the edge, away from buildings, at 1 to 10 m
    5. index.json is consistent (unique names, splits, families, folders)
    6. generator is reproducible: regenerating a scene with the same seed gives the same data
Sionna checks (skipped with --no-sionna)
    7. mesh matches height map: rays cast from above through the Sionna scene
    8. ray traced map follows free space path loss right above the station
Dataset checks (if the dataset/ folder exists)
    9. shapes, no NaN, value range, no scene shared between splits, plausible physics

Also prints statistics per family and saves check_report.png.
The script ends with exit code 0 if everything passed, 1 otherwise.

Usage (from this folder):
    python check_city_scenes.py                 # checks 60 random scenes with Sionna, the rest without
    python check_city_scenes.py --full          # Sionna checks on every scene (slower)
    python check_city_scenes.py --no-sionna     # only the checks that need no Sionna
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_city_scenes as gen

ROOT = Path(__file__).resolve().parent
RESULTS = []


def report(ok, text):
    RESULTS.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + text)


def summarize(bad, total, what):
    if not bad:
        report(True, f"{what} ({total} checked)")
    else:
        report(False, f"{what}: {len(bad)} of {total} failed, first: {bad[:5]}")


def load_npz(path):
    """Read a whole .npz into a dict and close the file (Windows cannot delete open files)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def read_ply(path):
    with open(path, "rb") as f:
        n_v = n_f = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_v = int(line.split()[-1])
            elif line.startswith("element face"):
                n_f = int(line.split()[-1])
            elif line == "end_header":
                break
        v = np.frombuffer(f.read(12 * n_v), dtype="<f4").reshape(n_v, 3).astype(float)
        rec = np.frombuffer(f.read(13 * n_f), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    return v, rec["idx"]


# ---------------------------------------------------------------- scene checks
def check_files(scenes_dir, names):
    required = ["{n}.xml", "meshes/ground.ply", "meshes/buildings.ply",
                "height_map.npz", "sources.npz", "meta.json"]
    bad = [n for n in names if not all((scenes_dir / n / r.format(n=n)).exists() for r in required)]
    summarize(bad, len(names), "all files present")


def check_heightmaps(scenes_dir, names):
    bad = []
    for n in names:
        hm = load_npz(scenes_dir / n / "height_map.npz")
        s, t = hm["surface"], hm["terrain"]
        ok = (s.shape == (gen.N_CELLS, gen.N_CELLS) and t.shape == s.shape
              and np.isfinite(s).all() and np.isfinite(t).all()
              and (s >= t - 1e-6).all()
              and (s - t).max() <= gen.HEIGHT_RANGE[1] + 10
              and np.abs(hm["x"]).max() < gen.HALF and np.abs(hm["y"]).max() < gen.HALF)
        if not ok:
            bad.append(n)
    summarize(bad, len(names), "height maps valid (shape, finite, surface >= terrain, heights)")


def check_not_empty(scenes_dir, index):
    bad = []
    for e in index:
        m = json.loads((scenes_dir / e["name"] / "meta.json").read_text())
        if m["num_buildings"] < gen.MIN_BUILDINGS[e["family"]] or m["built_fraction"] <= 0:
            bad.append(e["name"])
    summarize(bad, len(index), "no empty scenes (minimum number of buildings per family)")


def check_meshes(scenes_dir, names):
    bad_orient, bad_area = [], []
    for n in names:
        v, f = read_ply(scenes_dir / n / "meshes" / "buildings.ply")
        vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
        if vol <= 0:
            bad_orient.append(n)
        if np.abs(v[:, :2]).max() > gen.HALF - 1 + 1e-3:
            bad_area.append(n)
    summarize(bad_orient, len(names), "building faces point outward (positive volume)")
    summarize(bad_area, len(names), "buildings inside the area (1 m from the edge)")


def check_stations(scenes_dir, names):
    bad, counts = [], []
    for n in names:
        src = load_npz(scenes_dir / n / "sources.npz")
        hm = load_npz(scenes_dir / n / "height_map.npz")
        p = src["positions"]
        counts.append(len(p))
        ok = len(p) >= 8
        ok &= np.abs(p[:, :2]).max() <= gen.HALF - gen.EDGE_MARGIN + 1e-6
        agl = p[:, 2] - src["ground_z"]
        ok &= agl.min() >= gen.STATION_HEIGHT_AGL[0] - 1e-4 and agl.max() <= gen.STATION_HEIGHT_AGL[1] + 1e-4
        built = (hm["surface"] - hm["terrain"]) > 0.5
        if built.any():
            XC, YC = np.meshgrid(hm["x"], hm["y"])
            bx, by = XC[built], YC[built]
            dmin = np.hypot(p[:, 0, None] - bx[None], p[:, 1, None] - by[None]).min()
            ok &= dmin >= gen.BUILDING_MARGIN - 1e-6
        if not ok:
            bad.append(n)
    summarize(bad, len(names), f"stations valid (inside, >= {gen.BUILDING_MARGIN:.0f} m from buildings, "
                               f"1 to 10 m high, at least 8; min {min(counts)} per scene)")


def check_index(scenes_dir, index):
    names = [e["name"] for e in index]
    folders = sorted(p.name for p in scenes_dir.iterdir() if p.is_dir())
    report(len(set(names)) == len(names), "scene names are unique")
    report(sorted(names) == folders, "index.json matches the scene folders")
    good_split = all((e["split"] == "ood") == (e["family"] in gen.OOD_FAMILIES) for e in index)
    report(good_split, "unseen layout families are exactly the 'ood' split")
    for fam in gen.ID_FAMILIES:
        sp = [e["split"] for e in index if e["family"] == fam]
        report(set(sp) == {"train", "val", "test"}, f"family {fam} is split into train, val and test "
               f"({sp.count('train')}/{sp.count('val')}/{sp.count('test')})")


def check_reproducible(scenes_dir, index, n_check, seed):
    plan = gen.ID_FAMILIES + gen.OOD_FAMILIES
    rng = np.random.default_rng(1)
    pick = rng.choice(len(index), min(n_check, len(index)), replace=False)
    bad = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        for k in pick:
            e = index[k]
            i = int(e["name"].split("_")[-1])
            r = np.random.default_rng([seed, plan.index(e["family"]), i])
            gen.build_scene(tmp, e["name"], e["family"], r)
            a = load_npz(tmp / e["name"] / "height_map.npz")
            b = load_npz(scenes_dir / e["name"] / "height_map.npz")
            sa = load_npz(tmp / e["name"] / "sources.npz")["positions"]
            sb = load_npz(scenes_dir / e["name"] / "sources.npz")["positions"]
            if not (np.array_equal(a["surface"], b["surface"]) and np.array_equal(sa, sb)):
                bad.append(e["name"])
    summarize(bad, len(pick), "regenerating with the same seed gives identical scenes")


# ---------------------------------------------------------------- Sionna checks
def check_mesh_vs_heightmap(scenes_dir, names):
    import mitsuba as mi
    from sionna.rt import load_scene
    bad, worst = [], 0.0
    for n in names:
        scene = load_scene(str(scenes_dir / n / f"{n}.xml"))
        hm = load_npz(scenes_dir / n / "height_map.npz")
        XC, YC = np.meshgrid(hm["x"], hm["y"])
        top = 500.0
        ray = mi.Ray3f(mi.Point3f(XC.ravel().astype("f4"), YC.ravel().astype("f4"),
                                  np.full(XC.size, top, "f4")), mi.Vector3f(0, 0, -1))
        z = top - np.array(scene.mi_scene.ray_intersect(ray).t)
        err = np.abs(z.reshape(XC.shape) - hm["surface"]).max()
        worst = max(worst, err)
        if err > 0.05:
            bad.append(n)
    summarize(bad, len(names), f"Sionna mesh matches height map (worst error {worst:.3f} m)")


def check_free_space(scenes_dir, names):
    """Right above the station the map must follow free space path loss (line of sight)."""
    import drjit as dr
    from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver
    solver = RadioMapSolver()
    lam = 299792458.0 / 2.4e9
    bad, diffs = [], []
    for n in names:
        scene = load_scene(str(scenes_dir / n / f"{n}.xml"))
        scene.frequency = 2.4e9
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                     horizontal_spacing=0.5, pattern="iso", polarization="V")
        scene.rx_array = scene.tx_array
        s = load_npz(scenes_dir / n / "sources.npz")["positions"][0]
        scene.add(Transmitter(name="gs", position=[float(c) for c in s]))
        alt = 30.0
        rm = solver(scene, center=[0.0, 0.0, alt], orientation=[0.0, 0.0, 0.0], size=[gen.AREA_SIDE] * 2,
                    cell_size=[1.0, 1.0], samples_per_tx=10 ** 6, max_depth=5, los=True,
                    specular_reflection=True, diffuse_reflection=False, refraction=False, diffraction=False)
        g = np.array(rm.path_gain)[0]
        dr.sync_thread()
        ix, iy = int((s[0] + gen.HALF) // gen.CELL), int((s[1] + gen.HALF) // gen.CELL)
        cx, cy = (ix + 0.5) * gen.CELL - gen.HALF, (iy + 0.5) * gen.CELL - gen.HALF
        d = np.sqrt((cx - s[0]) ** 2 + (cy - s[1]) ** 2 + (alt - s[2]) ** 2)
        fspl = -20 * np.log10(4 * np.pi * d / lam)
        diff = 10 * np.log10(max(g[iy, ix], 1e-15)) - fspl
        diffs.append(diff)
        if not -3 <= diff <= 6:
            bad.append(n)
    summarize(bad, len(names), f"ray traced gain above the station follows free space "
                               f"(difference {np.min(diffs):+.1f} to {np.max(diffs):+.1f} dB)")


# ---------------------------------------------------------------- dataset checks
def check_dataset(scenes_dir, ds_dir, index):
    split_of = {e["name"]: e["split"] for e in index}
    lam = 299792458.0 / 2.4e9
    seen, stats = {}, {}
    for split in ["train", "val", "test", "ood"]:
        f = ds_dir / f"{split}.npz"
        if not f.exists():
            continue
        d = load_npz(f)
        n, h = len(d["altitude"]), d["heights"]
        t = d["target_db"]
        ok_shape = (t.shape == (n, gen.N_CELLS, gen.N_CELLS) and h.shape[1:] == t.shape[1:]
                    and d["station_pos"].shape == (n, 3) and d["scene_idx"].max() < len(h))
        report(ok_shape, f"[{split}] shapes consistent ({n} maps, {len(h)} scenes)")
        report(np.isfinite(t).all() and t.min() >= -150.01 and t.max() < -40,
               f"[{split}] no NaN, gains between {t.min():.1f} and {t.max():.1f} dB")
        report(all(split_of.get(s) == split for s in d["scene_names"]),
               f"[{split}] every scene belongs to this split in index.json")
        stale = [s for i, s in enumerate(d["scene_names"])
                 if not np.array_equal(h[i], load_npz(scenes_dir / s / "height_map.npz")["surface"].astype(np.float32))]
        report(not stale, f"[{split}] dataset heights equal the current scene height maps"
               + (f" (scenes changed since the dataset was made: {stale[:3]}, rerun make_dataset.py)"
                  if stale else ""))
        for s in d["scene_names"]:
            seen.setdefault(s, []).append(split)
        report((d["rt_time"] > 0).all(), f"[{split}] ray tracing times saved "
               f"(mean {d['rt_time'].mean():.2f} s per map)")

        # free space check above every station
        xs = d["station_pos"]
        ix = np.clip(((xs[:, 0] + gen.HALF) // gen.CELL).astype(int), 0, gen.N_CELLS - 1)
        iy = np.clip(((xs[:, 1] + gen.HALF) // gen.CELL).astype(int), 0, gen.N_CELLS - 1)
        cx, cy = (ix + 0.5) * gen.CELL - gen.HALF, (iy + 0.5) * gen.CELL - gen.HALF
        dist = np.sqrt((cx - xs[:, 0]) ** 2 + (cy - xs[:, 1]) ** 2 + (d["altitude"] - xs[:, 2]) ** 2)
        diff = t[np.arange(n), iy, ix] - (-20 * np.log10(4 * np.pi * dist / lam))
        frac = np.mean((diff > -3) & (diff < 6))
        report(frac >= 0.99, f"[{split}] gain above the station follows free space in {frac * 100:.1f}% of maps")

        valid = h[d["scene_idx"]] < d["altitude"][:, None, None] - 0.5
        stats[split] = dict(valid=valid.mean(), nosig=((t <= -149.9) & valid).sum() / valid.sum(),
                            nosig_map=((t <= -149.9) & valid).reshape(n, -1).sum(1) / valid.reshape(n, -1).sum(1))
        print(f"      [{split}] valid cells {stats[split]['valid'] * 100:.1f}%, "
              f"no signal in {stats[split]['nosig'] * 100:.1f}% of the valid cells")
    shared = [s for s, v in seen.items() if len(v) > 1]
    report(not shared, "no scene appears in two splits")
    return stats


# ---------------------------------------------------------------- statistics and figure
def statistics(scenes_dir, index, ds_stats):
    fams = gen.ID_FAMILIES + gen.OOD_FAMILIES
    data = {f: dict(built=[], hmax=[], nb=[]) for f in fams}
    for e in index:
        m = json.loads((scenes_dir / e["name"] / "meta.json").read_text())
        data[e["family"]]["built"].append(m["built_fraction"])
        data[e["family"]]["hmax"].append(m["max_height"])
        data[e["family"]]["nb"].append(m["num_buildings"])
    print("\nfamily      scenes  built fraction   max height (m)   buildings")
    for f in fams:
        v = data[f]
        print(f"{f:10s} {len(v['built']):6d}  {np.mean(v['built']):5.2f} "
              f"[{np.min(v['built']):.2f}-{np.max(v['built']):.2f}]   "
              f"{np.mean(v['hmax']):5.1f} [{np.min(v['hmax']):.0f}-{np.max(v['hmax']):.0f}]"
              f"   {np.mean(v['nb']):5.1f} [{np.min(v['nb'])}-{np.max(v['nb'])}]")
    n_pan = 4 if ds_stats else 3
    fig, axes = plt.subplots(1, n_pan, figsize=(4.6 * n_pan, 3.6), constrained_layout=True)
    for ax, key, title in zip(axes[:3], ["built", "hmax", "nb"],
                              ["built fraction of the area", "tallest building (m)", "number of buildings"]):
        for f in fams:
            ax.hist(data[f][key], bins=15, histtype="step", lw=1.8, label=f,
                    ls="-" if f in gen.ID_FAMILIES else "--")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    if ds_stats:
        for split, st in ds_stats.items():
            axes[3].hist(st["nosig_map"], bins=20, range=(0, 1), histtype="step", lw=1.8, label=split)
        axes[3].set_title("share of valid cells without signal, per map")
        axes[3].legend(fontsize=8)
    fig.savefig(ROOT / "check_report.png", dpi=120)
    plt.close(fig)
    print("\nsaved check_report.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=str(ROOT / "scenes_city"))
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--full", action="store_true", help="Sionna checks on every scene")
    ap.add_argument("--no-sionna", action="store_true")
    ap.add_argument("--n-sionna", type=int, default=60, help="scenes for the mesh check")
    ap.add_argument("--n-physics", type=int, default=8, help="scenes for the free space check")
    ap.add_argument("--n-repro", type=int, default=10)
    args = ap.parse_args()

    scenes_dir, ds_dir = Path(args.scenes), Path(args.dataset)
    if not (scenes_dir / "index.json").exists():
        sys.exit(f"{scenes_dir} has no index.json, run generate_city_scenes.py first")
    index = json.loads((scenes_dir / "index.json").read_text())
    config = json.loads((scenes_dir / "config.json").read_text())
    names = [e["name"] for e in index]
    print(f"{len(names)} scenes in {scenes_dir}\n")

    check_files(scenes_dir, names)
    check_heightmaps(scenes_dir, names)
    check_not_empty(scenes_dir, index)
    check_meshes(scenes_dir, names)
    check_stations(scenes_dir, names)
    check_index(scenes_dir, index)
    check_reproducible(scenes_dir, index, args.n_repro, config["SEED"])

    if not args.no_sionna:
        rng = np.random.default_rng(0)
        k = len(names) if args.full else min(args.n_sionna, len(names))
        check_mesh_vs_heightmap(scenes_dir, [names[i] for i in rng.choice(len(names), k, replace=False)])
        per_fam = max(1, args.n_physics // len(gen.ID_FAMILIES + gen.OOD_FAMILIES))
        phys = [e["name"] for fam in gen.ID_FAMILIES + gen.OOD_FAMILIES
                for e in [x for x in index if x["family"] == fam][:per_fam]]
        check_free_space(scenes_dir, phys)

    ds_stats = {}
    if ds_dir.exists() and any(ds_dir.glob("*.npz")):
        print()
        ds_stats = check_dataset(scenes_dir, ds_dir, index)
    else:
        print("\n(no dataset/ folder yet, dataset checks skipped)")

    statistics(scenes_dir, index, ds_stats)
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail} checks passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()