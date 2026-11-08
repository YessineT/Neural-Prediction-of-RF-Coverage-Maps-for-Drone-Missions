"""
Ray traces coverage maps for the scenes made by generate_city_scenes.py.

For every scene it picks a few ground stations and, for each station, computes one
radio map per drone altitude with Sionna RT (RadioMapSolver). The time of every map
is saved too, it is the baseline for the speed comparison with the neural network.

Output (default folder: dataset/)
    raw/<scene>.npz    one file per scene, written as soon as the scene is done,
                       so an interrupted run can be restarted (finished scenes are skipped).
                       Each file also stores the hash of the scene it was traced on and the
                       settings used, so it is never mixed with a changed scene or other settings.
    train.npz, val.npz, test.npz, ood.npz    merged files used for training:
        heights      (S, 128, 128)  surface height of every scene (ground + buildings), m
        scene_names  (S,)
        scene_hash   (S,)           hash of each scene's height map when it was traced
        scene_idx    (N,)           which scene each sample belongs to
        station_pos  (N, 3)         ground station x, y, z in m
        altitude     (N,)           height of the map plane (drone altitude), m
        target_db    (N, 128, 128)  path gain in dB, floored at -150 dB (= no signal)
        rt_time      (N,)           ray tracing time of that map, s (a warm-up map is run first,
                                    so the first timed map does not include kernel compilation)

A map cell is "valid" when the drone could be there, i.e. the plane is above the
surface:   valid = heights[scene_idx] < altitude - 0.5
Cells inside buildings taller than the altitude are not valid and should be left out of
the loss and of the evaluation.

Ray tracing settings (defaults in common.RT_DEFAULTS, chosen to be close to real propagation and converged):
line of sight, specular and diffuse reflections, diffraction, max_depth 8, 1e8 rays per map, no refraction.

Usage (from this folder, after generate_city_scenes.py and check_city_scenes.py --full --stations dataset):
    python make_dataset.py --max-scenes 2           # quick test on 2 scenes per split
    python make_dataset.py                          # everything (~30-40 min on an RTX 4070 Laptop, ~0.3 s per map)
    python make_dataset.py --samples 2000000 --max-depth 5 --no-diffraction --no-diffuse   # fast, less exact
    python make_dataset.py --splits train val       # only some splits
    python make_dataset.py --merge-only             # just rebuild the merged files
    python make_dataset.py --overwrite              # retrace everything (needed after changing settings)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import drjit as dr
from sionna.rt import Transmitter, RadioMapSolver

import generate_city_scenes as gen
from common import DATASET_DEFAULTS, RT_DEFAULTS, load_radio_scene, pick_stations, radio_map, scene_hash

FREQUENCY = RT_DEFAULTS["frequency"]   # Hz
FLOOR_DB = -150.0            # path gain used where no ray arrives
ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = ["stations", "altitudes", "samples", "max_depth", "side", "diffraction", "diffuse", "seed"]


def config_string(args):
    cfg = {k: getattr(args, k) for k in CONFIG_KEYS}
    cfg.update(frequency=FREQUENCY, floor_db=FLOOR_DB, cell=gen.CELL)
    return json.dumps(cfg, sort_keys=True)


def dataset_stations(name, n_src, args, bad=()):
    return pick_stations(name, n_src, args.stations, args.seed, bad)


def trace(solver, scene, alt, args):
    """Path gain map of the scene's one transmitter at height alt. np.array forces the computation and
    sync_thread waits for the GPU, so timing this call measures the whole ray tracing of one map."""
    rm = radio_map(solver, scene, alt, args.side, gen.CELL, args.samples, args.max_depth, args.diffraction,
                   args.diffuse)
    gain = np.array(rm.path_gain)[0]
    dr.sync_thread()
    return gain


def trace_scene(scene_dir, args, solver, bad=(), warm_up=False):
    """Every map of one scene, one timed solver call per map (rt_time is the per-map baseline for the
    neural network). warm_up: trace one untimed map first, so kernel compilation is not in rt_time."""
    name = scene_dir.name
    scene = load_radio_scene(scene_dir / f"{name}.xml", FREQUENCY)
    with np.load(scene_dir / "sources.npz") as f:
        src = f["positions"]
    picks = dataset_stations(name, len(src), args, bad)

    tx = Transmitter(name="gs", position=[float(c) for c in src[picks[0]]])
    scene.add(tx)
    if warm_up:
        trace(solver, scene, args.altitudes[0], args)

    targets, alts, pos, times = [], [], [], []
    for s in picks:
        tx.position = [float(c) for c in src[s]]
        for alt in args.altitudes:
            t0 = time.perf_counter()
            gain = trace(solver, scene, alt, args)
            times.append(time.perf_counter() - t0)
            if gain.shape != (gen.N_CELLS, gen.N_CELLS):
                raise RuntimeError(f"{name}: map is {gain.shape}, expected {(gen.N_CELLS, gen.N_CELLS)}; "
                                   f"--side must equal the scene size and cell size must match")
            if not (np.isfinite(gain).all() and (gain >= 0).all()):
                raise RuntimeError(f"{name}: ray traced gain has NaN, inf or negative values")
            db = 10 * np.log10(np.maximum(gain, 10 ** (FLOOR_DB / 10)))
            targets.append(db.astype(np.float32))
            alts.append(alt)
            pos.append(src[s])
    return dict(target_db=np.stack(targets), altitude=np.array(alts, np.float32),
                station_pos=np.array(pos, np.float32), rt_time=np.array(times, np.float32),
                scene_hash=np.array(scene_hash(scene_dir)), config=np.array(config_string(args)))


def raw_is_current(f, scenes_dir, name, cfg, args, bad=()):
    """A finished raw file is reusable only if the scene, the settings and the chosen stations are unchanged
    (the choice changes when the checker marks one of the traced stations as bad)."""
    try:
        with np.load(f) as r, np.load(scenes_dir / name / "sources.npz") as s:
            src = s["positions"]
            want = np.unique(src[dataset_stations(name, len(src), args, bad)].astype(np.float32), axis=0)
            return (str(r["scene_hash"]) == scene_hash(scenes_dir / name) and str(r["config"]) == cfg
                    and np.array_equal(np.unique(r["station_pos"], axis=0), want))
    except Exception:
        return False


def load_check_results(scenes_dir):
    """Per scene results written by check_city_scenes.py, or None if the checker was never run."""
    f = Path(scenes_dir) / "check_results.json"
    return json.loads(f.read_text()).get("scenes", {}) if f.exists() else None


def merge(scenes_dir, out_dir, splits):
    index = json.loads((scenes_dir / "index.json").read_text())
    for split in splits:
        in_split = [e["name"] for e in index if e["split"] == split]
        names, raws, stale, missing = [], [], [], []
        for name in in_split:
            f = out_dir / "raw" / f"{name}.npz"
            if not f.exists():
                missing.append(name)
                continue
            try:
                with np.load(f) as r:
                    raw = {k: r[k] for k in ("scene_hash", "target_db", "altitude", "station_pos", "rt_time")}
                ok = str(raw["scene_hash"]) == scene_hash(scenes_dir / name)
            except Exception:
                ok = False
            if ok:
                names.append(name)
                raws.append(raw)
            else:
                stale.append(name)
        if stale:
            print(f"{split}: WARNING {len(stale)} scenes changed since they were traced (or the raw file is "
                  f"unreadable) and are left out, retrace with --overwrite: {stale[:3]}")
        if missing:
            print(f"{split}: WARNING {len(missing)} of {len(in_split)} scenes have no raw file yet")
        if not names:
            print(f"{split}: nothing to merge")
            continue
        heights, hashes, idx, tgt, alt, pos, rt = [], [], [], [], [], [], []
        for i, (name, r) in enumerate(zip(names, raws)):
            with np.load(scenes_dir / name / "height_map.npz") as hm:
                heights.append(hm["surface"].astype(np.float32))
            idx.append(np.full(len(r["altitude"]), i, np.int32))
            hashes.append(str(r["scene_hash"]))
            tgt.append(r["target_db"])
            alt.append(r["altitude"])
            pos.append(r["station_pos"])
            rt.append(r["rt_time"])
        np.savez_compressed(out_dir / f"{split}.npz", heights=np.stack(heights),
                            scene_names=np.array(names), scene_hash=np.array(hashes),
                            scene_idx=np.concatenate(idx),
                            station_pos=np.concatenate(pos), altitude=np.concatenate(alt),
                            target_db=np.concatenate(tgt), rt_time=np.concatenate(rt))
        n_maps = sum(len(a) for a in alt)
        print(f"{split}: {len(names)} scenes, {n_maps} maps -> {out_dir / (split + '.npz')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=str(ROOT / "scenes_city"))
    ap.add_argument("--out", default=str(ROOT / "dataset"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "ood"])
    ap.add_argument("--stations", type=int, default=DATASET_DEFAULTS["stations"], help="ground stations per scene")
    ap.add_argument("--altitudes", type=float, nargs="+", default=list(gen.DRONE_ALTITUDES),
                    help="drone altitudes in m; the scenes keep roofs and stations clear of the generator's "
                         "DRONE_ALTITUDES, so other values lose that guarantee")
    ap.add_argument("--samples", type=int, default=RT_DEFAULTS["samples"], help="rays per map")
    ap.add_argument("--max-depth", type=int, default=RT_DEFAULTS["max_depth"], help="maximum number of bounces")
    ap.add_argument("--side", type=float, default=float(gen.AREA_SIDE),
                    help="map size in m, must match the scene")
    ap.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=RT_DEFAULTS["diffraction"],
                    help="diffraction over roofs and around corners (default on)")
    ap.add_argument("--diffuse", action=argparse.BooleanOptionalAction, default=RT_DEFAULTS["diffuse"],
                    help="diffuse reflections from rough surfaces, strength set by the scenes' scattering_coefficient "
                         "(default on)")
    ap.add_argument("--max-scenes", type=int, default=0, help="only this many scenes per split (test)")
    ap.add_argument("--seed", type=int, default=DATASET_DEFAULTS["seed"], help="seed of the station choice")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--ignore-checks", action="store_true",
                    help="also trace scenes that FAILED in check_results.json (written by check_city_scenes.py)")
    args = ap.parse_args()

    scenes_dir, out_dir = Path(args.scenes), Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        if abs(args.side - gen.AREA_SIDE) > 1e-6:
            raise SystemExit(f"--side {args.side} does not match the scene size {gen.AREA_SIDE}")
        for tmp in (out_dir / "raw").glob("*.tmp.npz"):   # leftovers of an interrupted write
            tmp.unlink()
        cfg = config_string(args)
        index = json.loads((scenes_dir / "index.json").read_text())
        scene_cfg = json.loads((scenes_dir / "config.json").read_text())
        if args.diffuse and not scene_cfg.get("SCATTERING_COEFFICIENT"):
            print("WARNING: the scenes have no scattering_coefficient, so diffuse reflections have no effect; "
                  "regenerate them with generate_city_scenes.py")
        scene_alts = scene_cfg.get("DRONE_ALTITUDES")
        if scene_alts is None or not set(args.altitudes) <= set(scene_alts):
            print(f"WARNING: altitudes {args.altitudes} are not all among the altitudes the scenes were made for "
                  f"({scene_alts}); roofs or stations may lie on or near a radio map plane")
        checks = None if args.ignore_checks else load_check_results(scenes_dir)
        if checks is None and not args.ignore_checks:
            print("WARNING: no check_results.json in the scenes folder, run check_city_scenes.py first "
                  "(the scenes are traced without being checked)")
        skipped, unchecked, n_bad_st, warmed = [], [], 0, False
        solver = RadioMapSolver()
        (out_dir / "settings.json").write_text(json.dumps(
            {**vars(args), "frequency": FREQUENCY, "floor_db": FLOOR_DB, "cell": gen.CELL}, indent=2))
        for split in args.splits:
            names = [e["name"] for e in index if e["split"] == split]
            if args.max_scenes:
                names = names[:args.max_scenes]
            t_start, done = time.perf_counter(), 0
            for k, name in enumerate(names):
                bad = []
                if checks is not None:
                    c = checks.get(name)
                    if c is None or c.get("hash") != scene_hash(scenes_dir / name):
                        unchecked.append(name)        # never checked, or changed since: trace, but say so
                    elif c["status"] == "fail":
                        skipped.append(name)
                        continue
                    else:
                        bad = c.get("bad_stations", [])
                        n_bad_st += len(bad)
                f = out_dir / "raw" / f"{name}.npz"
                if f.exists() and not args.overwrite:
                    if raw_is_current(f, scenes_dir, name, cfg, args, bad):
                        continue
                    raise SystemExit(f"{f.name} was made with a different scene, different settings or other "
                                     f"stations (bad stations from the checker). Use --overwrite to retrace, "
                                     f"or another --out folder.")
                res = trace_scene(scenes_dir / name, args, solver, bad, warm_up=not warmed)
                warmed = True
                tmp = f.with_name(f"{name}.tmp.npz")      # write then rename: no half-written files
                np.savez_compressed(tmp, **res)
                tmp.replace(f)
                done += 1
                el = time.perf_counter() - t_start
                eta = el / done * (len(names) - k - 1)
                print(f"[{split}] {k + 1}/{len(names)} {name}: {len(res['altitude'])} maps, "
                      f"mean {res['rt_time'].mean():.2f} s/map, elapsed {el / 60:.1f} min, "
                      f"eta {eta / 60:.1f} min", flush=True)
        if skipped:
            print(f"skipped {len(skipped)} scenes that FAILED in check_results.json: {skipped[:5]} "
                  f"(see the file for the reasons, or use --ignore-checks)")
        if n_bad_st:
            print(f"{n_bad_st} stations marked bad in check_results.json were left out of the station choice")
        if unchecked:
            print(f"WARNING: {len(unchecked)} traced scenes were not checked or changed since the check: "
                  f"{unchecked[:5]}; rerun check_city_scenes.py")

    merge(scenes_dir, out_dir, args.splits)


if __name__ == "__main__":
    main()