"""
Checks that YOUR towns do not overlap with the towns of the other team members.

It rebuilds the other people's towns with the shared generator (generate_city_scenes.py), using exactly their
town numbers and the same seed recipe (BASE_SEED, family number, town number), writes them to folders next to
this file, and compares your towns with them by content (height map + station positions), not only by name.

    reference folders (created if missing, reused if complete):
        scenes_city_yessine/   towns 0-149   (blocks, grid, downtown)  and 0-29  (canyon, courtyard, hilly)
        scenes_city_mahdi/     towns 150-299 (blocks, grid, downtown)  and 30-59 (canyon, courtyard, hilly)

Put this file in dataset-generation/ (next to generate_city_scenes.py and common.py) and run, for example:
    python check_all_overlap.py --mine scenes_city_<yourname>
    python check_all_overlap.py --mine scenes_city_<yourname> --rebuild        # rebuild the reference folders
    python check_all_overlap.py --mine scenes_city_<yourname> --others some_other_folder   # extra folders

Checks, for every one of your towns against every reference town:
    identical town      same buildings AND same stations
    same buildings      same height map, even if the stations differ
    same name           e.g. blocks_0200 exists on both sides
Also reported: duplicates inside your own folder, and your town numbers that fall inside a teammate's range.
Ends with "NO OVERLAP" (exit code 0) or "OVERLAP FOUND" (exit code 1).

Only exact copies are detected, not rotated or shifted versions of a layout (very unlikely with the
independent seed of every town).
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Town numbers already used. Add a line here when somebody else takes a range.
TEAM = {
    "yessine": dict(folder="scenes_city_yessine", id_start=0, id_count=150, ood_start=0, ood_count=30),
    "mahdi": dict(folder="scenes_city_mahdi", id_start=150, id_count=150, ood_start=30, ood_count=30),
}


# ---------------------------------------------------------------- fingerprints
def fingerprints(town_dir):
    """(content code, buildings-only code). Content = height map + station positions."""
    with np.load(town_dir / "height_map.npz") as f:
        surface = np.round(f["surface"], 6).tobytes()
    h_b = hashlib.sha1(surface).hexdigest()
    h = hashlib.sha1(surface)
    src = town_dir / "sources.npz"
    if src.exists():
        with np.load(src) as f:
            h.update(np.round(f["positions"], 6).tobytes())
    return h.hexdigest(), h_b


def towns_in(folder):
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"folder not found: {folder}")
    return {p.name: fingerprints(p) for p in sorted(folder.iterdir()) if (p / "height_map.npz").exists()}


# ---------------------------------------------------------------- rebuilding the other people's towns
def town_plan(gen, spec):
    """(name, family, split, family number, town number) of every town of a person, like make_more_towns.py."""
    plan = gen.ID_FAMILIES + gen.OOD_FAMILIES       # the family number is part of the seed
    out = []
    for fam in plan:
        ood = fam in gen.OOD_FAMILIES
        start, count = (spec["ood_start"], spec["ood_count"]) if ood else (spec["id_start"], spec["id_count"])
        for k, i in enumerate(range(start, start + count)):
            f = k / count
            split = "ood" if ood else ("train" if f < gen.ID_SPLIT[0] else
                                       "val" if f < gen.ID_SPLIT[0] + gen.ID_SPLIT[1] else "test")
            out.append((f"{fam}_{i:04d}", fam, split, plan.index(fam), i))
    return out


def build_reference(gen, parallel_map, person, spec, ref_dir, workers, rebuild):
    out = ref_dir / spec["folder"]
    towns = town_plan(gen, spec)
    expected = {t[0] for t in towns}
    present = set()
    if out.is_dir():
        present = {p.name for p in out.iterdir() if (p / "height_map.npz").exists() and (p / "sources.npz").exists()}
    if not rebuild and expected <= present:
        print(f"{person}: {len(expected)} towns already in {out}, reused (use --rebuild to regenerate)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(out, name, fam, split, gen.BASE_SEED, fam_id, i) for name, fam, split, fam_id, i in towns]
    index = []
    for e in parallel_map(gen.make_one, jobs, workers, min_jobs=20):
        index.append(e)
        print(f"      building {person}'s towns {len(index)}/{len(jobs)}   ", end="\r", flush=True)
    print(" " * 60, end="\r")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{person}: {len(index)} towns built in {out}")
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mine", required=True, help="folder with your towns")
    ap.add_argument("--gen-dir", default=str(HERE), help="folder with generate_city_scenes.py and common.py")
    ap.add_argument("--ref-dir", default=str(HERE), help="where the rebuilt reference folders are written")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the reference folders even if they exist")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--others", nargs="*", default=[], help="extra folders of towns to compare with")
    a = ap.parse_args()

    sys.path.insert(0, a.gen_dir)
    import generate_city_scenes as gen          # the shared generator, unchanged
    from common import parallel_map

    mine_dir = Path(a.mine).resolve()
    ref_dir = Path(a.ref_dir).resolve()
    refs = {}
    for person, spec in TEAM.items():
        if (ref_dir / spec["folder"]).resolve() == mine_dir:
            sys.exit(f"--mine points to the reference folder of {person} ({mine_dir}); use your own folder")
        refs[person] = build_reference(gen, parallel_map, person, spec, ref_dir, a.workers, a.rebuild)

    mine = towns_in(mine_dir)
    print(f"\nyour towns: {len(mine)} in {mine_dir}")
    if not mine:
        sys.exit("no towns found in your folder (no height_map.npz)")

    # all the other towns: name -> (label, fingerprints)
    others = {}
    for person, folder in refs.items():
        t = towns_in(folder)
        print(f"{person}'s towns: {len(t)}")
        for k, v in t.items():
            others[f"{person}/{k}"] = v
    for folder in a.others:
        t = towns_in(folder)
        print(f"other towns: {len(t)} in {folder}")
        for k, v in t.items():
            others[f"{Path(folder).name}/{k}"] = v

    by_full, by_build = {}, {}
    for label, (full, bld) in others.items():
        by_full.setdefault(full, []).append(label)
        by_build.setdefault(bld, []).append(label)

    identical = [(m, by_full[v[0]]) for m, v in mine.items() if v[0] in by_full]
    same_buildings = [(m, by_build[v[1]]) for m, v in mine.items() if v[1] in by_build and v[0] not in by_full]
    other_names = {k.split("/")[-1] for k in others}
    same_name = sorted(set(mine) & other_names)

    # your own folder: duplicates inside it
    seen, own_dups = {}, []
    for m, v in mine.items():
        if v[0] in seen:
            own_dups.append((m, seen[v[0]]))
        seen.setdefault(v[0], m)

    # your town numbers inside a teammate's range
    in_range = []
    for m in mine:
        match = re.fullmatch(r"([a-z]+)_(\d+)", m)
        if not match:
            continue
        fam, num = match.group(1), int(match.group(2))
        ood = fam in gen.OOD_FAMILIES
        for person, spec in TEAM.items():
            s, c = (spec["ood_start"], spec["ood_count"]) if ood else (spec["id_start"], spec["id_count"])
            if s <= num < s + c:
                in_range.append((m, person))

    print(f"\ncompared {len(mine)} of your towns with {len(others)} other towns")
    print(f"  identical towns (same buildings and stations): {len(identical)}")
    for m, o in identical[:10]:
        print(f"      {m} = {o}")
    print(f"  same buildings, different stations:            {len(same_buildings)}")
    for m, o in same_buildings[:10]:
        print(f"      {m} = {o}")
    print(f"  same town names:                               {len(same_name)}"
          + (f"  e.g. {same_name[:5]}" if same_name else ""))
    print(f"  town numbers inside a teammate's range:        {len(in_range)}"
          + (f"  e.g. {in_range[:5]}" if in_range else ""))
    print(f"  duplicates inside your own folder:             {len(own_dups)}")
    for m, o in own_dups[:10]:
        print(f"      {m} = {o}")

    ok = not (identical or same_buildings or same_name or own_dups)
    print("\nNO OVERLAP" if ok else "\nOVERLAP FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()