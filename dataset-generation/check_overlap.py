"""
Checks that two sets of towns do NOT overlap: no identical town, no shared town name.

A town is compared by its content (the height map and the ground-station positions), not only by its name,
so even a renamed copy would be found.

Put this file in dataset-generation/ and run, for example:
    python check_overlap.py --mine scenes_city_mahdi --make-yessine
        builds Yessine's 540 towns (numbers 0-149 and 0-29) in a temporary folder with his generator,
        then compares them with yours
    python check_overlap.py --mine scenes_city_mahdi --others scenes_city other_folder
        compares with towns that are already on your disk (any number of folders)
Ends with "NO OVERLAP" (exit code 0) or "OVERLAP FOUND" (exit code 1).
"""
import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fingerprint(town_dir):
    """A short code that is the same only if two towns have the same buildings AND the same stations."""
    h = hashlib.sha1()
    with np.load(town_dir / "height_map.npz") as f:
        h.update(np.round(f["surface"], 6).tobytes())
    with np.load(town_dir / "sources.npz") as f:
        h.update(np.round(f["positions"], 6).tobytes())
    return h.hexdigest()


def towns_in(folder):
    folder = Path(folder)
    return {p.name: fingerprint(p) for p in sorted(folder.iterdir()) if (p / "height_map.npz").exists()}


def make_yessine_towns(out):
    """Yessine's default run: towns 0-149 (blocks, grid, downtown) and 0-29 (canyon, courtyard, hilly)."""
    sys.path.insert(0, str(HERE))
    import generate_city_scenes as gen
    plan = gen.ID_FAMILIES + gen.OOD_FAMILIES
    n = 0
    for fam in plan:
        count = 30 if fam in gen.OOD_FAMILIES else 150
        for i in range(count):
            gen.make_one((out, f"{fam}_{i:04d}", fam, "-", gen.BASE_SEED, plan.index(fam), i))
            n += 1
            print(f"      building Yessine's towns {n}/540   ", end="\r", flush=True)
    print(" " * 50, end="\r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="folder with your towns")
    ap.add_argument("--others", nargs="*", default=[], help="folders with the other people's towns")
    ap.add_argument("--make-yessine", action="store_true",
                    help="build Yessine's 540 towns in a temporary folder and compare with them too")
    a = ap.parse_args()

    mine = towns_in(a.mine)
    print(f"your towns: {len(mine)} in {a.mine}")
    if len(set(mine.values())) != len(mine):
        print("WARNING: some of your own towns are identical to each other")

    others = {}
    for folder in a.others:
        t = towns_in(folder)
        print(f"other towns: {len(t)} in {folder}")
        others.update({f"{folder}/{k}": v for k, v in t.items()})
    tmp = None
    if a.make_yessine:
        tmp = tempfile.TemporaryDirectory()
        make_yessine_towns(Path(tmp.name))
        t = towns_in(tmp.name)
        print(f"Yessine's towns: {len(t)} (rebuilt with his generator)")
        others.update({f"yessine/{k}": v for k, v in t.items()})
    if not others:
        sys.exit("nothing to compare with: give --others folders and/or --make-yessine")

    by_code = {}
    for name, code in others.items():
        by_code.setdefault(code, []).append(name)
    same_content = [(m, by_code[c]) for m, c in mine.items() if c in by_code]
    other_names = {k.split("/")[-1] for k in others}
    same_name = sorted(set(mine) & other_names)

    print(f"\ncompared {len(mine)} of your towns with {len(others)} other towns")
    print(f"  identical towns (same buildings and stations): {len(same_content)}")
    for m, o in same_content[:10]:
        print(f"      {m} = {o}")
    print(f"  same town names: {len(same_name)}" + (f"  e.g. {same_name[:5]}" if same_name else ""))
    ok = not same_content and not same_name
    print("\nNO OVERLAP" if ok else "\nOVERLAP FOUND")
    if tmp:
        tmp.cleanup()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
