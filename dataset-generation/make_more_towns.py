"""
Make MORE towns with Yessine's generator (dataset-generation/generate_city_scenes.py), using town numbers he
does not use, so there is no overlap. Same generator, same settings, same seed recipe:
    seed of a town = (BASE_SEED 2026, family number, town number)   -> exactly like his main()

Put this file in dataset-generation/ (next to generate_city_scenes.py) and run:
    python make_more_towns.py
Defaults = Mahdi's share: towns 150-299 (blocks, grid, downtown) and 30-59 (canyon, courtyard, hilly),
written to dataset-generation/scenes_city_mahdi/ (index.json, config.json, overview.png included),
so check_city_scenes.py and make_dataset.py work on it with  --scenes scenes_city_mahdi

Sami's share : towns 300-449 (blocks, grid, downtown) and 60-89 (canyon, courtyard, hilly)
In order to generate the scenes, you need to run python make_more_towns.py --id-start 300 --id-count 150 --ood-start 60 --ood-count 30 --out scenes_city_sami
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--gen-dir", default=str(HERE), help="folder with generate_city_scenes.py and common.py")
ap.add_argument("--out", default=str(HERE / "scenes_city_mahdi"))
ap.add_argument("--id-start", type=int, default=150)
ap.add_argument("--id-count", type=int, default=150)
ap.add_argument("--ood-start", type=int, default=30)
ap.add_argument("--ood-count", type=int, default=30)
ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
a = ap.parse_args()

sys.path.insert(0, a.gen_dir)
import generate_city_scenes as gen          # Yessine's generator, unchanged
from common import parallel_map

if __name__ == "__main__":
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = gen.ID_FAMILIES + gen.OOD_FAMILIES      # same order as his main(): the family number is part of the seed
    jobs = []
    for fam in plan:
        ood = fam in gen.OOD_FAMILIES
        start, count = (a.ood_start, a.ood_count) if ood else (a.id_start, a.id_count)
        for k, i in enumerate(range(start, start + count)):
            f = k / count                              # same 80 / 10 / 10 rule as his, inside our range
            split = "ood" if ood else ("train" if f < gen.ID_SPLIT[0] else
                                       "val" if f < gen.ID_SPLIT[0] + gen.ID_SPLIT[1] else "test")
            jobs.append((out, f"{fam}_{i:04d}", fam, split, gen.BASE_SEED, plan.index(fam), i))

    index = []
    for e in parallel_map(gen.make_one, jobs, a.workers, min_jobs=20):   # his own per-town function
        index.append(e)
        print(f"      {len(index)}/{len(jobs)} {e['name']}        ", end="\r", flush=True)
    print(" " * 50, end="\r")

    (out / "index.json").write_text(json.dumps(index, indent=1))
    cfg = {k: v for k, v in vars(gen).items()
           if k.isupper() and k != "LAYOUTS" and isinstance(v, (int, float, str, tuple, list, dict))}
    cfg.update(SEED=gen.BASE_SEED, ID_RANGE=[a.id_start, a.id_start + a.id_count - 1],
               OOD_RANGE=[a.ood_start, a.ood_start + a.ood_count - 1])
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    gen.overview(out, index)
    counts = {s: sum(e["split"] == s for e in index) for s in ["train", "val", "test", "ood"]}
    print(f"{len(index)} towns written to {out} {counts}")
