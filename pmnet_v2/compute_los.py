"""
Visibility-map dataset (DISCRETE deliverable), computed with Yessine's own line_of_sight() function (yessine_port.py),
the same one that makes the "los" input channel of the models.

For every (scene, station, altitude) map: a 128 x 128 binary map, 1 = the straight segment from the ground station
to that cell of the drone plane stays above the buildings and the terrain (a drone there sees the station), 0 = blocked.
Cells inside buildings taller than the drone (invalid cells) are set to 0 with --mask-valid (recommended for the
deliverable; the model channel does not use that mask).

Output per split: <out>/los_<split>.npz
    los_packed   (n, 128, 16) uint8, bits packed along x:  np.unpackbits(los_packed, axis=-1) -> (n, 128, 128)
    scene_idx, station_pos, altitude   copied from the dataset (cell [row, col] = [y, x], 1 m cells, centred on 0)
    los_samples, masked_valid          the settings used

    python compute_los.py --out los_cache --mask-valid --check
    python compute_los.py --out los_cache --data ./npz_folder --splits val test --limit 200 --check
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import train_pmnet as T
import yessine_port as Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="")
    ap.add_argument("--hf-repo", default=T.HF_REPO)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", default="los_cache")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "ood"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mask-valid", action="store_true", help="set the cells inside tall buildings to 0")
    ap.add_argument("--check", action="store_true", help="compare with the ray traced path gain")
    ap.add_argument("--limit", type=int, default=0, help="only the first N maps of each split (testing)")
    a = ap.parse_args()

    src = dict(data=a.data, repo=a.hf_repo, cache=a.cache_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Y.CFG
    for split in a.splits:
        f = T.get_split_file(split, src)
        if f is None:
            print(f"{split}.npz not found, skipped")
            continue
        ds = Y.CoverageSplit(f, split)
        n = len(ds) if not a.limit else min(a.limit, len(ds))
        vis, ok = [], []
        for s in range(0, n, a.batch):
            h, st, alt, tgt = ds.batch(np.arange(s, min(s + a.batch, n)), Y.DEVICE)
            los = Y.line_of_sight(h, st, alt, cfg) > 0.5
            valid = h < alt.view(-1, 1, 1) - 0.5
            if a.mask_valid:
                los = los & valid
            vis.append(los.cpu().numpy())
            if a.check:
                ok.append((los.cpu(), valid.cpu(), tgt.cpu()))
            if (s // a.batch) % 250 == 0:
                print(f"  {split}: {s}/{n}", flush=True)
        vis = np.concatenate(vis)
        si = ds.scene_idx.numpy()[:n]
        np.savez_compressed(out / f"los_{split}.npz", los_packed=np.packbits(vis, axis=-1), scene_idx=si,
                            station_pos=ds.station.numpy()[:n], altitude=ds.altitude.numpy()[:n],
                            los_samples=cfg["los_samples"], masked_valid=a.mask_valid)
        print(f"{split}: {n} maps, {100 * vis.mean():.1f}% of the cells see the station -> {out / f'los_{split}.npz'}")
        if a.check:
            los = torch.cat([o[0] for o in ok]); valid = torch.cat([o[1] for o in ok]); tgt = torch.cat([o[2] for o in ok])
            seen, hidden = los & valid, ~los & valid
            print(f"    mean path gain  seen: {tgt[seen].mean():.1f} dB   hidden: {tgt[hidden].mean():.1f} dB")
            print(f"    seen cells at the no-signal floor (< -140 dB): {100 * (tgt[seen] < -140).float().mean():.2f}%  (should be ~0)")
            print(f"    hidden cells above -90 dB: {100 * (tgt[hidden] > -90).float().mean():.2f}%  (reflections, should be small)")


if __name__ == "__main__":
    main()