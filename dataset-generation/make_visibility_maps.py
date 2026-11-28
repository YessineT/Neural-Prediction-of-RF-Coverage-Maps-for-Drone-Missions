"""
Computes the visibility map of every map of the AIRWISE coverage dataset from its ray traced path gain.

A cell is visible when the signal-to-noise ratio (SNR) of the station-to-drone link in that cell reaches a threshold.
The SNR comes from the path gain with a link budget:

    received power (dBm) = Ptx + Gtx + Grx + path gain
    noise power    (dBm) = N0 + 10 log10(bandwidth) + noise figure          (N0 = -174 dBm/Hz, thermal noise at 290 K)
    SNR            (dB)  = received power - noise power
    visibility[k, i, j]  = 1 if SNR >= threshold, else 0

Unlike a line-of-sight map, which only says whether the direct path is free, the visibility map includes every path
found by the ray tracer (reflections, diffraction, diffuse scattering): a cell in the shadow of a building is visible
if enough signal reaches it, and a cell in line of sight is not visible if it is too far. Cells without any ray
(path gain -150 dB in the dataset) are not visible.

The output follows the coverage dataset exactly: visibility[k] belongs to the same scene, station and altitude as
target_db[k] of the coverage file of the same split, on the same 1 m grid (cell [i, j] has its centre at
x = j - 63.5 m, y = i - 63.5 m).

Output (default folder: dataset/visibility/), one file per split, visibility_<split>.npz:
    visibility    (N, 128, 128) uint8    1 = SNR >= threshold, 0 otherwise
    valid         (N, 128, 128) bool     the drone can be in the cell (surface < altitude - 0.5 m)
    snr_db        (N, 128, 128) float16  only with --save-snr: the SNR of every cell, dB
    heights       (S, 128, 128) float32  surface height of every scene (terrain + buildings), m
    scene_names   (S,)                   scene name, e.g. blocks_0007
    scene_idx     (N,) int32             scene of every map
    station_pos   (N, 3) float32         ground station x, y, z, m
    altitude      (N,) float32           drone altitude, m
    link_budget   ()                     the link budget used, as text (JSON)

The link budget has no default values: they must be given on the command line.

Usage (from the repository root):
    python dataset-generation/make_visibility_maps.py --tx-power-dbm P --bandwidth-hz B --noise-figure-db NF --snr-threshold-db T
    python dataset-generation/make_visibility_maps.py ... --splits val --save-snr     # one split, with the SNR values
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ID = "TYessine/airwise-rf-coverage-maps"
FLOOR_DB = -150.0


def load_split(split, data_dir):
    if data_dir is not None:
        path = Path(data_dir) / f"{split}.npz"
    else:
        from huggingface_hub import hf_hub_download            # public repository: no login needed to download
        path = Path(hf_hub_download(REPO_ID, f"{split}.npz", repo_type="dataset"))
    return np.load(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tx-power-dbm", type=float, required=True, help="transmit power of the ground station, dBm")
    ap.add_argument("--bandwidth-hz", type=float, required=True, help="receiver bandwidth, Hz")
    ap.add_argument("--noise-figure-db", type=float, required=True, help="receiver noise figure, dB")
    ap.add_argument("--snr-threshold-db", type=float, required=True, help="minimum SNR for a visible cell, dB")
    ap.add_argument("--tx-gain-dbi", type=float, default=0.0,
                    help="station antenna gain, dBi (default 0: the dataset uses isotropic antennas)")
    ap.add_argument("--rx-gain-dbi", type=float, default=0.0,
                    help="drone antenna gain, dBi (default 0: the dataset uses isotropic antennas)")
    ap.add_argument("--noise-psd-dbm-hz", type=float, default=-174.0, help="thermal noise density, dBm/Hz (290 K)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "ood"])
    ap.add_argument("--data-dir", default=None, help="folder with <split>.npz (default: download from Hugging Face)")
    ap.add_argument("--out", default="dataset/visibility")
    ap.add_argument("--save-snr", action="store_true", help="also save the SNR of every cell (float16, larger files)")
    ap.add_argument("--chunk", type=int, default=512, help="maps processed at once (limits memory)")
    args = ap.parse_args()

    noise_dbm = args.noise_psd_dbm_hz + 10 * np.log10(args.bandwidth_hz) + args.noise_figure_db
    gain_offset = args.tx_power_dbm + args.tx_gain_dbi + args.rx_gain_dbi - noise_dbm      # SNR = path gain + offset
    min_path_gain = args.snr_threshold_db - gain_offset
    budget = dict(tx_power_dbm=args.tx_power_dbm, tx_gain_dbi=args.tx_gain_dbi, rx_gain_dbi=args.rx_gain_dbi,
                  bandwidth_hz=args.bandwidth_hz, noise_figure_db=args.noise_figure_db,
                  noise_psd_dbm_hz=args.noise_psd_dbm_hz, noise_power_dbm=round(float(noise_dbm), 3),
                  snr_threshold_db=args.snr_threshold_db, min_path_gain_db=round(float(min_path_gain), 3))
    print("link budget:", json.dumps(budget))
    print(f"a cell is visible when its path gain is >= {min_path_gain:.2f} dB")
    if min_path_gain <= FLOOR_DB:
        print(f"WARNING: the threshold is at or below the dataset floor ({FLOOR_DB} dB): cells without any ray would "
              f"count as visible")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        t0 = time.perf_counter()
        d = load_split(split, args.data_dir)
        heights = d["heights"].astype(np.float32)
        scene_idx = d["scene_idx"].astype(np.int32)
        altitude = d["altitude"].astype(np.float32)
        target = d["target_db"]
        n = len(altitude)
        vis = np.zeros(target.shape, np.uint8)
        valid = np.zeros(target.shape, bool)
        snr = np.zeros(target.shape, np.float16) if args.save_snr else None
        for b in range(0, n, args.chunk):
            k = slice(b, min(b + args.chunk, n))
            s = np.maximum(target[k], FLOOR_DB) + gain_offset
            vis[k] = s >= args.snr_threshold_db
            valid[k] = heights[scene_idx[k]] < altitude[k, None, None] - 0.5
            if snr is not None:
                snr[k] = s.astype(np.float16)
        extra = {"snr_db": snr} if snr is not None else {}
        out = out_dir / f"visibility_{split}.npz"
        np.savez_compressed(out, visibility=vis, valid=valid, heights=heights, scene_names=d["scene_names"],
                            scene_idx=scene_idx, station_pos=d["station_pos"].astype(np.float32), altitude=altitude,
                            link_budget=np.array(json.dumps(budget)), **extra)
        print(f"{split}: {n} maps, {100 * vis[valid].mean():.1f} % of the valid cells visible, "
              f"{time.perf_counter() - t0:.0f} s -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
