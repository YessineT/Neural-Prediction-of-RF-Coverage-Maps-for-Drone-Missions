"""
Represents each radio map of a dataset as an image for an image-to-image network.

Input of one map (channels, 128 x 128, same pixel grid as the target map):
    height      surface height (ground + buildings) in m, divided by H_NORM
    station     ground station marker: a Gaussian blob (sigma 1.5 cells, peak 1) at the station position
    valid       1 where the drone plane is above the surface (heights < altitude - 0.5), 0 inside buildings
    altitude    constant image: drone altitude / ALT_NORM
    station_z   constant image: station height / H_NORM
    distance    (optional, distance=True) 3D distance from the station to every cell of the plane / DIST_NORM
Target: target_db (path gain in dB, -150 = no signal) and the valid mask, which tells where the loss is computed.

Pixel convention (as in generate_city_scenes.py): cell centres are -63.5 ... 63.5 m, array index [row = y, col = x].

Use it in your training code:
    from scene_images import SceneImageDataset
    ds = SceneImageDataset("dataset_sami", "train")            # works with torch DataLoader (numpy items)
    item = ds[0]          # dict: x (C,128,128) float32, y (128,128) dB, valid (128,128) bool, name, altitude
or as functions: make_inputs(z, idx)  ->  (n, C, 128, 128)

Command line, to look at the images and check the alignment:
    python scene_images.py --dataset dataset_sami --split train
It saves scene_images_examples.png in the dataset folder and checks that the station position is aligned with the
target maps (signal strongest near the station); a wrong orientation would silently ruin the training.
"""
import argparse
from pathlib import Path

import numpy as np

HALF, CELL, N = 64.0, 1.0, 128
H_NORM = 64.0          # m, scale of heights (buildings go up to 55 m; hilly terrain adds a few metres)
ALT_NORM = 60.0        # m, highest drone altitude of the dataset
DIST_NORM = 200.0      # m, > the longest 3D distance in the scene
STATION_SIGMA = 1.5    # cells
CHANNELS = ["height", "station", "valid", "altitude", "station_z"]

_xc = -HALF + CELL * (np.arange(N) + 0.5)          # cell centres, same for x and y
_X, _Y = np.meshgrid(_xc, _xc)                      # [row = y, col = x]


def channel_names(distance=False):
    return CHANNELS + (["distance"] if distance else [])


def make_input(height, station, altitude, distance=False, sigma=STATION_SIGMA):
    """One map as (C, 128, 128) float32. height (128,128) in m, station (x, y, z) in m, altitude in m."""
    sx, sy, sz = (float(v) for v in station)
    d2 = (_X - sx) ** 2 + (_Y - sy) ** 2
    chans = [height / H_NORM,
             np.exp(-d2 / (2 * sigma ** 2)),
             (height < altitude - 0.5).astype(np.float32),
             np.full((N, N), altitude / ALT_NORM, np.float32),
             np.full((N, N), sz / H_NORM, np.float32)]
    if distance:
        chans.append(np.sqrt(d2 + (altitude - sz) ** 2) / DIST_NORM)
    return np.stack(chans).astype(np.float32)


def make_inputs(z, idx=None, distance=False):
    """Inputs of the maps idx (default all) of a loaded split z (dict of arrays) as (n, C, 128, 128)."""
    idx = np.arange(len(z["altitude"])) if idx is None else np.atleast_1d(idx)
    return np.stack([make_input(z["heights"][z["scene_idx"][i]], z["station_pos"][i], float(z["altitude"][i]), distance)
                     for i in idx])


def station_pixel(station):
    """(row, col) of the cell that contains the station."""
    col = int(np.clip(np.floor((float(station[0]) + HALF) / CELL), 0, N - 1))
    row = int(np.clip(np.floor((float(station[1]) + HALF) / CELL), 0, N - 1))
    return row, col


def load_split(folder, split):
    f = Path(folder) / f"{split}.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


class SceneImageDataset:
    """Maps of one split as (input image, target, valid mask). Images are built when an item is read, so nothing
    big is stored. Compatible with torch.utils.data.DataLoader (items are dicts of numpy arrays)."""

    def __init__(self, folder, split, distance=False):
        self.z = load_split(folder, split)
        self.distance = distance
        self.channels = channel_names(distance)

    def __len__(self):
        return len(self.z["altitude"])

    def __getitem__(self, i):
        z = self.z
        s = z["scene_idx"][i]
        h = z["heights"][s]
        alt = float(z["altitude"][i])
        return dict(x=make_input(h, z["station_pos"][i], alt, self.distance),
                    y=z["target_db"][i].astype(np.float32),
                    valid=(h < alt - 0.5),
                    name=str(z["scene_names"][s]), altitude=alt)


# ---------------------------------------------------------------- checks
def alignment_check(z, max_maps=2000, seed=0):
    """Share of maps whose signal near the station is among the strongest of the map, for the maps as stored and for
    their transpose. The stored orientation must pass (about 90 % or more) and the transpose must fail."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z["altitude"]))[:max_maps]
    res = {"as stored": [0, 0], "transposed": [0, 0]}     # [passed, tested]
    for i in idx:
        s = z["scene_idx"][i]
        valid = z["heights"][s] < z["altitude"][i] - 0.5
        r, c = station_pixel(z["station_pos"][i])
        for name, tgt, val in (("as stored", z["target_db"][i], valid),
                               ("transposed", z["target_db"][i].T, valid)):
            win = (slice(max(r - 2, 0), r + 3), slice(max(c - 2, 0), c + 3))
            near = tgt[win][val[win]]
            if near.size < 5 or val.sum() < 100:
                continue
            res[name][1] += 1
            res[name][0] += near.mean() >= np.percentile(tgt[val], 90)
    return {k: (p / t if t else float("nan"), t) for k, (p, t) in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset folder (output of make_dataset.py)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-show", type=int, default=4, help="maps shown in the figure")
    ap.add_argument("--distance", action="store_true", help="add the distance channel")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    z = load_split(a.dataset, a.split)
    ds = SceneImageDataset(a.dataset, a.split, a.distance)
    n = len(ds)
    print(f"{a.dataset}/{a.split}: {n} maps from {len(z['scene_names'])} scenes, channels: {ds.channels}")
    print(f"surface height: min {z['heights'].min():.1f} m, max {z['heights'].max():.1f} m (H_NORM = {H_NORM:.0f})")
    print(f"altitudes: {np.unique(z['altitude']).tolist()} (ALT_NORM = {ALT_NORM:.0f}); "
          f"station z: {z['station_pos'][:, 2].min():.1f} to {z['station_pos'][:, 2].max():.1f} m")
    if z["heights"].max() > H_NORM:
        print("  note: some heights are above H_NORM, inputs go above 1 (fine, or raise H_NORM)")
    x0 = ds[0]["x"]
    print(f"one input: shape {x0.shape}, dtype {x0.dtype}, per-channel max {np.round(x0.max(axis=(1, 2)), 2).tolist()}")

    al = alignment_check(z)
    print("\nalignment of station and target maps (share of maps with the strongest signal near the station):")
    for k, (share, t) in al.items():
        print(f"    {k:11s} {share:5.1%}  ({t} maps)")
    ok = al["as stored"][0] >= 0.8 and al["as stored"][0] > al["transposed"][0] + 0.2
    print("    -> OK" if ok else "    -> WARNING: station and target maps do not look aligned, check the orientation")

    if a.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    show = rng.choice(n, size=min(a.n_show, n), replace=False)
    nc = len(ds.channels)
    fig, axes = plt.subplots(len(show), nc + 1, figsize=(2.9 * (nc + 1), 2.9 * len(show)),
                             constrained_layout=True, squeeze=False)
    for row, i in enumerate(show):
        it = ds[i]
        for c, name in enumerate(ds.channels):
            ax = axes[row, c]
            # fixed colour scale 0..1: a uniform image shows its real value instead of always being purple
            ax.imshow(it["x"][c], origin="lower", cmap="magma" if name == "height" else "viridis", vmin=0, vmax=1)
            if name in ("altitude", "station_z"):         # constant images: write the number in metres
                value = it["altitude"] if name == "altitude" else z["station_pos"][i][2]
                ax.text(0.5, 0.5, f"{value:.1f} m" if name == "station_z" else f"{value:.0f} m",
                        transform=ax.transAxes, ha="center", va="center", color="white", fontsize=16,
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.3"))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=10)
        ax = axes[row, nc]
        ax.imshow(np.where(it["valid"], it["y"], np.nan), origin="lower", cmap="viridis", vmin=-120, vmax=-60)
        sr, sc = station_pixel(z["station_pos"][i])
        ax.plot(sc, sr, "r*", ms=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("target [dB]" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(f"{it['name']}\n{it['altitude']:.0f} m", fontsize=9)
    out = Path(a.dataset) / "scene_images_examples_1.png"
    fig.savefig(out, dpi=110)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()