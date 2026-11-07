"""
Helpers shared by generate_city_scenes.py, check_city_scenes.py and make_dataset.py.

Keeping them in one place guarantees that the checker hashes scenes, picks stations and ray traces radio maps
exactly like the dataset does. Sionna is only imported by the functions that need it.
"""

import hashlib
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Ray tracing defaults of the dataset (and so of the physics checks). Chosen to be close to real propagation
# and converged; measured on 6 scenes x 2 stations x 2 altitudes (RTX 4070 Laptop):
#   diffraction    over roofs and around corners, the main way signal reaches shadowed cells. Without it,
#                  shadowed cells are off by up to ~18 dB (90th percentile).
#   diffuse        scattering from rough surfaces (scattering_coefficient in the scene XML, written by
#                  generate_city_scenes.py). Changes shadowed cells by up to ~18 dB (90th percentile).
#   refraction     (through walls) stays OFF: the buildings are hollow boxes without inner walls or floors, so
#                  energy would cross a whole building losing only two walls, far less than real buildings,
#                  and it would put signal inside the buildings.
#   max_depth 8    4.9 % of the cells switch between signal / no signal from depth 5 to 10, 1.2 % from 8 to 10.
#   samples 1e8    no line of sight holes; shadowed cells within ~0.6 dB (median) of 2e8 rays. ~0.36 s per map.
RT_DEFAULTS = dict(frequency=2.4e9, samples=100_000_000, max_depth=8, diffraction=True, diffuse=True)
DATASET_DEFAULTS = dict(stations=3, seed=2026)      # stations per scene, seed of the station choice


def scene_hash(scene_dir):
    """sha1 of the surface height map; changes whenever the scene is regenerated differently."""
    with np.load(Path(scene_dir) / "height_map.npz") as f:
        return hashlib.sha1(f["surface"].tobytes()).hexdigest()


def pick_stations(name, n_src, n_pick, seed, bad=()):
    """Stations of a scene used for the dataset: a fixed random choice (seed + scene name). A chosen station that
    is in bad (its ray traced maps FAILED in check_results.json) is replaced by the next good one in a fixed random
    order; the other chosen stations stay the same, so marking a station bad only changes scenes that used it."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    first = rng.choice(n_src, size=min(n_pick, n_src), replace=False)
    bad = {int(b) for b in bad}
    keep = [int(i) for i in first if int(i) not in bad]
    if len(keep) < len(first):
        for i in rng.permutation(n_src):
            if len(keep) == len(first):
                break
            if int(i) not in bad and int(i) not in keep:
                keep.append(int(i))
    return np.array(keep, int)


def parallel_map(fn, jobs, workers, min_jobs):
    """fn over jobs, results in job order. With workers > 1 and at least min_jobs jobs, the jobs run in a
    process pool (fn must be a top level function). Starting the pool costs ~1-2 s, so small runs stay in
    this process."""
    if workers <= 1 or len(jobs) < min_jobs:
        yield from map(fn, jobs)
        return
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        yield from ex.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * workers)))


def load_radio_scene(path, frequency):
    """Sionna scene with one isotropic, vertically polarised antenna for transmitters and receivers."""
    from sionna.rt import load_scene, PlanarArray
    scene = load_scene(str(path))
    scene.frequency = frequency
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    return scene


def radio_map(solver, scene, alt, side, cell, samples, max_depth, diffraction, diffuse):
    """One horizontal radio map (side x side m, centred on the origin) at height alt for every transmitter of
    the scene: line of sight, specular reflections, and optionally diffraction and diffuse reflections.
    No refraction (see RT_DEFAULTS)."""
    return solver(scene, center=[0.0, 0.0, float(alt)], orientation=[0.0, 0.0, 0.0],
                  size=[side, side], cell_size=[cell, cell], samples_per_tx=int(samples),
                  max_depth=int(max_depth), los=True, specular_reflection=True,
                  diffuse_reflection=bool(diffuse), refraction=False, diffraction=bool(diffraction))
