"""
Loads each scene in Sionna and traces a few sources to one receiver.
"""

from pathlib import Path

import numpy as np
from sionna.rt import load_scene, PathSolver, PlanarArray, Receiver, Transmitter

FREQUENCY = 2.4e9        # Hz
NUM_SOURCES = 5          # sources traced per scene
RX_XY = (-40.0, -40.0)   # m
RX_AGL = 30.0            # m above the highest ground

SCENES = ["flat", "flat_building", "irregular", "irregular_building"]
ROOT = Path(__file__).resolve().parent / "scenes"


def main():
    solver = PathSolver()
    for name in SCENES:
        scene = load_scene(str(ROOT / name / f"{name}.xml"))
        scene.frequency = FREQUENCY
        iso = dict(num_rows=1, num_cols=1, vertical_spacing=0.5,
                   horizontal_spacing=0.5, pattern="iso", polarization="V")
        scene.tx_array = PlanarArray(**iso)
        scene.rx_array = PlanarArray(**iso)

        terrain = np.load(ROOT / name / "height_map.npz")["terrain"]
        src = np.load(ROOT / name / "sources.npz")["positions"][:NUM_SOURCES]

        scene.add(Receiver(name="rx", position=[RX_XY[0], RX_XY[1],
                                                 float(terrain.max()) + RX_AGL]))
        for i, p in enumerate(src):
            scene.add(Transmitter(name=f"tx{i}", position=[float(c) for c in p]))

        paths = solver(scene, max_depth=3, los=True, specular_reflection=True,
                       diffuse_reflection=False, refraction=False)
        a_re, a_im = paths.a
        power = np.abs(np.array(a_re) + 1j * np.array(a_im)) ** 2
        power = power.reshape(len(src), -1).sum(axis=1)
        valid = np.array(paths.valid).reshape(len(src), -1).sum(axis=1)

        print(f"\n{name}: objects {list(scene.objects.keys())}")
        for i in range(len(src)):
            p_db = 10 * np.log10(power[i]) if power[i] > 0 else -np.inf
            print(f"  tx{i}: {int(valid[i])} paths, path gain {p_db:.1f} dB")


if __name__ == "__main__":
    main()
