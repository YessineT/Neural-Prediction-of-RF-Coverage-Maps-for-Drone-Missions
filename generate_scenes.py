"""
Builds four synthetic scenes (flat, flat_building, irregular, irregular_building)
with the same Poisson set of ground sources. Run from this folder:
    python generate_scenes.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 2026

AREA_SIDE = 100.0            # m
GRID_STEP = 1.0              # m, terrain mesh step

TERRAIN_AMPLITUDE = 3.0      # m, max height
TERRAIN_SCALE = 25.0         # m, hill size

BUILDING_CENTER = (0.0, 0.0)          # m
BUILDING_FOOTPRINT = (30.0, 20.0)     # m
BUILDING_HEIGHT = 15.0                # m

SOURCE_DENSITY = 0.15                 # sources per m^2
MIN_SOURCES = 1000
SOURCE_HEIGHT_AGL = (1.0, 10.0)       # m above ground
EDGE_MARGIN = 5.0                     # m
BUILDING_MARGIN = 2.0                 # m

# ITU materials, ground valid only 1 to 10 GHz
GROUND_MATERIAL = "medium_dry_ground"
BUILDING_MATERIAL = "concrete"

OUT_DIR = Path(__file__).resolve().parent / "scenes"


def write_ply(path, vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    records = np.empty(len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    records["n"] = 3
    records["idx"] = faces
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())
        f.write(records.tobytes())


def terrain_mesh(x, y, z):
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    iy, ix = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    v00 = (iy * nx + ix).ravel()
    v10 = v00 + 1
    v01 = v00 + nx
    v11 = v01 + 1
    faces = np.concatenate([
        np.column_stack([v00, v10, v11]),
        np.column_stack([v00, v11, v01]),
    ])
    return vertices, faces


def box_mesh(x0, x1, y0, y1, z0, z1):
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = np.array([
        [4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7], [1, 2, 6], [1, 6, 5],
    ])
    return vertices, faces


def ground_height(px, py, x, y, z):
    """Terrain height at (px, py), exact on the triangles."""
    fx = (px - x[0]) / GRID_STEP
    fy = (py - y[0]) / GRID_STEP
    ix = np.clip(np.floor(fx).astype(int), 0, len(x) - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, len(y) - 2)
    fx -= ix
    fy -= iy
    z00 = z[iy, ix]
    z10 = z[iy, ix + 1]
    z01 = z[iy + 1, ix]
    z11 = z[iy + 1, ix + 1]
    lower = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    upper = z00 + fy * (z01 - z00) + fx * (z11 - z01)
    return np.where(fx >= fy, lower, upper)


def irregular_terrain(n, rng):
    """Low pass filtered white noise."""
    noise = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n, d=GRID_STEP)
    KX, KY = np.meshgrid(k, k)
    lowpass = np.exp(-0.5 * (np.pi * np.hypot(KX, KY) * TERRAIN_SCALE) ** 2)
    z = np.real(np.fft.ifft2(np.fft.fft2(noise) * lowpass))
    z -= z.mean()
    return z * TERRAIN_AMPLITUDE / np.abs(z).max()


def write_xml(path, shapes):
    lines = ['<scene version="2.1.0">']
    for mat_id, itu_type in [("mat-itu-ground", GROUND_MATERIAL),
                             ("mat-itu-building", BUILDING_MATERIAL)]:
        lines += [
            f'    <bsdf type="itu-radio-material" id="{mat_id}">',
            f'        <string name="type" value="{itu_type}"/>',
            '        <float name="thickness" value="0.2"/>',
            '    </bsdf>',
        ]
    for shape_id, ply, mat_id in shapes:
        lines += [
            f'    <shape type="ply" id="{shape_id}">',
            f'        <string name="filename" value="{ply}"/>',
            f'        <ref id="{mat_id}" name="bsdf"/>',
            '    </shape>',
        ]
    lines.append("</scene>")
    path.write_text("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(SEED)
    half = AREA_SIDE / 2
    n = int(round(AREA_SIDE / GRID_STEP)) + 1
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)

    terrains = {
        "flat": np.zeros((n, n)),
        "irregular": irregular_terrain(n, rng),
    }

    cx, cy = BUILDING_CENTER
    bx0, bx1 = cx - BUILDING_FOOTPRINT[0] / 2, cx + BUILDING_FOOTPRINT[0] / 2
    by0, by1 = cy - BUILDING_FOOTPRINT[1] / 2, cy + BUILDING_FOOTPRINT[1] / 2
    X, Y = np.meshgrid(x, y)
    under = (X >= bx0 - GRID_STEP) & (X <= bx1 + GRID_STEP) & \
            (Y >= by0 - GRID_STEP) & (Y <= by1 + GRID_STEP)
    footprint = (X >= bx0) & (X <= bx1) & (Y >= by0) & (Y <= by1)

    # same sources in every scene
    while True:
        n_total = rng.poisson(SOURCE_DENSITY * AREA_SIDE ** 2)
        sx = rng.uniform(-half, half, n_total)
        sy = rng.uniform(-half, half, n_total)
        h_agl = rng.uniform(*SOURCE_HEIGHT_AGL, n_total)
        keep = (np.abs(sx) <= half - EDGE_MARGIN) & (np.abs(sy) <= half - EDGE_MARGIN)
        near_building = (sx >= bx0 - BUILDING_MARGIN) & (sx <= bx1 + BUILDING_MARGIN) & \
                        (sy >= by0 - BUILDING_MARGIN) & (sy <= by1 + BUILDING_MARGIN)
        keep &= ~near_building
        if keep.sum() >= MIN_SOURCES:
            break
    sx, sy, h_agl = sx[keep], sy[keep], h_agl[keep]
    ids = np.arange(len(sx))
    print(f"Poisson draw: {n_total} sources, {len(sx)} kept after the margins")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for terrain_name, z in terrains.items():
        for with_building in (False, True):
            name = terrain_name + ("_building" if with_building else "")
            scene_dir = OUT_DIR / name
            (scene_dir / "meshes").mkdir(parents=True, exist_ok=True)

            v, f = terrain_mesh(x, y, z)
            write_ply(scene_dir / "meshes" / "ground.ply", v, f)
            shapes = [("ground", "meshes/ground.ply", "mat-itu-ground")]

            surface = z.copy()
            if with_building:
                z_base = z[under].min() - 0.5
                z_top = z[under].max() + BUILDING_HEIGHT
                v, f = box_mesh(bx0, bx1, by0, by1, z_base, z_top)
                write_ply(scene_dir / "meshes" / "building.ply", v, f)
                shapes.append(("building", "meshes/building.ply", "mat-itu-building"))
                surface[footprint] = z_top

            write_xml(scene_dir / f"{name}.xml", shapes)

            z_ground = ground_height(sx, sy, x, y, z)
            positions = np.column_stack([sx, sy, z_ground + h_agl])
            np.savez(scene_dir / "sources.npz", ids=ids, positions=positions,
                     height_agl=h_agl, ground_z=z_ground)
            np.savez(scene_dir / "height_map.npz", x=x, y=y,
                     terrain=z, surface=surface)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(surface, origin="lower", cmap="terrain",
                           extent=[x[0], x[-1], y[0], y[-1]])
            ax.scatter(sx, sy, s=6, c="red", label=f"{len(sx)} sources")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(name)
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(im, ax=ax, label="surface height [m]")
            fig.tight_layout()
            fig.savefig(scene_dir / "topview.png", dpi=150)
            plt.close(fig)

            print(f"{name:20s} ground {z.min():+.2f} to {z.max():+.2f} m, "
                  f"surface max {surface.max():.2f} m")

    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, str, tuple))}
    config["NUM_SOURCES"] = int(len(sx))
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Scenes written to {OUT_DIR}")


if __name__ == "__main__":
    main()
