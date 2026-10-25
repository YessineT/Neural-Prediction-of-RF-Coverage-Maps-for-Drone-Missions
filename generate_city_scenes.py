"""
Random city generator for the AIRWISE project.

Builds many synthetic urban scenes (128 x 128 m) in the same format as the
student_datasets scenes, so Sionna RT, check_scenes.py and the notebook work on them.

Per scene it writes:
    <name>.xml                 Sionna / Mitsuba scene description
    meshes/ground.ply          ground surface
    meshes/buildings.ply       all buildings merged into one mesh
    height_map.npz             x, y (cell centres), terrain, surface (ground + buildings), 128 x 128
    sources.npz                candidate ground station positions (ids, positions, height_agl, ground_z)
    meta.json                  family, parameters, building statistics
and globally:
    index.json                 list of scenes with family and split (train / val / test / ood)
    config.json                parameters used
    overview.png               top views of a few scenes per family

Layout families
    in distribution (split into train / val / test):
        blocks      scattered buildings of random size, orientation and height (boxes, L shapes, podium + tower)
        grid        street grid (Manhattan style), rotated, blocks cut into lots
        downtown    dense centre with tall buildings, lower towards the edges
    out of distribution (only used to test generalisation, split = "ood"):
        canyon      long parallel buildings with narrow streets between them
        courtyard   ring / U shaped buildings around courtyards
        hilly       "blocks" on hilly terrain

Usage (from this folder):
    python generate_city_scenes.py                       # default: 450 id scenes + 90 ood scenes
    python generate_city_scenes.py --n-id 20 --n-ood 5   # quick test
    python generate_city_scenes.py --verify 10           # also check mesh against height map in Sionna
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
BASE_SEED = 2026
AREA_SIDE = 128.0            # m, scene is AREA_SIDE x AREA_SIDE, centred on (0, 0)
CELL = 1.0                   # m, pixel size of the height map (and of the radio maps)

HEIGHT_RANGE = (6.0, 55.0)   # m, building height limits
MIN_GAP = 3.0                # m, minimum space between buildings in "blocks"

N_STATIONS = 24              # candidate ground stations saved per scene
STATION_HEIGHT_AGL = (1.0, 10.0)   # m above ground (same as the student datasets)
EDGE_MARGIN = 5.0            # m, stations stay away from the scene border
BUILDING_MARGIN = 2.0        # m, and from the buildings

GROUND_MATERIAL = "medium_dry_ground"   # ITU, valid 1 to 10 GHz
BUILDING_MATERIAL = "concrete"          # ITU

MIN_BUILDINGS = dict(blocks=3, grid=3, downtown=3, canyon=2, courtyard=1, hilly=3)

ID_FAMILIES = ["blocks", "grid", "downtown"]
OOD_FAMILIES = ["canyon", "courtyard", "hilly"]
ID_SPLIT = (0.8, 0.1, 0.1)   # train, val, test fractions of each in distribution family

HALF = AREA_SIDE / 2
N_CELLS = int(round(AREA_SIDE / CELL))


# ---------------------------------------------------------------- geometry helpers
def rot(ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s], [s, c]])


def rect_corners(cx, cy, w, d, ang):
    local = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    return local @ rot(ang).T + [cx, cy]


def inside_rect(px, py, p, grow=0.0):
    """True where the points (px, py) are inside the rotated rectangle of part p."""
    c, s = np.cos(p["ang"]), np.sin(p["ang"])
    dx, dy = px - p["cx"], py - p["cy"]
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (np.abs(u) <= p["w"] / 2 + grow) & (np.abs(v) <= p["d"] / 2 + grow)


def part(cx, cy, w, d, ang, h, h0=None):
    """One box. h = roof height above the local ground, h0 = bottom height (None = on the ground)."""
    return dict(cx=float(cx), cy=float(cy), w=float(w), d=float(d), ang=float(ang),
                h=float(h), h0=None if h0 is None else float(h0))


def local_to_world(lx, ly, cx, cy, ang, sx=1, sy=1):
    x, y = rot(ang) @ np.array([lx * sx, ly * sy])
    return cx + x, cy + y


def fits(building, margin=1.0):
    lim = HALF - margin
    for p in building:
        if np.abs(rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"])).max() > lim:
            return False
    return True


# ---------------------------------------------------------------- building shapes
# A building is a list of boxes ("parts") that share the same ground reference.
def box_building(cx, cy, w, d, ang, h):
    return [part(cx, cy, w, d, ang, h)]


def l_building(rng, cx, cy, w, d, ang, h):
    """L shape made of two touching boxes, the second one can be lower."""
    cw, cd = rng.uniform(0.35, 0.6) * w, rng.uniform(0.35, 0.6) * d
    sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
    y0 = -d / 2 + (d - cd)
    a = local_to_world(0, -d / 2 + (d - cd) / 2, cx, cy, ang, sx, sy)
    b = local_to_world(-w / 2 + (w - cw) / 2, (y0 + d / 2) / 2, cx, cy, ang, sx, sy)
    return [part(*a, w, d - cd, ang, h),
            part(*b, w - cw, cd, ang, h * rng.uniform(0.6, 1.0))]


def podium_tower(rng, cx, cy, w, d, ang, h):
    """Low wide base with a taller narrower tower on top."""
    h_p = rng.uniform(0.3, 0.55) * h
    tw, td = rng.uniform(0.35, 0.6) * w, rng.uniform(0.35, 0.6) * d
    ox = rng.uniform(-1, 1) * (w - tw) / 2 * 0.8
    oy = rng.uniform(-1, 1) * (d - td) / 2 * 0.8
    t = local_to_world(ox, oy, cx, cy, ang)
    return [part(cx, cy, w, d, ang, h_p), part(*t, tw, td, ang, h, h0=h_p)]


def ring_building(rng, cx, cy, w, d, ang, h, t):
    """Four bars around a courtyard, sometimes one bar missing (U shape)."""
    bars = [((0, -d / 2 + t / 2), (w, t)), ((0, d / 2 - t / 2), (w, t)),
            ((-w / 2 + t / 2, 0), (t, d - 2 * t)), ((w / 2 - t / 2, 0), (t, d - 2 * t))]
    if rng.random() < 0.4:
        bars.pop(rng.integers(4))
    return [part(*local_to_world(lx, ly, cx, cy, ang), bw, bd, ang, h * rng.uniform(0.85, 1.0))
            for (lx, ly), (bw, bd) in bars]


def random_shape(rng, cx, cy, w, d, ang, h, allow_complex=True):
    r = rng.random()
    if allow_complex and min(w, d) >= 14 and r < 0.15:
        return l_building(rng, cx, cy, w, d, ang, h)
    if allow_complex and min(w, d) >= 14 and r < 0.30:
        return podium_tower(rng, cx, cy, w, d, ang, h)
    return box_building(cx, cy, w, d, ang, h)


def random_height(rng, median=16.0):
    return float(np.clip(rng.lognormal(np.log(median), 0.5), *HEIGHT_RANGE))


# ---------------------------------------------------------------- layout families
def layout_scatter(rng, height_fn=None, coverage=None, ang_mode="random"):
    """Random non overlapping buildings. Used by blocks, downtown and hilly."""
    coverage = coverage if coverage is not None else rng.uniform(0.08, 0.35)
    theta = rng.uniform(0, np.pi / 2)
    buildings, circles, area = [], [], 0.0
    for _ in range(3000):
        if area >= coverage * AREA_SIDE ** 2:
            break
        w, d = rng.uniform(8, 28), rng.uniform(8, 28)
        ang = rng.uniform(0, np.pi) if ang_mode == "random" else theta + rng.normal(0, 0.12)
        r = np.hypot(w, d) / 2
        cx, cy = rng.uniform(-HALF + r, HALF - r, 2)
        if any(np.hypot(cx - x, cy - y) < r + r2 + MIN_GAP for x, y, r2 in circles):
            continue
        h = height_fn(rng, cx, cy) if height_fn else random_height(rng)
        b = random_shape(rng, cx, cy, w, d, ang, h)
        if not fits(b):
            continue
        buildings.append(b)
        circles.append((cx, cy, r))
        area += w * d
    return buildings, dict(coverage_target=float(coverage), theta=float(theta))


def layout_blocks(rng):
    return layout_scatter(rng)


def layout_downtown(rng):
    centre = rng.uniform(-25, 25, 2)
    h_max, sigma = rng.uniform(30, 55), rng.uniform(25, 50)

    def height_fn(rng_, cx, cy):
        r2 = (cx - centre[0]) ** 2 + (cy - centre[1]) ** 2
        return float(np.clip(h_max * np.exp(-r2 / (2 * sigma ** 2)) * rng_.uniform(0.6, 1.2),
                             *HEIGHT_RANGE))

    b, info = layout_scatter(rng, height_fn, coverage=rng.uniform(0.25, 0.45), ang_mode="aligned")
    info.update(centre=centre.tolist(), h_max=float(h_max), sigma=float(sigma))
    return b, info


def layout_grid(rng):
    """Street grid, rotated, every block cut into 1 to 4 lots."""
    street = rng.uniform(8, 16)
    bw, bd = rng.uniform(26, 48, 2)
    theta = rng.uniform(-np.pi / 4, np.pi / 4)
    ox, oy = rng.uniform(0, bw + street), rng.uniform(0, bd + street)
    n = int(HALF * 1.6 / min(bw, bd)) + 2
    buildings = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            bx, by = i * (bw + street) + ox, j * (bd + street) + oy
            if np.hypot(bx, by) > HALF * 1.5:
                continue
            block_h = random_height(rng, 18.0)
            gap = rng.uniform(1.0, 2.5)
            u = rng.random()
            nx, ny = (1, 1) if u < 0.35 else ((2, 1) if bw >= bd else (1, 2)) if u < 0.75 else (2, 2)
            lw, ld = bw / nx, bd / ny
            for a in range(nx):
                for b in range(ny):
                    lx = -bw / 2 + lw * (a + 0.5)
                    ly = -bd / 2 + ld * (b + 0.5)
                    wx, wy = rot(theta) @ np.array([bx + lx, by + ly])
                    h = float(np.clip(block_h * rng.uniform(0.75, 1.25), *HEIGHT_RANGE))
                    bld = random_shape(rng, wx, wy, lw - gap, ld - gap, theta, h)
                    if fits(bld):
                        buildings.append(bld)
    return buildings, dict(street=float(street), block=[float(bw), float(bd)], theta=float(theta))


def layout_canyon(rng):
    """Long parallel buildings with narrow streets: not seen in training."""
    theta = rng.uniform(0, np.pi)
    n_rows = int(rng.integers(2, 5))
    street, depth = rng.uniform(10, 22), rng.uniform(10, 20)
    buildings = []
    for j in range(n_rows):
        ly = (j - (n_rows - 1) / 2) * (depth + street) + rng.uniform(-3, 3)
        row_h = rng.uniform(12, 45)
        x = -90.0
        while x < 90:
            seg = rng.uniform(25, 70)
            h = float(np.clip(row_h * rng.uniform(0.8, 1.2), *HEIGHT_RANGE))
            wx, wy = rot(theta) @ np.array([x + seg / 2, ly])
            bld = box_building(wx, wy, seg, depth, theta, h)
            if fits(bld):
                buildings.append(bld)
            x += seg + rng.uniform(4, 14)
    return buildings, dict(theta=float(theta), rows=n_rows, street=float(street), depth=float(depth))


def layout_courtyard(rng):
    """Ring and U shaped buildings: not seen in training."""
    theta = rng.uniform(-0.3, 0.3) + rng.choice([0.0, np.pi / 4])
    nx_, ny_ = rng.integers(1, 3, 2)
    street, t = rng.uniform(10, 16), rng.uniform(9, 14)
    # largest ring that still fits in the area after the rotation
    spread = abs(np.cos(theta)) + abs(np.sin(theta))
    n_max = int(max(nx_, ny_))
    max_size = (2 * (HALF - 1) / spread - (n_max - 1) * street) / n_max
    size = rng.uniform(min(35.0, max_size), min(60.0, max_size))
    buildings = []
    for i in range(nx_):
        for j in range(ny_):
            lx = (i - (nx_ - 1) / 2) * (size + street)
            ly = (j - (ny_ - 1) / 2) * (size + street)
            wx, wy = rot(theta) @ np.array([lx, ly])
            bld = ring_building(rng, wx, wy, size, size * rng.uniform(0.8, 1.0), theta,
                                rng.uniform(12, 40), t)
            if fits(bld):
                buildings.append(bld)
    return buildings, dict(theta=float(theta), size=float(size), thickness=float(t))


def layout_hilly(rng):
    return layout_scatter(rng)


LAYOUTS = dict(blocks=layout_blocks, grid=layout_grid, downtown=layout_downtown,
               canyon=layout_canyon, courtyard=layout_courtyard, hilly=layout_hilly)


# ---------------------------------------------------------------- terrain
def hilly_terrain(rng, n_vertices):
    amp, scale = rng.uniform(2.0, 5.0), rng.uniform(15.0, 35.0)
    noise = rng.standard_normal((n_vertices, n_vertices))
    k = np.fft.fftfreq(n_vertices, d=CELL)
    KX, KY = np.meshgrid(k, k)
    lowpass = np.exp(-0.5 * (np.pi * np.hypot(KX, KY) * scale) ** 2)
    z = np.real(np.fft.ifft2(np.fft.fft2(noise) * lowpass))
    z -= z.mean()
    return z * amp / np.abs(z).max(), dict(amplitude=float(amp), scale=float(scale))


def ground_height(px, py, vx, vy, z):
    """Terrain height at (px, py), exact on the mesh triangles. vx, vy are the vertex coordinates."""
    fx = (px - vx[0]) / CELL
    fy = (py - vy[0]) / CELL
    ix = np.clip(np.floor(fx).astype(int), 0, len(vx) - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, len(vy) - 2)
    fx, fy = fx - ix, fy - iy
    z00, z10, z01, z11 = z[iy, ix], z[iy, ix + 1], z[iy + 1, ix], z[iy + 1, ix + 1]
    lower = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    upper = z00 + fy * (z01 - z00) + fx * (z11 - z01)
    return np.where(fx >= fy, lower, upper)


# ---------------------------------------------------------------- meshes and files
def write_ply(path, vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    records = np.empty(len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    records["n"] = 3
    records["idx"] = faces
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())
        f.write(records.tobytes())


BOX_FACES = np.array([[4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [1, 2, 6], [1, 6, 5]])


def box_mesh(corners_xy, z0, z1):
    v = np.vstack([np.column_stack([corners_xy, np.full(4, z0)]),
                   np.column_stack([corners_xy, np.full(4, z1)])])
    return v, BOX_FACES


def ground_mesh(vx, vy, z, flat):
    if flat:    # two triangles are enough
        h = HALF
        v = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
        return v, np.array([[0, 1, 2], [0, 2, 3]])
    nx, ny = len(vx), len(vy)
    X, Y = np.meshgrid(vx, vy)
    v = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    iy, ix = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    v00 = (iy * nx + ix).ravel()
    v10, v01 = v00 + 1, v00 + nx
    v11 = v01 + 1
    return v, np.concatenate([np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])])


def write_xml(path):
    lines = ['<scene version="2.1.0">']
    for mat_id, itu in [("mat-itu-ground", GROUND_MATERIAL), ("mat-itu-building", BUILDING_MATERIAL)]:
        lines += [f'    <bsdf type="itu-radio-material" id="{mat_id}">',
                  f'        <string name="type" value="{itu}"/>',
                  '        <float name="thickness" value="0.2"/>', '    </bsdf>']
    for shape_id, ply, mat_id in [("ground", "meshes/ground.ply", "mat-itu-ground"),
                                  ("buildings", "meshes/buildings.ply", "mat-itu-building")]:
        lines += [f'    <shape type="ply" id="{shape_id}">',
                  f'        <string name="filename" value="{ply}"/>',
                  f'        <ref id="{mat_id}" name="bsdf"/>', '    </shape>']
    lines.append("</scene>")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- one scene
def build_scene(out_dir, name, family, rng):
    scene_dir = out_dir / name
    (scene_dir / "meshes").mkdir(parents=True, exist_ok=True)

    for _ in range(20):     # draw again if the layout came out too empty
        buildings, layout_info = LAYOUTS[family](rng)
        if len(buildings) >= MIN_BUILDINGS[family]:
            break

    # terrain on the vertex grid (mesh) and on the cell centres (height map)
    vx = vy = np.linspace(-HALF, HALF, N_CELLS + 1)
    if family == "hilly":
        terrain_v, terrain_info = hilly_terrain(rng, N_CELLS + 1)
    else:
        terrain_v, terrain_info = np.zeros((N_CELLS + 1, N_CELLS + 1)), {}
    flat = family != "hilly"
    xc = yc = (np.arange(N_CELLS) + 0.5) * CELL - HALF
    XC, YC = np.meshgrid(xc, yc)
    terrain_c = ground_height(XC, YC, vx, vy, terrain_v)
    VX, VY = np.meshgrid(vx, vy)

    surface = terrain_c.copy()
    all_vertices, all_faces, n_vert = [], [], 0
    for b in buildings:
        # ground reference of the whole building, so stacked boxes always touch
        near = np.zeros_like(VX, dtype=bool)
        for p in b:
            near |= inside_rect(VX, VY, p, grow=1.0)
        g = terrain_v[near] if near.any() else np.array([0.0])
        g_min, g_max = g.min(), g.max()
        for p in b:
            z0 = g_min - 0.5 if p["h0"] is None else g_max + p["h0"]
            z1 = g_max + p["h"]
            v, f = box_mesh(rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"]), z0, z1)
            all_vertices.append(v)
            all_faces.append(f + n_vert)
            n_vert += len(v)
            mask = inside_rect(XC, YC, p)
            surface[mask] = np.maximum(surface[mask], z1)

    write_ply(scene_dir / "meshes" / "ground.ply", *ground_mesh(vx, vy, terrain_v, flat))
    if all_vertices:
        write_ply(scene_dir / "meshes" / "buildings.ply",
                  np.vstack(all_vertices), np.vstack(all_faces))
    else:   # keep the file valid, one tiny box far below the ground
        write_ply(scene_dir / "meshes" / "buildings.ply",
                  *box_mesh(rect_corners(0, 0, 1, 1, 0), -50, -49))
    write_xml(scene_dir / f"{name}.xml")
    np.savez(scene_dir / "height_map.npz", x=xc, y=yc, terrain=terrain_c, surface=surface)

    # candidate ground stations: away from the border and from the buildings
    parts = [p for b in buildings for p in b]
    cand = rng.uniform(-HALF + EDGE_MARGIN, HALF - EDGE_MARGIN, (4000, 2))
    ok = np.ones(len(cand), dtype=bool)
    for p in parts:
        ok &= ~inside_rect(cand[:, 0], cand[:, 1], p, grow=BUILDING_MARGIN)
    cand = cand[ok][:N_STATIONS]
    h_agl = rng.uniform(*STATION_HEIGHT_AGL, len(cand))
    z_ground = ground_height(cand[:, 0], cand[:, 1], vx, vy, terrain_v)
    np.savez(scene_dir / "sources.npz", ids=np.arange(len(cand)),
             positions=np.column_stack([cand, z_ground + h_agl]),
             height_agl=h_agl, ground_z=z_ground)

    built = surface - terrain_c > 0.5
    meta = dict(name=name, family=family, num_buildings=len(buildings), num_parts=len(parts),
                built_fraction=float(built.mean()),
                max_height=float((surface - terrain_c).max()),
                mean_building_height=float((surface - terrain_c)[built].mean()) if built.any() else 0.0,
                num_stations=int(len(cand)), layout=layout_info, terrain=terrain_info)
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta, surface - terrain_c, cand


# ---------------------------------------------------------------- verification with Sionna
def verify(out_dir, names):
    """Cast rays from above through the Sionna scene and compare with height_map.npz."""
    import mitsuba as mi
    from sionna.rt import load_scene
    print("\nChecking mesh against height map (ray cast from above):")
    for name in names:
        scene = load_scene(str(out_dir / name / f"{name}.xml"))
        hm = np.load(out_dir / name / "height_map.npz")
        XC, YC = np.meshgrid(hm["x"], hm["y"])
        n = XC.size
        top = 500.0
        ray = mi.Ray3f(mi.Point3f(XC.ravel().astype("f4"), YC.ravel().astype("f4"),
                                  np.full(n, top, "f4")), mi.Vector3f(0, 0, -1))
        t = np.array(scene.mi_scene.ray_intersect(ray).t)
        z = top - t
        err = np.abs(z.reshape(XC.shape) - hm["surface"])
        print(f"  {name:16s} max error {err.max():.3f} m, cells off by more than 5 cm: "
              f"{int((err > 0.05).sum())} of {n}")


# ---------------------------------------------------------------- overview figure
def overview(out_dir, index, per_family=4):
    families = ID_FAMILIES + OOD_FAMILIES
    fig, axes = plt.subplots(len(families), per_family, figsize=(2.6 * per_family, 2.6 * len(families)),
                             constrained_layout=True)
    for r, fam in enumerate(families):
        names = [e["name"] for e in index if e["family"] == fam][:per_family]
        for c in range(per_family):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if c >= len(names):
                ax.axis("off")
                continue
            hm = np.load(out_dir / names[c] / "height_map.npz")
            src = np.load(out_dir / names[c] / "sources.npz")["positions"]
            h = hm["surface"] - hm["terrain"]
            ax.imshow(np.where(h > 0.5, h, np.nan), origin="lower", cmap="magma", vmin=0, vmax=55,
                      extent=[-HALF, HALF, -HALF, HALF])
            ax.set_facecolor("#cfe3cf" if fam != "hilly" else "#d9d2b6")
            ax.scatter(src[:, 0], src[:, 1], s=4, c="tab:blue")
            if c == 0:
                ax.set_ylabel(fam, fontsize=11)
    fig.suptitle("Generated scenes: building height in m (colour), candidate stations (blue)")
    fig.savefig(out_dir / "overview.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "scenes_city"))
    ap.add_argument("--n-id", type=int, default=150, help="scenes per in distribution family")
    ap.add_argument("--n-ood", type=int, default=30, help="scenes per out of distribution family")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--verify", type=int, default=0, help="check this many scenes in Sionna")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    plan = [(f, args.n_id) for f in ID_FAMILIES] + [(f, args.n_ood) for f in OOD_FAMILIES]
    for fam_id, (family, n) in enumerate(plan):
        for i in range(n):
            name = f"{family}_{i:04d}"
            if family in OOD_FAMILIES:
                split = "ood"
            else:
                f = i / n
                split = "train" if f < ID_SPLIT[0] else "val" if f < ID_SPLIT[0] + ID_SPLIT[1] else "test"
            rng = np.random.default_rng([args.seed, fam_id, i])
            meta, _, _ = build_scene(out_dir, name, family, rng)
            index.append(dict(name=name, family=family, split=split,
                              num_buildings=meta["num_buildings"],
                              built_fraction=round(meta["built_fraction"], 3),
                              max_height=round(meta["max_height"], 1)))
        sub = [e for e in index if e["family"] == family]
        print(f"{family:10s} {n:4d} scenes, mean built fraction "
              f"{np.mean([e['built_fraction'] for e in sub]):.2f}, "
              f"mean max height {np.mean([e['max_height'] for e in sub]):.1f} m")

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    config = {k: v for k, v in globals().items()
              if k.isupper() and k != "LAYOUTS" and isinstance(v, (int, float, str, tuple, list, dict))}
    config["SEED"] = args.seed
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    overview(out_dir, index)
    counts = {s: sum(e["split"] == s for e in index) for s in ["train", "val", "test", "ood"]}
    print(f"\n{len(index)} scenes written to {out_dir}  {counts}")

    if args.verify:
        rng = np.random.default_rng(0)
        pick = [index[i]["name"] for i in rng.choice(len(index), min(args.verify, len(index)), replace=False)]
        verify(out_dir, pick)


if __name__ == "__main__":
    main()