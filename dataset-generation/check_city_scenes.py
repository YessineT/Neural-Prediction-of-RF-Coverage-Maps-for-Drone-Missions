"""
Checks the scenes made by generate_city_scenes.py and the dataset made by make_dataset.py,
so that every scene is safe to give to a ray tracer (Sionna) and gives correct training data.

Every check prints PASS, WARN or FAIL:
    FAIL  the scene (or dataset) would give wrong ray tracing results or wrong training data
    WARN  worth a look, but does not break anything
The script ends with exit code 1 if anything FAILED, 0 otherwise.

Scene checks (no Sionna needed, run in parallel over --workers processes)
    files (incl. the exact building geometry in meta.json), NaN / inf, height map (incl. occupancy), not empty,
        XML, ground mesh
    building meshes: finite, no zero-area faces, inside the area, closed and consistently oriented,
        outward faces, one closed mesh per building, not floating
    coplanar overlapping faces between different meshes (WARN: z-fighting, false diffraction edges)
    exact geometry from meta.json: gives exactly the saved height map, every building mesh is exactly its
        meta.json building (bounding box, floor, every roof level, volume), no overlap, gap between buildings
        (FAIL below MIN_SLOT = 2 pixels, WARN below the generator's MIN_GAP), thin parts (WARN)
    roofs clear of the drone altitudes (FAIL closer than 0.5 m, where the validity mask
        "surface < altitude - 0.5" becomes ambiguous; WARN closer than the generator's ROOF_CLEARANCE)
    closed pockets of free space too small to matter (WARN: rays get trapped in tiny courtyards)
    stations: count, unique, edge margin, height above ground, ground_z, exact distance to every wall,
        clearance below the lowest drone altitude, spacing (WARN), stored wall distance and class (WARN)
Global checks
    index.json, duplicates, reproducibility (same seed -> identical files, in parallel),
    train / val / test drawn from the same distribution (two sample KS test, WARN)
Sionna checks (skipped with --no-sionna, on the GPU through Mitsuba / Dr.Jit)
    loads, radio materials valid AT THE DATASET FREQUENCY, mesh = height map (vertical rays through every pixel)
    and walls (horizontal rays along every row and column, in 4 directions, at 6 heights),
    stations outside every solid, and for every drone altitude the validity mask computed
    from the mesh (ray cast upwards: hitting the back of a face means "inside") equals the height map mask
    used for training
Physics checks on ray traced radio maps (a few scenes per family, every scene with --full; by default the first
and last station of a scene, the stations the dataset will use with --stations dataset, every station with
--stations all; every check on every map with --exhaustive)
    radio map cells line up with the height map pixels (rm.cell_centers), gain finite and >= 0,
    free space right above the station, energy bound, shadowing with exact line of sight (Mitsuba rays),
    no real signal inside buildings (FAIL; tiny numerical leaks through edges are a WARN),
    holes in line of sight (WARN: too few rays), Monte Carlo noise N vs 2N rays (WARN; the solver's seed does
    not change the ray directions), convergence in max_depth (WARN), uninformative maps (WARN)
    The ray tracing settings (frequency, rays, depth, diffraction, diffuse reflections, stations) are read from
    dataset/settings.json when it exists, else common.RT_DEFAULTS (the defaults of make_dataset.py), so the checks
    trace exactly the maps of the dataset. At the default 1e8 rays a map takes ~0.36 s on an RTX 4070 Laptop.
Dataset checks (if the dataset/ folder exists)
    shapes, no NaN, value range, splits, stale scenes, timings, free space above the station,
    station below the plane, no signal inside buildings, uninformative maps (WARN)

Outputs
    check_results.json   in the scenes folder: status, hash and problems of every scene.
                         make_dataset.py skips scenes that FAILED and never picks a station listed in
                         bad_stations (a station whose ray traced map failed a check; the rest of the scene is
                         fine). With --only, only those entries are updated.
    check_report.png     statistics per family (path set with --report)

Usage (from this folder):
    python check_city_scenes.py                               # everything, physics on a few scenes
    python check_city_scenes.py --full --stations dataset     # before make_dataset.py: every map it will use
    python check_city_scenes.py --full                        # physics on every scene (first + last station)
    python check_city_scenes.py --full --stations all         # physics on every map of every scene (hours)
    python check_city_scenes.py --exhaustive                  # every check on every possible map (a day or more)
    python check_city_scenes.py --full --samples 2000000 --max-depth 5 --no-diffraction --no-diffuse   # fast
    python check_city_scenes.py --no-sionna                   # only the checks that need no Sionna
    python check_city_scenes.py --only grid_0002 hilly_0010   # re-check single scenes
    python check_city_scenes.py --workers 1                   # no process pool (one CPU process)
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import generate_city_scenes as gen
from common import (DATASET_DEFAULTS, RT_DEFAULTS, load_radio_scene, parallel_map, pick_stations, radio_map,
                    scene_hash)

ROOT = Path(__file__).resolve().parent

# ---- tolerances and thresholds
MIN_TRI_AREA = 1e-6          # m^2, smaller triangles count as degenerate
SNAP = 1e-4                  # m, vertices closer than this are the same point (watertight test)
GROUND_TOL = 0.5             # m, allowed difference between ground mesh / ground_z and terrain map
MIN_SLOT = 2.0 * gen.CELL    # m, a gap between two buildings below this cannot be seen in the height map
MIN_PART_SIDE = 3.0          # m, thinner building parts: WARN
ROOF_HARD_CLEARANCE = 0.5    # m, the validity mask is surface < altitude - 0.5
MIN_POCKET_AREA = 32.0       # m^2, closed pockets of free space smaller than this: WARN
FS_WINDOW_DB = (-3.0, 6.0)   # gain above the station relative to free space
ENERGY_MARGIN_DB = 10.0      # gain must not exceed free space by more than this ...
ENERGY_MAX_FRACTION = 0.005  # ... in more than this fraction of the valid cells
SHADOW_MIN_DROP_DB = 3.0     # median gain of shadowed cells must be at least this much lower than LoS cells
SHADOW_MIN_CELLS = 200       # skip the shadow test if fewer LoS / shadowed cells than this
WALL_MAX_WRONG = 0.002       # share of horizontal rays that may disagree with the height map: FAIL above,
                             # WARN for any (all 540 generated scenes have none; a building moved by 0.3 m gives ~1%)
WALL_LEVELS = (2.0, 6.0, 12.0, 20.0, 30.0, 45.0)   # m above the lowest ground, heights of the horizontal rays
GEOM_TOL = 2e-3              # m, mesh (float32) against meta.json geometry: bounding box and roof heights
GEOM_REL_TOL = 1e-4          # relative tolerance for floor area, roof areas and volume
MASK_MAX_MISMATCH = 0.001    # share of cells where mesh and height map disagree on validity: FAIL above (WARN > 0)
HOLE_MAX_FRACTION = 0.001    # share of valid line of sight cells without any ray: WARN above
LEAK_MAX_POWER = 1e-4        # power inside buildings (inner cells) as a share of the power of the valid cells:
                             # FAIL above, WARN for any. Measured at 1e8 rays: rays slipping numerically through
                             # edges give 2e-9 to 2e-8 (and reach more cells the more rays are traced, so the
                             # number of cells is no criterion); one missing wall panel gives 2e-2 to 1.5e-1
NOISE_MAX_LOS_DB = 0.5       # median |gain difference| N vs 2N rays, line of sight cells: WARN above
                             # (the seed of RadioMapSolver does not change the ray directions, so N vs 2N is used)
NOISE_MAX_NLOS_DB = 2.0      # same for shadowed cells that get signal
DEPTH_EXTRA = 3              # max_depth is raised by this much for the convergence test
DEPTH_MAX_DB = 0.5           # median |gain difference| after raising max_depth: WARN above
DEPTH_MAX_FLIP = 0.01        # share of valid cells that switch between signal and no signal: WARN above
TRIVIAL_FRACTION = 0.95      # maps with more than this share of valid cells in LoS / without signal: WARN
MIN_VALID_FRACTION = 0.05    # maps with fewer valid cells than this: WARN
KS_MIN_P = 0.01              # train / val / test distributions differ if p is below this: WARN
FLOOR_DB = -150.0            # "no signal" value used in the dataset
DEFAULT_PHYSICS = dict(RT_DEFAULTS)   # ray tracing settings of the physics checks unless dataset/settings.json exists

# ---- what a scene folder must contain (written by generate_city_scenes.py)
SCENE_FILES = ("{name}.xml", "meshes/ground.ply", "meshes/buildings.ply", "height_map.npz", "sources.npz",
               "meta.json")
HM_KEYS = ("x", "y", "terrain", "surface", "occupancy")
SRC_KEYS = ("positions", "ground_z", "wall_distance", "station_class")
META_KEYS = ("num_buildings", "num_meshes", "built_fraction", "buildings")
CFG_KEYS = ("SEED", "DRONE_ALTITUDES", "MIN_GAP", "ROOF_CLEARANCE", "PLANE_CLEARANCE", "STATION_MIN_SPACING",
            "BUILDING_MARGIN", "EDGE_MARGIN", "STATION_HEIGHT_AGL", "STATION_CLASSES")


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self):
        self.counts = {"pass": 0, "warn": 0, "fail": 0}
        self.scenes = {}        # name -> {"fail": [...], "warn": [...], "station_fail": [...]}
        self.bad_stations = {}  # name -> station ids whose ray traced maps FAILED (the scene itself is fine)

    def line(self, level, text):
        self.counts[level] += 1
        print(f"{level.upper():5s} {text}", flush=True)

    def remember(self, name, level, text):
        self.scenes.setdefault(name, {"fail": [], "warn": []})[level].append(text)

    def check(self, what, problems, total, level="fail", info=""):
        """problems: list of (scene, detail) or (scene, detail, station). One line for the whole check,
        problems kept per scene. A FAIL tied to one station only marks that station as bad."""
        for pr in problems:
            name, detail = pr[0], pr[1]
            station = pr[2] if len(pr) > 2 else None
            text = what + (f" ({detail})" if detail else "")
            if station is not None and level == "fail":
                self.bad_stations.setdefault(name, set()).add(int(station))
                self.scenes.setdefault(name, {"fail": [], "warn": []}).setdefault("station_fail", []).append(text)
            else:
                self.remember(name, level, text)
        info = f", {info}" if info else ""
        if not problems:
            self.line("pass", f"{what} ({total} checked{info})")
        else:
            n_sc = len({p[0] for p in problems})
            first = [f"{p[0]} ({p[1]})" if p[1] else p[0] for p in problems[:4]]
            self.line(level, f"{what}: {n_sc} of {total} {'failed' if level == 'fail' else 'flagged'}{info}; "
                             f"first: {first}")

    def single(self, ok, text, level="fail", owner="(global)"):
        if ok:
            self.line("pass", text)
        else:
            self.line(level, text)
            self.remember(owner, level, text)


R = Report()


class Problems:
    """Collects problems of many scenes for a fixed, ordered list of checks."""

    def __init__(self, titles):
        self.titles = titles                      # key -> (title, level)
        self.found = {k: [] for k in titles}
        self.info = {}

    def add(self, key, name, detail="", station=None):
        self.found[key].append((name, detail) if station is None else (name, detail, station))

    def emit(self, total):
        for k, (title, level) in self.titles.items():
            R.check(title, self.found[k], total, level, self.info.get(k, ""))


# ---------------------------------------------------------------- small helpers
def load_npz(path):
    """Read a whole .npz into a dict and close the file (Windows cannot delete open files)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def cell_index(x, y):
    ix = np.clip(((np.asarray(x) + gen.HALF) // gen.CELL).astype(int), 0, gen.N_CELLS - 1)
    iy = np.clip(((np.asarray(y) + gen.HALF) // gen.CELL).astype(int), 0, gen.N_CELLS - 1)
    return ix, iy


def erode(mask):
    """True where the cell and its 8 neighbours are all True (outside the map counts as False).
    Works on one map (H, W) or on a stack of maps (..., H, W), map by map."""
    square = np.ones((1,) * (mask.ndim - 2) + (3, 3), bool)
    return ndimage.binary_erosion(mask, structure=square, border_value=0)


def ks_test(a, b):
    """Two sample Kolmogorov-Smirnov statistic and asymptotic p value."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    v = np.concatenate([a, b])
    d = float(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)).max())
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    if lam < 1e-3:
        return d, 1.0
    k = np.arange(1, 101)
    return d, float(np.clip(2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))


# ---------------------------------------------------------------- PLY reading
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
              "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def read_ply(path):
    """Binary little endian triangle PLY. The header is parsed; anything unexpected raises ValueError."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt, elements = None, []  # element = [name, count, [property tokens]]
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header not terminated")
            parts = line.decode("ascii").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append([parts[1], int(parts[2]), []])
            elif parts[0] == "property":
                elements[-1][2].append(parts[1:])
            elif parts[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt}")
        data, list_names = {}, {}
        for name, count, props in elements:
            fields = []
            for p in props:
                if p[0] == "list":
                    fields += [(p[3] + "_n", _PLY_TYPES[p[1]]), (p[3], _PLY_TYPES[p[2]], (3,))]
                    list_names.setdefault(name, p[3])
                else:
                    fields.append((p[1], _PLY_TYPES[p[0]]))
            dt = np.dtype(fields)
            arr = np.frombuffer(f.read(dt.itemsize * count), dtype=dt)
            if len(arr) != count:
                raise ValueError(f"truncated PLY ({name})")
            data[name] = arr
    vert, face = data["vertex"], data["face"]
    v = np.stack([vert[c] for c in ("x", "y", "z")], axis=1).astype(float)
    ln = list_names["face"]
    if (face[ln + "_n"] != 3).any():
        raise ValueError("PLY has faces that are not triangles")
    idx = face[ln].astype(np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= len(v)):
        raise ValueError("PLY face index out of range")
    return v, idx


# ---------------------------------------------------------------- mesh helpers
def terrain_at(hm, x, y):
    """Bilinear interpolation of the terrain height map at (x, y)."""
    t = hm["terrain"]
    x0, y0 = hm["x"][0], hm["y"][0]
    dx, dy = hm["x"][1] - x0, hm["y"][1] - y0
    fx = np.clip((np.asarray(x, float) - x0) / dx, 0, t.shape[1] - 1)
    fy = np.clip((np.asarray(y, float) - y0) / dy, 0, t.shape[0] - 1)
    i0 = np.minimum(np.floor(fx).astype(int), t.shape[1] - 2)
    j0 = np.minimum(np.floor(fy).astype(int), t.shape[0] - 2)
    ax, ay = fx - i0, fy - j0
    return (t[j0, i0] * (1 - ax) * (1 - ay) + t[j0, i0 + 1] * ax * (1 - ay)
            + t[j0 + 1, i0] * (1 - ax) * ay + t[j0 + 1, i0 + 1] * ax * ay)


def triangle_geometry(v, f):
    """Triangles (F x 3 x 3), their cross products (normal * 2 * area) and areas."""
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return tri, cr, 0.5 * np.linalg.norm(cr, axis=1)


def unit_z(cr):
    """z component of the unit face normals."""
    return cr[:, 2] / np.maximum(np.linalg.norm(cr, axis=1), 1e-12)


def connected_components(f, n_vertices):
    """Label every face with its connected component (faces sharing a vertex index)."""
    labels = np.arange(n_vertices)
    while True:
        m = labels[f].min(axis=1)
        new = labels.copy()
        np.minimum.at(new, f.ravel(), np.repeat(m, 3))
        new = new[new]
        if np.array_equal(new, labels):
            break
        labels = new
    _, comp = np.unique(labels[f[:, 0]], return_inverse=True)
    return comp.reshape(-1)


def component_bounds(tri, comp, n_comp):
    """Bounding box (lo, hi) of every connected component."""
    pts, owner = tri.reshape(-1, 3), np.repeat(comp, 3)
    lo, hi = np.full((n_comp, 3), np.inf), np.full((n_comp, 3), -np.inf)
    np.minimum.at(lo, owner, pts)
    np.maximum.at(hi, owner, pts)
    return lo, hi


def is_watertight(v, f):
    """Closed and consistently oriented: every directed edge a->b appears once and b->a also once."""
    key = np.round(v / SNAP).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1)
    g = vid[f]
    a, b = g.ravel(), g[:, [1, 2, 0]].ravel()
    n = int(vid.max()) + 1
    dk, rk = a * n + b, b * n + a
    _, cnt = np.unique(dk, return_counts=True)
    return not ((cnt > 1).any() or not np.isin(rk, dk).all())


def tri2d_overlap(A, B, eps=1e-6):
    """Separating axis test: do two 2D triangles overlap with positive area?"""
    for T in (A, B):
        for i in range(3):
            e = T[(i + 1) % 3] - T[i]
            n = np.hypot(e[0], e[1])
            if n < 1e-12:
                continue
            ax = np.array([-e[1], e[0]]) / n
            pa, pb = A @ ax, B @ ax
            if pa.max() <= pb.min() + eps or pb.max() <= pa.min() + eps:
                return False
    return True


def coplanar_overlaps(tri, cr, comp):
    """Number of face pairs from DIFFERENT meshes that lie in the same plane and overlap with positive area.
    Such faces flicker in the ray tracer (which one is hit is random) and their edges look like real edges
    to the diffraction model."""
    n = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-12)
    k = np.abs(n).argmax(1)
    n = n * np.sign(n[np.arange(len(n)), k])[:, None]        # same plane, either side -> same key
    d = np.einsum("ij,ij->i", n, tri[:, 0])
    key = np.column_stack([np.round(n * 1e3), np.round(d * 1e2)]).astype(np.int64)
    _, grp, size = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(grp.reshape(-1), kind="stable")
    start = np.cumsum(size) - size
    count = 0
    for s0, sz in zip(start[size > 1], size[size > 1]):     # only planes shared by at least two faces
        g = order[s0:s0 + sz]
        if len(np.unique(comp[g])) < 2:
            continue
        keep = [a for a in range(3) if a != k[g[0]]]
        t2 = tri[g][:, :, keep]
        lo, hi = t2.min(1), t2.max(1)
        for a in range(len(g) - 1):
            cand = np.arange(a + 1, len(g))
            cand = cand[(comp[g[cand]] != comp[g[a]]) & np.all(lo[cand] < hi[a] - 1e-6, axis=1)
                        & np.all(lo[a] < hi[cand] - 1e-6, axis=1) & (np.abs(d[g[cand]] - d[g[a]]) < 1e-3)]
            count += sum(tri2d_overlap(t2[a], t2[b]) for b in cand)
    return count


# ---------------------------------------------------------------- exact geometry (meta.json)
def scene_parts(meta):
    """World rectangles of every building from meta.json."""
    out = []
    for bi, b in enumerate(meta["buildings"]):
        bb = gen.building(b["cx"], b["cy"], b["ang"], b["rects"])
        for p, r in zip(gen.world_parts(bb), b["rects"]):
            p.update(building=bi, z_top=r["z_top"],
                     corners=gen.rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"]),
                     r=float(np.hypot(p["w"], p["d"]) / 2))
            out.append(p)
    return out


def mesh_vs_geometry(tri, cr, area, comp, lo, hi, vol, meta):
    """Compare every closed building mesh with the building of meta.json it should be: bounding box, floor area,
    roof area at every roof height, and volume. Together these pin the shape down: a wall in the wrong place,
    a wrong roof level or a missing / extra piece changes at least one of them.
    lo, hi, vol: bounding box and volume of every mesh. Returns None if everything matches, else a short
    description of the first problem."""
    nz = unit_z(cr)
    floor_m = np.bincount(comp, weights=np.where(nz < -0.999, area, 0.0), minlength=len(lo))
    roofs_m = [{} for _ in range(len(lo))]
    up = np.flatnonzero(nz > 0.999)
    for c, z, a in zip(comp[up], tri[up, 0, 2], area[up]):
        z = round(float(z), 3)
        roofs_m[c][z] = roofs_m[c].get(z, 0.0) + a
    used = set()
    for bi, b in enumerate(meta["buildings"]):
        bb = gen.building(b["cx"], b["cy"], b["ang"], b["rects"])
        corners = np.vstack([gen.rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"]) for p in gen.world_parts(bb)])
        xs, ys, top = gen.footprint_grid(b["rects"], key="z_top")
        cell_a = np.diff(ys)[:, None] * np.diff(xs)[None, :]
        solid = ~np.isnan(top)
        roofs = {}
        for z, a in zip(top[solid], cell_a[solid]):
            roofs[float(z)] = roofs.get(float(z), 0.0) + a
        lo_e = np.r_[corners.min(0), b["z_bottom"]]
        hi_e = np.r_[corners.max(0), top[solid].max()]
        floor = float(cell_a[solid].sum())
        vol_e = float(np.sum(cell_a[solid] * (top[solid] - b["z_bottom"])))
        # the mesh of this building: the one with the same bounding box
        k = int(np.argmin(np.abs(lo - lo_e).max(1) + np.abs(hi - hi_e).max(1)))
        if k in used or np.abs(lo[k] - lo_e).max() > GEOM_TOL or np.abs(hi[k] - hi_e).max() > GEOM_TOL:
            return f"building {bi}: no mesh with its bounding box"
        used.add(k)
        if abs(floor_m[k] - floor) > GEOM_REL_TOL * floor + 1e-3:
            return f"building {bi}: floor {floor_m[k]:.3f} m^2 in the mesh, {floor:.3f} m^2 expected"
        ez, mz = sorted(roofs), sorted(roofs_m[k])
        if len(ez) != len(mz) or any(abs(a - b_) > GEOM_TOL for a, b_ in zip(ez, mz)):
            return f"building {bi}: roof heights {[round(z, 2) for z in mz]} in the mesh, {[round(z, 2) for z in ez]} expected"
        for a, b_ in zip(ez, mz):
            if abs(roofs[a] - roofs_m[k][b_]) > GEOM_REL_TOL * floor + 1e-3:
                return f"building {bi}: roof at {a:.2f} m covers {roofs_m[k][b_]:.3f} m^2 in the mesh, {roofs[a]:.3f} expected"
        if abs(vol[k] - vol_e) > GEOM_REL_TOL * vol_e + 1e-2:
            return f"building {bi}: volume {vol[k]:.2f} m^3 in the mesh, {vol_e:.2f} m^3 expected"
    return None


def point_segment_distance(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def polygon_distance(A, B):
    """Distance between two convex polygons that do not overlap."""
    return min(min(point_segment_distance(A, B[i], B[(i + 1) % len(B)]).min() for i in range(len(B))),
               min(point_segment_distance(B, A[i], A[(i + 1) % len(A)]).min() for i in range(len(A))))


# ---------------------------------------------------------------- per scene checks (no Sionna)
def static_titles(cfg, alts):
    return {
        "files": ("all files present and readable (incl. the exact building geometry in meta.json)", "fail"),
        "finite": ("height maps and sources contain no NaN / inf", "fail"),
        "heightmap": ("height map valid (shape, surface >= terrain, plausible heights, occupancy)", "fail"),
        "empty": ("not empty (minimum number of buildings per family)", "fail"),
        "xml": ("XML valid (parses, .ply files exist, every shape has a material)", "fail"),
        "ground": ("ground mesh valid (faces up, no degenerate faces, covers the area, follows terrain)", "fail"),
        "b_read": ("buildings: mesh readable and finite", "fail"),
        "b_area": ("buildings: no zero-area faces", "fail"),
        "b_inside": ("buildings: inside the area", "fail"),
        "b_closed": ("buildings: closed and consistently oriented (watertight)", "fail"),
        "b_outward": ("buildings: faces point outward (positive volume)", "fail"),
        "b_count": ("buildings: number of closed meshes as in meta.json", "fail"),
        "b_floating": ("buildings: standing on the ground, not floating", "fail"),
        "coplanar": ("no coplanar overlapping faces between different meshes", "warn"),
        "g_match": ("geometry: meta.json geometry gives exactly the saved height map", "fail"),
        "g_mesh": ("geometry: every building mesh is exactly its meta.json building "
                   "(bounding box, floor, every roof level, volume)", "fail"),
        "g_overlap": ("geometry: buildings do not overlap (exact)", "fail"),
        "g_slot": (f"geometry: gap between buildings >= {MIN_SLOT:.0f} m (visible in the height map)", "fail"),
        "g_gap": (f"geometry: gap between buildings >= MIN_GAP ({cfg['MIN_GAP']:.1f} m)", "warn"),
        "g_thin": (f"geometry: building parts at least {MIN_PART_SIDE:.0f} m thick", "warn"),
        "roof_hard": (f"roofs at least {ROOF_HARD_CLEARANCE} m from the drone altitudes {list(alts)}", "fail"),
        "roof": (f"roofs at least ROOF_CLEARANCE ({cfg['ROOF_CLEARANCE']} m) from the drone altitudes", "warn"),
        "pocket": (f"no closed pockets of free space smaller than {MIN_POCKET_AREA:.0f} m^2", "warn"),
        "st_basic": ("stations valid (at least 8, unique, inside the edge margin, height above ground, "
                     "ground_z, above the surface)", "fail"),
        "st_wall": (f"stations at least {cfg['BUILDING_MARGIN']:.0f} m from every wall", "fail"),
        "st_plane": (f"stations at least {cfg['PLANE_CLEARANCE']:.0f} m below the lowest drone "
                     f"altitude ({min(alts):.0f} m)", "fail"),
        "st_spacing": (f"stations at least {cfg['STATION_MIN_SPACING']:.0f} m apart", "warn"),
        "st_labels": ("stations: stored wall distance and class are consistent", "warn"),
    }


def load_scene_files(d, name, add):
    """meta, height map and sources of a scene, or None (problem reported) if anything is missing or broken."""
    missing = [r.format(name=name) for r in SCENE_FILES if not (d / r.format(name=name)).exists()]
    if missing:
        add("files", f"missing {missing}")
        return None
    try:
        meta = json.loads((d / "meta.json").read_text())
        hm, src = load_npz(d / "height_map.npz"), load_npz(d / "sources.npz")
    except Exception as ex:
        add("files", f"{type(ex).__name__}: {ex}")
        return None
    missing = ([f"meta.json: {k}" for k in META_KEYS if k not in meta]
               + [f"height_map.npz: {k}" for k in HM_KEYS if k not in hm]
               + [f"sources.npz: {k}" for k in SRC_KEYS if k not in src])
    if missing:
        add("files", f"missing {missing} (made by an older generator, regenerate the scene)")
        return None
    if not all(np.isfinite(np.asarray(a, float)).all() for a in [*hm.values(), *src.values()]):
        add("finite")
        return None
    s, t, occ = hm["surface"], hm["terrain"], hm["occupancy"]
    built = s - t > 0.5
    if not (s.shape == (gen.N_CELLS, gen.N_CELLS) and t.shape == s.shape and (s >= t - 1e-6).all()
            and (s - t).max() <= gen.HEIGHT_RANGE[1] + 10 and np.abs(hm["x"]).max() < gen.HALF
            and np.abs(hm["y"]).max() < gen.HALF
            and occ.shape == s.shape and occ.min() >= 0 and occ.max() <= 1 and (occ[built] > 0).all()):
        add("heightmap")
        return None
    return meta, hm, src


def check_xml(d, name):
    p = d / f"{name}.xml"
    root = ET.parse(p).getroot()
    bsdf_ids = {b.get("id") for b in root.iter("bsdf") if b.get("id")}
    files, shapes = set(), list(root.iter("shape"))
    if not shapes:
        return "no shapes"
    for sh in shapes:
        fn = next((s.get("value") for s in sh.findall("string") if s.get("name") == "filename"), None)
        if fn is None:
            return "shape without filename"
        if not (p.parent / fn).exists():
            return f"missing file {fn}"
        files.add(Path(fn).name)
        ref, inline = sh.find("ref"), sh.find("bsdf")
        if ref is not None and ref.get("id") not in bsdf_ids:
            return f"material {ref.get('id')} not defined"
        if ref is None and inline is None:
            return f"{fn} has no material"
    if not {"ground.ply", "buildings.ply"} <= files:
        return "ground.ply or buildings.ply not used by the scene"
    return None


def check_ground(d, hm, add):
    """Ground mesh checks. Returns its (triangles, cross products) when it could be read, else None."""
    half = gen.HALF
    try:
        vg, fg = read_ply(d / "meshes" / "ground.ply")
        tri, cr, area = triangle_geometry(vg, fg)
        ok = (np.isfinite(vg).all() and len(fg) > 0 and (area > MIN_TRI_AREA).all() and (unit_z(cr) > 0).all()
              and vg[:, 0].min() <= -half + 1e-3 and vg[:, 0].max() >= half - 1e-3
              and vg[:, 1].min() <= -half + 1e-3 and vg[:, 1].max() >= half - 1e-3
              and np.abs(vg[:, 2] - terrain_at(hm, vg[:, 0], vg[:, 1])).max() <= GROUND_TOL)
    except Exception:
        add("ground")
        return None
    if not ok:
        add("ground")
    return tri, cr


def check_buildings(d, meta, hm, ground, add):
    """Building meshes: readable, no degenerate faces, inside, watertight, outward, one mesh per building,
    standing on the ground, exactly the meta.json buildings, no coplanar faces with other meshes or the ground."""
    try:
        v, f = read_ply(d / "meshes" / "buildings.ply")
        assert np.isfinite(v).all() and len(f) > 0
    except Exception as ex:
        add("b_read", f"{type(ex).__name__}: {ex}")
        return
    if meta["num_buildings"] == 0:
        return
    tri, cr, area = triangle_geometry(v, f)
    comp = connected_components(f, len(v))
    n_comp = int(comp.max()) + 1
    if (area <= MIN_TRI_AREA).any():
        add("b_area", f"{int((area <= MIN_TRI_AREA).sum())} faces")
    if np.abs(v[:, :2]).max() > gen.HALF - 1 + 1e-3:
        add("b_inside", f"max |x|, |y| = {np.abs(v[:, :2]).max():.3f} m")
    if n_comp != meta["num_meshes"]:
        add("b_count", f"{n_comp} closed meshes in the file, {meta['num_meshes']} in meta")
    if not is_watertight(v, f):
        add("b_closed")
    vol = np.bincount(comp, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6,
                      minlength=n_comp)
    if (vol <= 0).any():
        add("b_outward", f"{int((vol <= 0).sum())} of {n_comp} meshes")
    lo, hi = component_bounds(tri, comp, n_comp)
    for c in range(n_comp):
        ix0, iy0 = cell_index(lo[c, 0], lo[c, 1])
        ix1, iy1 = cell_index(hi[c, 0], hi[c, 1])
        g = hm["terrain"][iy0:iy1 + 1, ix0:ix1 + 1].max()
        if lo[c, 2] > g + 0.1 or hi[c, 2] > g + gen.HEIGHT_RANGE[1] + 10:
            add("b_floating")
            break
    if n_comp == len(meta["buildings"]):
        problem = mesh_vs_geometry(tri, cr, area, comp, lo, hi, vol, meta)
        if problem:
            add("g_mesh", problem)
    if ground is not None:      # coplanar overlapping faces, between buildings and between buildings and ground
        tri_g, cr_g = ground
        n_cop = coplanar_overlaps(np.vstack([tri, tri_g]), np.vstack([cr, cr_g]),
                                  np.concatenate([comp, np.full(len(tri_g), n_comp)]))
        if n_cop:
            add("coplanar", f"{n_cop} face pairs")


def check_exact_geometry(hm, parts, cfg, add):
    """meta.json geometry: gives the saved height map, no overlap, gaps, thin parts."""
    XC, YC = np.meshgrid(hm["x"], hm["y"])
    surf = hm["terrain"].copy()
    for p in parts:
        m = gen.inside_rect(XC, YC, p)
        surf[m] = np.maximum(surf[m], p["z_top"])
    err = np.abs(surf - hm["surface"])
    if err.max() > 1e-6:
        add("g_match", f"{int((err > 1e-6).sum())} cells differ")
    min_gap, overlap = np.inf, False
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            pa, pb = parts[a], parts[b]
            if pa["building"] == pb["building"]:
                continue
            if np.hypot(pa["cx"] - pb["cx"], pa["cy"] - pb["cy"]) - pa["r"] - pb["r"] > 10.0:
                continue
            if gen.outlines_overlap(pa, pb):
                overlap = True
            else:
                min_gap = min(min_gap, polygon_distance(pa["corners"], pb["corners"]))
    if overlap:
        add("g_overlap")
    if min_gap < MIN_SLOT:
        add("g_slot", f"{min_gap:.2f} m")
    elif min_gap < cfg["MIN_GAP"] - 1e-3:
        add("g_gap", f"{min_gap:.2f} m")
    if parts:
        thin = min(min(p["w"], p["d"]) for p in parts)
        if thin < MIN_PART_SIDE:
            add("g_thin", f"{thin:.2f} m")


def check_roofs_and_pockets(hm, parts, cfg, alts, add):
    """Roofs clear of the drone altitudes; no tiny closed pockets of free space."""
    roofs = np.array([p["z_top"] for p in parts])
    if len(roofs):
        dist = np.abs(roofs[:, None] - np.asarray(alts, float)[None]).min()
        if dist < ROOF_HARD_CLEARANCE:
            add("roof_hard", f"roof {dist:.2f} m from an altitude")
        elif dist < cfg["ROOF_CLEARANCE"] - 1e-6:
            add("roof", f"roof {dist:.2f} m from an altitude")
    built = hm["surface"] - hm["terrain"] > 0.5
    lab, _ = ndimage.label(gen.enclosed_cells(~built))       # 4-connected regions
    size = np.bincount(lab.ravel())[1:]
    small = size[size * gen.CELL ** 2 < MIN_POCKET_AREA]
    if len(small):
        add("pocket", f"{len(small)} pockets, sizes {small[:3].tolist()} cells")


def check_stations(src, hm, parts, cfg, alts, add):
    p = src["positions"]
    if len(p) == 0:
        add("st_basic", "no stations")
        return
    s = hm["surface"]
    agl = p[:, 2] - src["ground_z"]
    ix, iy = cell_index(p[:, 0], p[:, 1])
    why = []
    if len(p) < 8:
        why.append(f"only {len(p)}")
    if len(np.unique(np.round(p, 3), axis=0)) != len(p):
        why.append("duplicates")
    if np.abs(p[:, :2]).max() > gen.HALF - cfg["EDGE_MARGIN"] + 1e-6:
        why.append("too close to the border")
    lo_agl, hi_agl = cfg["STATION_HEIGHT_AGL"]
    if agl.min() < lo_agl - 1e-4 or agl.max() > hi_agl + 1e-4:
        why.append(f"height above ground {agl.min():.2f} to {agl.max():.2f} m")
    if np.abs(src["ground_z"] - terrain_at(hm, p[:, 0], p[:, 1])).max() > GROUND_TOL:
        why.append("ground_z does not match the terrain")
    if not (p[:, 2] > s[iy, ix] + 0.5).all():
        why.append("below the surface")
    if why:
        add("st_basic", ", ".join(why))
    wall = (np.min([gen.rect_distance(p[:, 0], p[:, 1], q) for q in parts], axis=0) if parts
            else np.full(len(p), np.inf))
    if wall.min() < cfg["BUILDING_MARGIN"] - 1e-6:
        add("st_wall", f"{wall.min():.2f} m")
    clearance = min(alts) - p[:, 2].max()
    if clearance < cfg["PLANE_CLEARANCE"] - 1e-6:
        add("st_plane", f"station at {p[:, 2].max():.2f} m, {clearance:.2f} m below the plane")
    if len(p) > 1:
        dd = np.hypot(p[:, 0, None] - p[None, :, 0], p[:, 1, None] - p[None, :, 1])
        np.fill_diagonal(dd, np.inf)
        if dd.min() < cfg["STATION_MIN_SPACING"] - 1e-6:
            add("st_spacing", f"{dd.min():.2f} m")
    bad = []
    if np.abs(src["wall_distance"] - wall).max() > 1e-6:
        bad.append("wall_distance")
    cls = src["station_class"]
    if cls.min() < 0 or cls.max() >= len(cfg["STATION_CLASSES"]):
        bad.append("station_class")
    if bad:
        add("st_labels", ", ".join(bad))


def static_scene(job):
    """Every check of one scene that needs no Sionna. job = (scene folder, name, family, cfg, alts).
    Returns the problems found as a list of (check key, scene, detail); no shared state, so scenes can be
    checked in parallel processes."""
    d, name, family, cfg, alts = job
    found = []

    def add(key, detail=""):
        found.append((key, name, detail))

    loaded = load_scene_files(d, name, add)
    if loaded is None:
        return found
    meta, hm, src = loaded
    if meta["num_buildings"] < gen.MIN_BUILDINGS[family] or meta["built_fraction"] <= 0:
        add("empty", f"{meta['num_buildings']} buildings")
    try:
        problem = check_xml(d, name)
    except Exception as ex:
        problem = f"{type(ex).__name__}: {ex}"
    if problem:
        add("xml", problem)
    ground = check_ground(d, hm, add)
    check_buildings(d, meta, hm, ground, add)
    parts = scene_parts(meta)
    check_exact_geometry(hm, parts, cfg, add)
    check_roofs_and_pockets(hm, parts, cfg, alts, add)
    check_stations(src, hm, parts, cfg, alts, add)
    return found


def check_static(scenes_dir, selected, cfg, alts, workers):
    P = Problems(static_titles(cfg, alts))
    jobs = [(scenes_dir / e["name"], e["name"], e["family"], cfg, alts) for e in selected]
    for k, found in enumerate(parallel_map(static_scene, jobs, workers, min_jobs=100)):   # ~15 ms per scene
        print(f"      scene {k + 1}/{len(jobs)}        ", end="\r", flush=True)
        for key, name, detail in found:
            P.add(key, name, detail)
    print(" " * 60, end="\r")
    P.emit(len(selected))


# ---------------------------------------------------------------- global checks
def check_index(scenes_dir, index):
    names = [e["name"] for e in index]
    folders = sorted(p.name for p in scenes_dir.iterdir() if p.is_dir())
    R.single(len(set(names)) == len(names), "scene names are unique")
    extra, miss = sorted(set(folders) - set(names)), sorted(set(names) - set(folders))
    R.single(not extra and not miss, "index.json matches the scene folders"
             + (f" (folders not in index: {extra[:3]}, index without folder: {miss[:3]})" if extra or miss else ""))
    R.single(all((e["split"] == "ood") == (e["family"] in gen.OOD_FAMILIES) for e in index),
             "unseen layout families are exactly the 'ood' split")
    for fam in gen.ID_FAMILIES:
        sp = [e["split"] for e in index if e["family"] == fam]
        R.single(set(sp) == {"train", "val", "test"}, f"family {fam} is split into train, val and test "
                 f"({sp.count('train')}/{sp.count('val')}/{sp.count('test')})")


def check_duplicates(scenes_dir, names):
    seen = {}
    for n in names:
        seen.setdefault(scene_hash(scenes_dir / n), []).append(n)
    dup = [(v[0], f"same as {v[1:]}") for v in seen.values() if len(v) > 1]
    R.check("no two scenes have identical height maps", dup, len(names))


def same_file(a, b):
    """Same content: arrays for .npz (np.savez files are not byte stable), bytes for everything else."""
    if not (a.exists() and b.exists()):
        return False
    if a.suffix != ".npz":
        return file_hash(a) == file_hash(b)
    da, db_ = load_npz(a), load_npz(b)
    return da.keys() == db_.keys() and all(np.array_equal(da[k], db_[k]) for k in da)


def regenerate(job):
    """Build one scene again with its seed in a temporary folder and compare every file.
    Returns (scene, which files differ) or None."""
    scenes_dir, name, family, seed = job
    plan = gen.ID_FAMILIES + gen.OOD_FAMILIES
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        gen.build_scene(tmp, name, family, np.random.default_rng([seed, plan.index(family), int(name.split("_")[-1])]))
        diff = [rel for rel in (r.format(name=name) for r in SCENE_FILES)
                if not same_file(tmp / name / rel, scenes_dir / name / rel)]
    return (name, f"differs: {diff}") if diff else None


def check_reproducible(scenes_dir, index, n_check, seed, workers):
    pick = np.random.default_rng(1).choice(len(index), min(n_check, len(index)), replace=False)
    jobs = [(scenes_dir, index[k]["name"], index[k]["family"], seed) for k in pick]
    bad = [r for r in parallel_map(regenerate, jobs, workers, min_jobs=20) if r is not None]   # ~0.15 s per scene
    R.check("regenerating with the same seed gives identical scenes (data and files)", bad, len(pick))


def check_split_distributions(scenes_dir, index):
    """Train, val and test of every in distribution family should come from the same distribution."""
    metas = {e["name"]: json.loads((scenes_dir / e["name"] / "meta.json").read_text()) for e in index
             if e["family"] in gen.ID_FAMILIES}
    bad, worst = [], 1.0
    for fam in gen.ID_FAMILIES:
        for key in ("built_fraction", "max_height", "num_buildings"):
            vals = {sp: [metas[e["name"]][key] for e in index if e["family"] == fam and e["split"] == sp]
                    for sp in ("train", "val", "test")}
            for sp in ("val", "test"):
                d, pv = ks_test(vals["train"], vals[sp])
                worst = min(worst, pv)
                if pv < KS_MIN_P:
                    bad.append(("(global)", f"{fam} {key}: train vs {sp}, KS D = {d:.2f}, p = {pv:.3f}"))
    R.check("train, val and test have the same distributions (KS test per family)", bad,
            len(gen.ID_FAMILIES) * 6, "warn", f"lowest p = {worst:.3f}")


# ---------------------------------------------------------------- Sionna helpers
def _scalar(x):
    return float(np.array(x).ravel()[0])


def up_inside(mi_scene, X, Y, Z):
    """True where the point is inside a solid: a ray cast upwards hits the BACK of a face first
    (Mitsuba's si.n is the geometric normal given by the winding, i.e. it points out of the solid)."""
    import mitsuba as mi
    n = np.size(X)
    z = np.broadcast_to(np.asarray(Z, "f4"), (n,)).copy()
    ray = mi.Ray3f(mi.Point3f(np.ravel(X).astype("f4"), np.ravel(Y).astype("f4"), z), mi.Vector3f(0, 0, 1))
    si = mi_scene.ray_intersect(ray)
    return (np.array(si.is_valid()) & (np.array(si.n.z) > 0)).reshape(np.shape(X))


def los_rays(mi_scene, stations, X, Y, alt):
    """Exact line of sight from every station (k x 3) to every cell centre at height alt, all in one
    Mitsuba ray_test. Returns k x H x W."""
    import mitsuba as mi
    stations = np.atleast_2d(np.asarray(stations, float))
    tgt = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, alt)])
    org = np.repeat(stations, X.size, axis=0)
    dvec = np.tile(tgt, (len(stations), 1)) - org
    dist = np.linalg.norm(dvec, axis=1)
    dvec /= dist[:, None]
    ray = mi.Ray3f(mi.Point3f(*[org[:, i].astype("f4") for i in range(3)]),
                   mi.Vector3f(*[dvec[:, i].astype("f4") for i in range(3)]))
    ray.maxt = mi.Float((dist * (1 - 1e-4)).astype("f4"))
    return ~np.array(mi_scene.ray_test(ray)).reshape(len(stations), *X.shape)


def wall_rays(mi_scene, hm):
    """Horizontal rays along every row and every column of pixel centres, in the 4 directions (+x, -x, +y, -y),
    at WALL_LEVELS above the lowest ground. Each ray must stop at the first wall the height map predicts on its
    line: not later than the first solid pixel centre, and not earlier than the free pixel centre before it
    (an earlier hit is only accepted next to ground or roofs at that height, e.g. a hill between two pixel
    centres). A ray must hit nothing where the height map has nothing. Rays that would enter the area below the
    terrain are left out (they run under the open ground surface, which no real ray can do).
    Returns one bool per ray: agrees with the height map."""
    import mitsuba as mi
    s, t = hm["surface"], hm["terrain"]
    nc, start = gen.N_CELLS, gen.HALF + 5.0
    zl = float(t.min()) + np.asarray(WALL_LEVELS, float)
    pad = np.pad(s, 2, mode="edge")
    dil = np.max([pad[2 + dy:2 + dy + nc, 2 + dx:2 + dx + nc] for dy in range(-2, 3) for dx in range(-2, 3)], axis=0)
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if dy == 0:     # one ray per row, cells in travel order
            prof_s, prof_t, pos, lines = (s, t, hm["x"], hm["y"]) if dx > 0 else (s[:, ::-1], t[:, ::-1], hm["x"][::-1], hm["y"])
        else:           # one ray per column
            prof_s, prof_t, pos, lines = (s.T, t.T, hm["y"], hm["x"]) if dy > 0 else (s.T[:, ::-1], t.T[:, ::-1], hm["y"][::-1], hm["x"])
        sc = start + (dx + dy) * pos                       # distance from the ray origin to each cell centre
        li = np.repeat(np.arange(len(lines)), len(zl))
        L, Z = lines[li], np.tile(zl, len(lines))
        ox = np.full(L.size, -dx * start) if dy == 0 else L
        oy = L if dy == 0 else np.full(L.size, -dy * start)
        ray = mi.Ray3f(mi.Point3f(ox.astype("f4"), oy.astype("f4"), Z.astype("f4")), mi.Vector3f(dx, dy, 0))
        tt = np.array(mi_scene.ray_intersect(ray).t)
        hit = np.isfinite(tt)
        tz = np.where(hit, tt, 0.0)
        ix, iy = cell_index(ox + dx * tz, oy + dy * tz)
        ok_graze = hit & (dil[iy, ix] >= Z - 0.5)
        above = prof_s[li] > Z[:, None]                     # rays x cells
        exp_hit = above.any(axis=1)
        s_first = sc[above.argmax(axis=1)]                  # first solid cell centre on the line
        ok_solid = hit & (tt <= s_first + 0.05) & ((tt >= s_first - gen.CELL - 0.05) | ok_graze)
        agree = np.where(exp_hit, ok_solid, ~hit | ok_graze)
        keep = Z > prof_t[li, :2].max(axis=1) + 0.3
        out.append(agree[keep])
    out = np.concatenate(out)
    return out if out.size else np.ones(1, bool)


# ---------------------------------------------------------------- Sionna checks
def sionna_titles(alts, frequency, scattering):
    return {
        "load": ("every scene loads in Sionna", "fail"),
        "materials": (f"every object has a valid radio material at {frequency / 1e9:.2f} GHz with scattering "
                      f"coefficient {scattering} (as in config.json)", "fail"),
        "vertical": ("Sionna mesh matches the height map (vertical rays)", "fail"),
        "walls": (f"Sionna walls match the height map (horizontal rays along every row and column, 4 directions, "
                  f"{len(WALL_LEVELS)} heights; at most {WALL_MAX_WRONG * 100:.1f}% disagree)", "fail"),
        "walls_warn": ("Sionna walls match the height map for every horizontal ray", "warn"),
        "st_inside": ("stations are outside every solid (rays cast upwards)", "fail"),
        "mask": (f"validity mask from the mesh = height map mask at {list(alts)} m "
                 f"(at most {MASK_MAX_MISMATCH * 100:.1f}% of the cells differ)", "fail"),
        "mask_warn": ("validity mask from the mesh = height map mask in every cell", "warn"),
    }


def check_sionna_scenes(scenes_dir, names, alts, frequency, scattering):
    """scattering: the scattering coefficient every material must have once loaded (Sionna silently drops it for
    some material ids, see generate_city_scenes.write_xml)."""
    import mitsuba as mi
    P = Problems(sionna_titles(alts, frequency, scattering))
    worst, worst_wall, worst_mask, materials = 0.0, 1.0, 0.0, set()
    for k, n in enumerate(names):
        print(f"      sionna {k + 1}/{len(names)} {n}        ", end="\r", flush=True)
        try:
            scene = load_radio_scene(scenes_dir / n / f"{n}.xml", frequency)
        except Exception as ex:
            P.add("load", n, f"{type(ex).__name__}: {ex}")
            continue
        try:
            objs = list(scene.objects.values())
            why = [] if len(objs) >= 2 else [f"only {len(objs)} objects"]
            for o in objs:
                m = o.radio_material
                if m is None:
                    why.append(f"{o.name} has no material")
                    continue
                materials.add(m.name)
                eps, sig = _scalar(m.relative_permittivity), _scalar(m.conductivity)
                if not (np.isfinite(eps) and eps >= 1 and np.isfinite(sig) and sig >= 0):
                    why.append(f"{m.name}: eps {eps}, sigma {sig}")
                s_m = _scalar(m.scattering_coefficient)
                if abs(s_m - scattering) > 1e-6:
                    why.append(f"{m.name}: scattering coefficient {s_m:g} instead of {scattering:g}")
        except Exception as ex:    # e.g. ITU material not defined at this frequency
            why = [f"{type(ex).__name__}: {ex}"]
        if why:
            P.add("materials", n, "; ".join(why))

        hm = load_npz(scenes_dir / n / "height_map.npz")
        XC, YC = np.meshgrid(hm["x"], hm["y"])
        # vertical rays: roofs and terrain
        top = 500.0
        ray = mi.Ray3f(mi.Point3f(XC.ravel().astype("f4"), YC.ravel().astype("f4"),
                                  np.full(XC.size, top, "f4")), mi.Vector3f(0, 0, -1))
        z = top - np.array(scene.mi_scene.ray_intersect(ray).t)
        err = np.abs(z.reshape(XC.shape) - hm["surface"]).max()
        worst = max(worst, err if np.isfinite(err) else np.inf)
        if not err <= 0.05:
            P.add("vertical", n, f"{err:.3f} m")

        # horizontal rays along every row and column, in the 4 directions, at several heights: walls
        agree = wall_rays(scene.mi_scene, hm)
        worst_wall = min(worst_wall, agree.mean())
        if (~agree).any():
            P.add("walls" if (~agree).mean() > WALL_MAX_WRONG else "walls_warn", n,
                  f"{int((~agree).sum())} of {agree.size} rays disagree")

        # stations outside every solid
        pos = load_npz(scenes_dir / n / "sources.npz")["positions"]
        inside = up_inside(scene.mi_scene, pos[:, 0], pos[:, 1], pos[:, 2])
        if inside.any():
            P.add("st_inside", n, f"stations {np.flatnonzero(inside)[:5].tolist()}")

        # validity mask of every drone altitude: mesh against height map
        for alt in alts:
            mesh_invalid = up_inside(scene.mi_scene, XC, YC, alt)
            hm_invalid = ~(hm["surface"] < alt - 0.5)
            frac = float((mesh_invalid != hm_invalid).mean())
            worst_mask = max(worst_mask, frac)
            if frac > MASK_MAX_MISMATCH:
                P.add("mask", n, f"{alt:.0f} m: {frac * 100:.2f}% of the cells")
            elif frac > 0:
                P.add("mask_warn", n, f"{alt:.0f} m: {int(round(frac * XC.size))} cells")
    print(" " * 60, end="\r")
    if materials:
        print(f"      radio materials used: {sorted(materials)}")
    P.info.update(vertical=f"worst error {worst:.3f} m", walls=f"lowest agreement {worst_wall * 100:.1f}%",
                  mask=f"worst {worst_mask * 100:.3f}% of the cells")
    P.emit(len(names))


# ---------------------------------------------------------------- physics on ray traced maps
def physics_titles(phys):
    return {
        "align": ("radio map cells line up with the height map pixels (rm.cell_centers)", "fail"),
        "nan": ("ray traced gain is finite and non-negative", "fail"),
        "free": ("gain right above the station follows free space", "fail"),
        "energy": (f"gain never exceeds free space by more than {ENERGY_MARGIN_DB:.0f} dB", "fail"),
        "shadow": ("shadowed cells get clearly less gain than line of sight cells (exact LoS)", "fail"),
        "inside": (f"no real signal inside buildings (at most {LEAK_MAX_POWER:.0e} of the map's power in the inner "
                   f"building cells)", "fail"),
        "inside_warn": ("no rays leak into buildings at all (numerical leaks through edges and corners)", "warn"),
        "holes": (f"line of sight cells without any ray at most {HOLE_MAX_FRACTION * 100:.1f}% "
                  f"({phys['samples']:.0e} rays per map)", "warn"),
        "noise": (f"Monte Carlo noise, {phys['samples']:.0e} vs twice as many rays (median LoS <= "
                  f"{NOISE_MAX_LOS_DB} dB, shadow <= {NOISE_MAX_NLOS_DB} dB)", "warn"),
        "depth": (f"converged in max_depth ({phys['max_depth']} vs {phys['max_depth'] + DEPTH_EXTRA}: median "
                  f"<= {DEPTH_MAX_DB} dB, <= {DEPTH_MAX_FLIP * 100:.0f}% of the cells switch on / off)", "warn"),
        "trivial": ("maps are informative (not almost all LoS, not almost all dead, enough valid cells)", "warn"),
        "many_bad": (f"most stations of a scene give good maps (more than half failing points at the scene)",
                     "fail"),
    }


def db(g):
    return 10 * np.log10(np.maximum(g, 1e-30))


def select_stations(n_pos, mode):
    """Stations to trace: 'ends' (first and last), 'all', or a number spread evenly over the list."""
    if mode == "all":
        return list(range(n_pos))
    if mode == "ends":
        return sorted({0, n_pos - 1})
    return sorted(set(np.linspace(0, n_pos - 1, min(int(mode), n_pos)).round().astype(int).tolist()))


def check_physics(scenes_dir, names, phys, alts, stations="ends", test_alts=None, n_depth=3, tx_batch=4,
                  dataset_choice=(DATASET_DEFAULTS["stations"], DATASET_DEFAULTS["seed"])):
    """Ray traced radio maps of the chosen stations of every scene, at every drone altitude.
    stations: 'ends', 'all', a number (see select_stations) or 'dataset': the stations make_dataset.py picks
    (dataset_choice = stations per scene, seed), plus the replacement of every one that fails.
    Up to tx_batch stations are traced together in one solver call (one map per station).
    Every map gets the per map checks. The noise (N vs 2N rays) and max_depth tests run at the altitudes
    test_alts (default: the middle one); the max_depth test only on the first n_depth scenes.
    A FAIL on one map marks that station as bad (make_dataset.py will not use it); if more than half of
    the stations of a scene fail, the scene itself fails. Returns {scene: number of stations checked}."""
    import drjit as dr
    from sionna.rt import Transmitter, RadioMapSolver
    solver = RadioMapSolver()
    lam = 299792458.0 / phys["frequency"]
    P = Problems(physics_titles(phys))
    stats = dict(fs=[], frac=[], drop=[], noise_los=[], noise_nlos=[], noise_p95=[], depth=[], flip=[], holes=[],
                 skipped=0, maps=0)
    test_alts = test_alts or [alts[len(alts) // 2]]
    checked, bad = {}, {}

    def flag(key, n, detail, si=None):
        P.add(key, n, detail, si)
        if si is not None and P.titles[key][1] == "fail":
            bad.setdefault(n, set()).add(int(si))

    def trace(scene, alt, depth=None, samples=None):
        """Maps of every transmitter in the scene, with the dataset's ray tracing settings (common.radio_map)."""
        rm = radio_map(solver, scene, alt, gen.AREA_SIDE, gen.CELL, samples or phys["samples"],
                       depth or phys["max_depth"], phys["diffraction"], phys["diffuse"])
        g = np.array(rm.path_gain)
        dr.sync_thread()
        return rm, g

    def check_batch(scene, n, k_sc, pos, chunk, masks, XC, YC):
        """Trace the stations chunk of scene n together and run every per map check on their maps."""
        S = pos[chunk]                                                  # stations of this batch, k x 3
        for k, si in enumerate(chunk):
            scene.add(Transmitter(name=f"chk{k}", position=[float(c) for c in pos[si]]))
        for alt in alts:
            rm, g = trace(scene, alt)
            stats["maps"] += len(chunk)
            if alt not in masks:
                cc = np.array(rm.cell_centers)
                if cc.shape != (gen.N_CELLS, gen.N_CELLS, 3) or np.abs(cc[..., 0] - XC).max() > 1e-3 \
                        or np.abs(cc[..., 1] - YC).max() > 1e-3 or np.abs(cc[..., 2] - alt).max() > 1e-3:
                    flag("align", n, f"{alt:.0f} m: cell centres {cc.shape} do not match the height map grid")
                inside = up_inside(scene.mi_scene, XC, YC, alt)
                masks[alt] = (~inside, erode(inside))
                if (~inside).mean() < MIN_VALID_FRACTION:
                    flag("trivial", n, f"{alt:.0f} m: only {(~inside).mean() * 100:.1f}% valid cells")
            valid, core = masks[alt]
            # whole batch at once: gain relative to free space and exact line of sight, k x H x W
            d = np.sqrt((XC - S[:, 0, None, None]) ** 2 + (YC - S[:, 1, None, None]) ** 2
                        + (alt - S[:, 2, None, None]) ** 2)
            diff_all = db(g) + 20 * np.log10(4 * np.pi * d / lam)
            los_all = los_rays(scene.mi_scene, S, XC, YC, alt)
            saved = {}
            for k, si in enumerate(chunk):
                gk, s, diff, los = g[k], pos[si], diff_all[k], los_all[k]
                tag = f"tx {si}, {alt:.0f} m"
                if not (np.isfinite(gk).all() and (gk >= 0).all()):
                    flag("nan", n, tag, si)
                    continue
                ix, iy = cell_index(s[0], s[1])
                stats["fs"].append(diff[iy, ix])
                if not FS_WINDOW_DB[0] <= diff[iy, ix] <= FS_WINDOW_DB[1]:
                    flag("free", n, f"{tag}: {diff[iy, ix]:+.1f} dB", si)
                frac = float((diff[valid] > ENERGY_MARGIN_DB).mean())
                stats["frac"].append(frac)
                if frac > ENERGY_MAX_FRACTION:
                    flag("energy", n, f"{tag}: {frac * 100:.2f}% of the cells", si)
                leak = core & (gk > 0)
                if leak.any():
                    share = gk[leak].sum() / max(gk[valid].sum(), 1e-300)
                    detail = (f"{tag}: {int(leak.sum())} of {int(core.sum())} inner cells, strongest "
                              f"{db(gk[leak]).max():.0f} dB, {share:.0e} of the power of the valid cells")
                    flag("inside" if share > LEAK_MAX_POWER else "inside_warn", n, detail, si)
                lv, bv = valid & los, valid & ~los
                if lv.sum() >= SHADOW_MIN_CELLS and bv.sum() >= SHADOW_MIN_CELLS:
                    drop = np.median(diff[lv]) - np.median(diff[bv])
                    stats["drop"].append(drop)
                    if drop < SHADOW_MIN_DROP_DB or not FS_WINDOW_DB[0] <= np.median(diff[lv]) <= FS_WINDOW_DB[1]:
                        flag("shadow", n, f"{tag}: drop {drop:.1f} dB", si)
                else:
                    stats["skipped"] += 1
                if lv.any():
                    holes = float((gk[lv] == 0).mean())
                    stats["holes"].append(holes)
                    if holes > HOLE_MAX_FRACTION:
                        flag("holes", n, f"{tag}: {holes * 100:.2f}%", si)
                if valid.any():
                    if lv.sum() / valid.sum() > TRIVIAL_FRACTION:
                        flag("trivial", n, f"{tag}: {lv.sum() / valid.sum() * 100:.0f}% of the valid cells in LoS",
                             si)
                    if (gk[valid] == 0).mean() > TRIVIAL_FRACTION:
                        flag("trivial", n, f"{tag}: {(gk[valid] == 0).mean() * 100:.0f}% of the valid cells "
                                           f"dead", si)
                if alt in test_alts:
                    saved[k] = (gk, los)
            if not saved:
                continue
            # Monte Carlo noise: same maps with twice as many rays (the seed does not change the ray directions)
            _, g2 = trace(scene, alt, samples=2 * int(phys["samples"]))
            for k, (gk, los) in saved.items():
                both = (gk > 0) & (g2[k] > 0) & valid
                dl, dn = np.abs(db(gk) - db(g2[k]))[both & los], np.abs(db(gk) - db(g2[k]))[both & ~los]
                ml, mn = (float(np.median(x)) if x.size else 0.0 for x in (dl, dn))
                stats["noise_los"].append(ml)
                stats["noise_nlos"].append(mn)
                stats["noise_p95"].append(float(np.percentile(np.concatenate([dl, dn]), 95)) if both.any() else 0.0)
                if ml > NOISE_MAX_LOS_DB or mn > NOISE_MAX_NLOS_DB:
                    flag("noise", n, f"tx {chunk[k]}, {alt:.0f} m: median LoS {ml:.2f} dB, shadow {mn:.2f} dB",
                         chunk[k])
            # convergence in max_depth
            if k_sc < n_depth:
                _, gd = trace(scene, alt, depth=phys["max_depth"] + DEPTH_EXTRA)
                for k, (gk, _) in saved.items():
                    both = (gk > 0) & (gd[k] > 0) & valid
                    med = float(np.median(np.abs(db(gk) - db(gd[k]))[both])) if both.any() else 0.0
                    flip = float(((gk > 0) != (gd[k] > 0))[valid].mean()) if valid.any() else 0.0
                    stats["depth"].append(med)
                    stats["flip"].append(flip)
                    if med > DEPTH_MAX_DB or flip > DEPTH_MAX_FLIP:
                        flag("depth", n, f"tx {chunk[k]}, {alt:.0f} m: median {med:.2f} dB, "
                                         f"{flip * 100:.1f}% switch", chunk[k])
        for k in range(len(chunk)):
            scene.remove(f"chk{k}")

    t_start = time.perf_counter()
    for k_sc, n in enumerate(names):
        el = time.perf_counter() - t_start
        eta = el / k_sc * (len(names) - k_sc) if k_sc else 0.0
        print(f"      physics {k_sc + 1}/{len(names)} {n}, {stats['maps']} maps, elapsed {el / 60:.1f} min, "
              f"eta {eta / 60:.1f} min      ", end="\r", flush=True)
        scene = load_radio_scene(scenes_dir / n / f"{n}.xml", phys["frequency"])
        hm = load_npz(scenes_dir / n / "height_map.npz")
        pos = load_npz(scenes_dir / n / "sources.npz")["positions"]
        XC, YC = np.meshgrid(hm["x"], hm["y"])
        masks = {}      # altitude -> (valid, inner cells), the same for every station
        if stations == "dataset":
            # the stations make_dataset.py will use; when one of them fails, the one replacing it is checked too
            done = set()
            while True:
                todo = [int(s) for s in pick_stations(n, len(pos), *dataset_choice, bad.get(n, ()))
                        if int(s) not in done]
                if not todo:
                    break
                for b0 in range(0, len(todo), tx_batch):
                    check_batch(scene, n, k_sc, pos, todo[b0:b0 + tx_batch], masks, XC, YC)
                done |= set(todo)
            checked[n] = len(done)
        else:
            picks = select_stations(len(pos), stations)
            checked[n] = len(picks)
            for b0 in range(0, len(picks), tx_batch):
                check_batch(scene, n, k_sc, pos, picks[b0:b0 + tx_batch], masks, XC, YC)
    print(" " * 100, end="\r")

    for n, b in bad.items():
        if len(b) > checked[n] / 2:
            flag("many_bad", n, f"{len(b)} of {checked[n]} stations fail")
    mx = lambda key, f=max, d=0.0: f(stats[key], default=d)
    n_bad = sum(len(b) for b in bad.values())
    P.info.update(
        free=f"{mx('fs', min):+.1f} to {mx('fs'):+.1f} dB",
        energy=f"worst {mx('frac') * 100:.2f}% of the cells",
        shadow=f"smallest drop {mx('drop', min, float('nan')):.1f} dB, {stats['skipped']} maps without enough shadow",
        holes=f"worst {mx('holes') * 100:.2f}%",
        noise=(f"at {test_alts} m, worst median LoS {mx('noise_los'):.2f} dB, shadow {mx('noise_nlos'):.2f} dB, "
               f"95th percentile up to {mx('noise_p95'):.1f} dB"),
        depth=(f"at {test_alts} m on {min(n_depth, len(names))} scenes, worst median {mx('depth'):.2f} dB, "
               f"{mx('flip') * 100:.2f}% switch" if stats["depth"] else "not run"),
        many_bad=f"{n_bad} bad stations in {len(bad)} scenes, {stats['maps']} maps checked")
    print(f"      {stats['maps']} maps traced and checked ({sum(checked.values())} stations x {len(alts)} altitudes) "
          f"in {(time.perf_counter() - t_start) / 60:.1f} min")
    P.emit(len(names))
    return checked


# ---------------------------------------------------------------- dataset checks
DS_KEYS = ("heights", "scene_names", "scene_hash", "scene_idx", "station_pos", "altitude", "target_db", "rt_time")


def check_dataset(scenes_dir, ds_dir, index, alts, cfg):
    split_of = {e["name"]: e["split"] for e in index}
    settings = json.loads((ds_dir / "settings.json").read_text()) if (ds_dir / "settings.json").exists() else {}
    lam = 299792458.0 / settings.get("frequency", DEFAULT_PHYSICS["frequency"])
    floor = settings.get("floor_db", FLOOR_DB)
    clearance = cfg["PLANE_CLEARANCE"]
    seen, stats = {}, {}
    own = "(dataset)"
    if settings.get("altitudes"):
        extra = sorted(set(settings["altitudes"]) - set(alts))
        R.single(not extra, f"dataset altitudes are among the altitudes the scenes were made for {list(alts)}"
                 + (f" (not: {extra})" if extra else ""), "warn", own)
    for split in ["train", "val", "test", "ood"]:
        f = ds_dir / f"{split}.npz"
        if not f.exists():
            continue
        d = load_npz(f)
        missing = [k for k in DS_KEYS if k not in d]
        R.single(not missing, f"[{split}] has every array" + (f" (missing {missing}, rerun make_dataset.py)"
                                                             if missing else ""), owner=own)
        if missing:
            continue
        n, h = len(d["altitude"]), d["heights"]
        t = d["target_db"]
        ok_shape = (t.shape == (n, gen.N_CELLS, gen.N_CELLS) and h.shape[1:] == t.shape[1:]
                    and d["station_pos"].shape == (n, 3) and d["scene_idx"].max() < len(h))
        R.single(ok_shape, f"[{split}] shapes consistent ({n} maps, {len(h)} scenes)", owner=own)
        R.single(np.isfinite(t).all() and t.min() >= floor - 0.01 and t.max() < 0,
                 f"[{split}] no NaN, gains between {t.min():.1f} and {t.max():.1f} dB", owner=own)
        R.single(all(split_of.get(s) == split for s in d["scene_names"]),
                 f"[{split}] every scene belongs to this split in index.json", owner=own)
        stale = [s for s, hh in zip(d["scene_names"], d["scene_hash"])
                 if not (scenes_dir / s).exists() or scene_hash(scenes_dir / s) != str(hh)]
        R.single(not stale, f"[{split}] dataset heights equal the current scene height maps"
                 + (f" (scenes changed since the dataset was made: {stale[:3]}, rerun make_dataset.py)"
                    if stale else ""), owner=own)
        for s in d["scene_names"]:
            seen.setdefault(s, []).append(split)
        R.single((d["rt_time"] > 0).all(), f"[{split}] ray tracing times saved "
                 f"(mean {d['rt_time'].mean():.2f} s per map)", owner=own)

        xs, alt = d["station_pos"], d["altitude"]
        gap = alt - xs[:, 2]
        R.single(gap.min() >= clearance - 1e-4, f"[{split}] every station at least {clearance:.0f} m below its map "
                 f"plane (smallest gap {gap.min():.2f} m)", owner=own)
        ix, iy = cell_index(xs[:, 0], xs[:, 1])
        cx, cy = (ix + 0.5) * gen.CELL - gen.HALF, (iy + 0.5) * gen.CELL - gen.HALF
        dist = np.sqrt((cx - xs[:, 0]) ** 2 + (cy - xs[:, 1]) ** 2 + (alt - xs[:, 2]) ** 2)
        diff = t[np.arange(n), iy, ix] - (-20 * np.log10(4 * np.pi * dist / lam))
        frac = np.mean((diff > FS_WINDOW_DB[0]) & (diff < FS_WINDOW_DB[1]))
        R.single(frac >= 0.99, f"[{split}] gain above the station follows free space in {frac * 100:.1f}% of maps",
                 owner=own)

        hs = h[d["scene_idx"]]
        valid = hs < alt[:, None, None] - 0.5
        core = erode(hs > alt[:, None, None] + 0.5)          # inner building cells of every map at once
        has_sig = t > floor + 0.01
        leak = has_sig & core
        power = np.where(has_sig, 10.0 ** (t / 10), 0.0)
        share = (power * leak).sum(axis=(1, 2)) / np.maximum((power * valid).sum(axis=(1, 2)), 1e-300)
        big = int((share > LEAK_MAX_POWER).sum())
        n_leak = int(leak.any(axis=(1, 2)).sum())
        R.single(big == 0, f"[{split}] no real signal inside buildings ({big} maps with large leaks)", owner=own)
        R.single(n_leak == 0, f"[{split}] no rays leak into buildings at all ({n_leak} maps with small numerical "
                 f"leaks; those cells are masked out anyway)", "warn", own)
        nosig = (t <= floor + 0.01) & valid
        nv = valid.reshape(n, -1).sum(1)
        nosig_map = nosig.reshape(n, -1).sum(1) / np.maximum(nv, 1)
        dead = int((nosig_map > TRIVIAL_FRACTION).sum())
        few = int((nv / valid[0].size < MIN_VALID_FRACTION).sum())
        R.single(dead == 0 and few == 0, f"[{split}] maps are informative ({dead} almost dead, {few} with fewer "
                 f"than {MIN_VALID_FRACTION * 100:.0f}% valid cells)", "warn", own)
        stats[split] = dict(valid=valid.mean(), nosig=nosig.sum() / max(valid.sum(), 1), nosig_map=nosig_map)
        print(f"      [{split}] valid cells {stats[split]['valid'] * 100:.1f}%, "
              f"no signal in {stats[split]['nosig'] * 100:.1f}% of the valid cells")
    shared = [s for s, v in seen.items() if len(v) > 1]
    R.single(not shared, "no scene appears in two splits", owner=own)
    return stats


# ---------------------------------------------------------------- statistics and figure
def statistics(scenes_dir, index, ds_stats, report_path):
    fams = gen.ID_FAMILIES + gen.OOD_FAMILIES
    data = {f: dict(built=[], hmax=[], nb=[]) for f in fams}
    for e in index:
        m = json.loads((scenes_dir / e["name"] / "meta.json").read_text())
        data[e["family"]]["built"].append(m["built_fraction"])
        data[e["family"]]["hmax"].append(m["max_height"])
        data[e["family"]]["nb"].append(m["num_buildings"])
    print("\nfamily      scenes  built fraction   max height (m)   buildings")
    for f in fams:
        v = data[f]
        if not v["built"]:
            continue
        print(f"{f:10s} {len(v['built']):6d}  {np.mean(v['built']):5.2f} "
              f"[{np.min(v['built']):.2f}-{np.max(v['built']):.2f}]   "
              f"{np.mean(v['hmax']):5.1f} [{np.min(v['hmax']):.0f}-{np.max(v['hmax']):.0f}]"
              f"   {np.mean(v['nb']):5.1f} [{np.min(v['nb'])}-{np.max(v['nb'])}]")
    n_pan = 4 if ds_stats else 3
    fig, axes = plt.subplots(1, n_pan, figsize=(4.6 * n_pan, 3.6), constrained_layout=True)
    for ax, key, title in zip(axes[:3], ["built", "hmax", "nb"],
                              ["built fraction of the area", "tallest building (m)", "number of buildings"]):
        for f in fams:
            ax.hist(data[f][key], bins=15, histtype="step", lw=1.8, label=f,
                    ls="-" if f in gen.ID_FAMILIES else "--")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    if ds_stats:
        for split, st in ds_stats.items():
            axes[3].hist(st["nosig_map"], bins=20, range=(0, 1), histtype="step", lw=1.8, label=split)
        axes[3].set_title("share of valid cells without signal, per map")
        axes[3].legend(fontsize=8)
    fig.savefig(report_path, dpi=120)
    plt.close(fig)
    print(f"\nsaved {report_path}")


def write_results(scenes_dir, checked, sionna_set, physics_stations, only):
    """check_results.json: one entry per scene, read by make_dataset.py.
    status "fail" = the scene itself is broken (skipped); bad_stations = stations whose maps failed (not used)."""
    path = scenes_dir / "check_results.json"
    res = {"scenes": {}}
    if only and path.exists():
        res = json.loads(path.read_text())
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for n in checked:
        e = R.scenes.get(n, {"fail": [], "warn": []})
        st_fail = e.get("station_fail", [])
        res["scenes"][n] = dict(status="fail" if e["fail"] else "warn" if e["warn"] or st_fail else "pass",
                                hash=scene_hash(scenes_dir / n) if (scenes_dir / n / "height_map.npz").exists()
                                else None,
                                sionna_checked=n in sionna_set, physics_checked=n in physics_stations,
                                stations_checked=physics_stations.get(n, 0),
                                bad_stations=sorted(R.bad_stations.get(n, [])),
                                checked_at=stamp, fail=e["fail"], station_fail=st_fail, warn=e["warn"])
    for key in ("(global)", "(dataset)"):
        if key in R.scenes or not only:
            res[key] = R.scenes.get(key, {"fail": [], "warn": []})
    st = [e["status"] for e in res["scenes"].values()]
    res["summary"] = {s: st.count(s) for s in ("pass", "warn", "fail")}
    res["summary"]["bad_stations"] = sum(len(e.get("bad_stations", [])) for e in res["scenes"].values())
    path.write_text(json.dumps(res, indent=1))
    print(f"saved {path}  (scenes: {res['summary']})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=str(ROOT / "scenes_city"))
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--only", nargs="+", help="check only these scenes (results are merged into check_results.json)")
    ap.add_argument("--full", action="store_true", help="physics check (ray traced maps) on every scene")
    ap.add_argument("--exhaustive", action="store_true",
                    help="every check on every possible map: all scenes, all stations, all altitudes, noise and "
                         "max_depth tests at every altitude of every scene, reproducibility of every scene (slow)")
    ap.add_argument("--no-sionna", action="store_true")
    ap.add_argument("--n-sionna", type=int, default=0,
                    help="limit the Sionna load/mesh/mask checks to this many random scenes (0 = all)")
    ap.add_argument("--n-physics", type=int, default=12, help="scenes for the ray traced physics checks")
    ap.add_argument("--stations", default="ends",
                    help="stations per scene for the physics checks: ends (first and last), all, a number, or "
                         "dataset (the stations make_dataset.py will use, and the replacement of any that fails)")
    ap.add_argument("--n-depth", type=int, default=3, help="of the physics scenes, how many get the max_depth test")
    ap.add_argument("--n-repro", type=int, default=10)
    # ray tracing settings: given here, else as in dataset/settings.json, else common.RT_DEFAULTS
    ap.add_argument("--samples", type=int, default=0, help="rays per map for the physics checks")
    ap.add_argument("--max-depth", type=int, default=0, help="max_depth for the physics checks")
    ap.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--diffuse", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--tx-batch", type=int, default=4, help="stations traced together in one solver call "
                                                            "(4 fits 8 GB of GPU memory at 1e8 rays)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes for the checks that need no Sionna (1 = no process pool)")
    ap.add_argument("--report", default=str(ROOT / "check_report.png"))
    args = ap.parse_args()
    if args.stations not in ("ends", "all", "dataset") and not args.stations.isdigit():
        sys.exit("--stations must be ends, all, dataset or a number")
    if args.exhaustive:
        args.full, args.stations, args.n_sionna = True, "all", 0

    scenes_dir, ds_dir = Path(args.scenes), Path(args.dataset)
    if not (scenes_dir / "index.json").exists():
        sys.exit(f"{scenes_dir} has no index.json, run generate_city_scenes.py first")
    index = json.loads((scenes_dir / "index.json").read_text())
    cfg = json.loads((scenes_dir / "config.json").read_text())
    missing = [k for k in CFG_KEYS if k not in cfg]
    if missing:
        sys.exit(f"{scenes_dir / 'config.json'} lacks {missing}, regenerate the scenes with generate_city_scenes.py")
    alts = [float(a) for a in cfg["DRONE_ALTITUDES"]]
    phys = dict(DEFAULT_PHYSICS)
    dataset_choice = (DATASET_DEFAULTS["stations"], DATASET_DEFAULTS["seed"])
    if (ds_dir / "settings.json").exists():
        st = json.loads((ds_dir / "settings.json").read_text())
        phys.update({k: st[k] for k in phys if k in st})
        dataset_choice = (st.get("stations", dataset_choice[0]), st.get("seed", dataset_choice[1]))
    given = dict(samples=args.samples or None, max_depth=args.max_depth or None,      # None: not on the command line
                 diffraction=args.diffraction, diffuse=args.diffuse)
    phys.update({k: v for k, v in given.items() if v is not None})
    if phys["diffuse"] and not cfg.get("SCATTERING_COEFFICIENT"):
        print("WARNING: the scenes have no scattering_coefficient, so diffuse reflections have no effect; "
              "regenerate them with generate_city_scenes.py\n")
    if args.exhaustive:
        args.n_repro = len(index)

    by_name = {e["name"]: e for e in index}
    if args.only:
        unknown = [n for n in args.only if n not in by_name]
        if unknown:
            sys.exit(f"not in index.json: {unknown}")
        selected = [by_name[n] for n in args.only]
    else:
        selected = index
    names = [e["name"] for e in selected]
    print(f"{len(names)} of {len(index)} scenes in {scenes_dir}; drone altitudes {alts} m; "
          f"physics at {phys['frequency'] / 1e9:.2f} GHz, {phys['samples']:.0e} rays, depth {phys['max_depth']}"
          f"{', diffraction' if phys['diffraction'] else ''}{', diffuse reflections' if phys['diffuse'] else ''}\n")

    t0 = time.perf_counter()
    check_static(scenes_dir, selected, cfg, alts, args.workers)
    print(f"      ({time.perf_counter() - t0:.0f} s)")

    if not args.only:
        check_index(scenes_dir, index)
        check_duplicates(scenes_dir, names)
        check_split_distributions(scenes_dir, index)
        check_reproducible(scenes_dir, index, args.n_repro, cfg["SEED"], args.workers)

    sionna_set, physics_stations = set(), {}
    if not args.no_sionna:
        rng = np.random.default_rng(0)
        k = len(names) if args.n_sionna <= 0 else min(args.n_sionna, len(names))
        sionna_names = [names[i] for i in sorted(rng.choice(len(names), k, replace=False))]
        print()
        check_sionna_scenes(scenes_dir, sionna_names, alts, phys["frequency"], cfg.get("SCATTERING_COEFFICIENT", 0.0))
        sionna_set = set(sionna_names)
        if args.full or args.only:
            phys_names = names
        else:
            fams = gen.ID_FAMILIES + gen.OOD_FAMILIES
            per_fam = max(1, args.n_physics // len(fams))
            phys_names = []
            for fam in fams:
                cand = [e["name"] for e in selected if e["family"] == fam]
                phys_names += [cand[i] for i in sorted(rng.choice(len(cand), min(per_fam, len(cand)),
                                                                  replace=False))]
        print()
        if args.exhaustive:
            physics_stations = check_physics(scenes_dir, phys_names, phys, alts, "all", test_alts=alts,
                                             n_depth=len(phys_names), tx_batch=args.tx_batch)
        else:
            physics_stations = check_physics(scenes_dir, phys_names, phys, alts, args.stations,
                                             n_depth=args.n_depth, tx_batch=args.tx_batch,
                                             dataset_choice=dataset_choice)

    ds_stats = {}
    if not args.only:
        if ds_dir.exists() and any(ds_dir.glob("*.npz")):
            print()
            ds_stats = check_dataset(scenes_dir, ds_dir, index, alts, cfg)
        else:
            print("\n(no dataset/ folder yet, dataset checks skipped)")
        statistics(scenes_dir, index, ds_stats, args.report)

    write_results(scenes_dir, names, sionna_set, physics_stations, bool(args.only))
    c = R.counts
    print(f"\n{c['pass']} checks passed, {c['warn']} warnings, {c['fail']} failed "
          f"({time.perf_counter() - t0:.0f} s)")
    sys.exit(1 if c["fail"] else 0)


if __name__ == "__main__":
    main()
