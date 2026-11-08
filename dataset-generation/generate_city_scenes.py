"""
Random city generator for the AIRWISE project.

Builds many synthetic urban scenes (128 x 128 m) in the same format as the
student_datasets scenes, so Sionna RT, check_city_scenes.py and the notebook work on them.

Per scene it writes:
    <name>.xml                 Sionna / Mitsuba scene description
    meshes/ground.ply          ground surface
    meshes/buildings.ply       all buildings in one file, every building is ONE closed mesh (no inner faces)
    height_map.npz             x, y (cell centres), 128 x 128 maps:
                                   terrain     ground height at the cell centre
                                   surface     ground + buildings at the cell centre
                                   occupancy   share of the cell covered by buildings (0 to 1, 4 x 4 sub-samples)
    sources.npz                candidate ground stations: ids, positions, height_agl, ground_z,
                               wall_distance (m to the nearest wall), station_class (index into STATION_CLASSES)
    meta.json                  family, parameters, statistics and the exact geometry of every building
and globally:
    index.json                 list of scenes with family and split (train / val / test / ood)
    config.json                parameters used
    overview.png               top views of a few scenes per family

Layout families
    in distribution (split into train / val / test):
        blocks      scattered buildings of random size, orientation and height (boxes, L shapes, podium + tower)
        grid        street grid (Manhattan style), rotated, blocks cut into lots (lots are cut at the area border)
        downtown    dense centre with tall buildings, lower towards the edges
    out of distribution (only used to test generalisation, split = "ood"):
        canyon      long parallel buildings with narrow streets between them (rows are cut at the area border)
        courtyard   ring / U shaped buildings around courtyards
        hilly       "blocks" on hilly terrain

Rules every scene follows (so the ray tracer and the height map agree):
    - two different buildings are at least MIN_GAP apart (no slots thinner than the height map can show)
    - no roof is closer than ROOF_CLEARANCE to one of the DRONE_ALTITUDES (radio map planes)
    - parts of one building whose heights differ by less than HEIGHT_MERGE share one roof level
    - stations are at least PLANE_CLEARANCE below the lowest drone altitude, BUILDING_MARGIN from every wall,
      EDGE_MARGIN from the border and STATION_MIN_SPACING from each other

Usage (from this folder):
    python generate_city_scenes.py                       # default: 450 id scenes + 90 ood scenes
    python generate_city_scenes.py --n-id 20 --n-ood 5   # quick test
    python generate_city_scenes.py --clean               # also delete old scene folders not in the new index
    python generate_city_scenes.py --workers 1           # one process (same scenes, slower)
Then check them with check_city_scenes.py (mesh against height map in Sionna, physics, ...).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import parallel_map

# ---------------------------------------------------------------- settings
BASE_SEED = 2026
AREA_SIDE = 128.0            # m, scene is AREA_SIDE x AREA_SIDE, centred on (0, 0)
CELL = 1.0                   # m, pixel size of the height map (and of the radio maps)
SUPERSAMPLE = 4              # sub-samples per cell side for the occupancy map

HEIGHT_RANGE = (6.0, 55.0)   # m, building height limits
MIN_GAP = 3.0                # m, minimum space between two different buildings (every family)
MIN_LOT_SIDE = 8.0           # m, a lot cut at the area border is dropped when it gets thinner than this
MIN_COURTYARD = 8.0          # m, minimum inner width of a courtyard
HEIGHT_MERGE = 1.0           # m, parts of one building closer in height than this get the same roof

DRONE_ALTITUDES = (15.0, 30.0, 45.0, 60.0)   # m, radio map planes (default of make_dataset.py)
ROOF_CLEARANCE = 1.0         # m, no roof closer than this to a drone altitude
PLANE_CLEARANCE = 5.0        # m, stations stay at least this far below the lowest drone altitude

N_STATIONS = 24              # candidate ground stations saved per scene
STATION_HEIGHT_AGL = (1.0, 10.0)   # m above ground (same as the student datasets)
STATION_MIN_SPACING = 5.0    # m, between two stations of the same scene
EDGE_MARGIN = 5.0            # m, stations stay away from the scene border
BUILDING_MARGIN = 2.0        # m, and from the walls
OPEN_DISTANCE = 10.0         # m, a station at least this far from every wall is "open"
STATION_CLASSES = ["open", "street", "courtyard"]   # courtyard = free space closed in by buildings

GROUND_MATERIAL = "medium_dry_ground"   # ITU, valid 1 to 10 GHz
BUILDING_MATERIAL = "concrete"          # ITU
SCATTERING_COEFFICIENT = 0.3   # share of the reflected energy scattered diffusely by rough surfaces (0 = mirror);
                               # measured building facades are typically 0.2 to 0.5. Used when the ray tracer
                               # runs with diffuse reflections (make_dataset.py does by default)

MIN_BUILDINGS = dict(blocks=3, grid=3, downtown=3, canyon=4, courtyard=1, hilly=3)

ID_FAMILIES = ["blocks", "grid", "downtown"]
OOD_FAMILIES = ["canyon", "courtyard", "hilly"]
ID_SPLIT = (0.8, 0.1, 0.1)   # train, val, test fractions of each in distribution family

HALF = AREA_SIDE / 2
N_CELLS = int(round(AREA_SIDE / CELL))
SNAP_DIGITS = 5              # coordinates are rounded to 10 um when building meshes
CELL_CENTRE_CLEARANCE = 1e-3 # m, no wall passes closer than this to a cell centre (see clear_of_cell_centres)
NUDGE_RADII = (2e-3, 4e-3, 8e-3, 12e-3, 20e-3)   # m, small moves tried to get a building clear of cell centres
GAP_SLACK = 0.05             # m, added to MIN_GAP and to the border margin when placing, so a nudged
                             # building (at most 2 cm) still keeps MIN_GAP and stays inside the area
BORDER_MARGIN = 1.0          # m, buildings stay this far from the area border


def snap(z):
    return round(float(z), SNAP_DIGITS)


# ---------------------------------------------------------------- geometry helpers
def rot(ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s], [s, c]])


def rect_corners(cx, cy, w, d, ang):
    local = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    return local @ rot(ang).T + [cx, cy]


def _local_uv(px, py, p):
    c, s = np.cos(p["ang"]), np.sin(p["ang"])
    dx, dy = px - p["cx"], py - p["cy"]
    return c * dx + s * dy, -s * dx + c * dy


def inside_rect(px, py, p, grow=0.0):
    """True where the points (px, py) are inside the rotated rectangle p (dict cx, cy, w, d, ang)."""
    u, v = _local_uv(px, py, p)
    return (np.abs(u) <= p["w"] / 2 + grow) & (np.abs(v) <= p["d"] / 2 + grow)


def inside_rect_on_grid(gx, gy, p, grow=0.0):
    """inside_rect on the grid meshgrid(gx, gy) (sorted 1D coordinates), evaluated only on the window the
    rectangle can reach. Returns (row slice, column slice, mask of that window); the rest of the grid is False."""
    c = rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"])
    m = 1.5 * grow + 1e-6        # the grown rectangle lies within its corners' bounding box + grow * sqrt(2)
    cols = slice(np.searchsorted(gx, c[:, 0].min() - m), np.searchsorted(gx, c[:, 0].max() + m, "right"))
    rows = slice(np.searchsorted(gy, c[:, 1].min() - m), np.searchsorted(gy, c[:, 1].max() + m, "right"))
    X, Y = np.meshgrid(gx[cols], gy[rows])
    return rows, cols, inside_rect(X, Y, p, grow)


def rect_distance(px, py, p):
    """Euclidean distance from the points (px, py) to the rotated rectangle p (0 inside)."""
    u, v = _local_uv(px, py, p)
    return np.hypot(np.maximum(np.abs(u) - p["w"] / 2, 0.0), np.maximum(np.abs(v) - p["d"] / 2, 0.0))


def outline(cx, cy, w, d, ang):
    """Rotated rectangle with its corners and circumradius, for fast overlap tests."""
    return dict(cx=float(cx), cy=float(cy), w=float(w), d=float(d), ang=float(ang),
                corners=rect_corners(cx, cy, w, d, ang), r=float(np.hypot(w, d) / 2))


def outlines_overlap(a, b):
    """Separating axis test for two rotated rectangles made by outline()."""
    if np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) >= a["r"] + b["r"]:
        return False
    for ang in (a["ang"], b["ang"]):
        c, s = np.cos(ang), np.sin(ang)
        for axis in ((c, s), (-s, c)):
            pa, pb = a["corners"] @ axis, b["corners"] @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


# ---------------------------------------------------------------- buildings
# A building has a centre, an orientation and a list of axis aligned rectangles in its own frame.
# Every rectangle has a roof height h above the building's ground. Where rectangles overlap the
# highest roof wins, so a tower on a podium is just a smaller, higher rectangle inside a bigger one.
# The whole building is turned into ONE closed mesh (building_mesh), so there are no inner or
# coincident faces that could give wrong reflections or false diffraction edges.
def rect(x0, x1, y0, y1, h):
    return dict(x0=float(min(x0, x1)), x1=float(max(x0, x1)),
                y0=float(min(y0, y1)), y1=float(max(y0, y1)), h=float(h))


def building(cx, cy, ang, rects):
    return dict(cx=float(cx), cy=float(cy), ang=float(ang), rects=rects)


def world_parts(b):
    """The rectangles of a building as rotated rectangles in world coordinates."""
    R = rot(b["ang"])
    out = []
    for r in b["rects"]:
        c = R @ np.array([(r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2])
        out.append(dict(cx=b["cx"] + c[0], cy=b["cy"] + c[1], w=r["x1"] - r["x0"], d=r["y1"] - r["y0"],
                        ang=b["ang"]))
    return out


def fits(b, margin=BORDER_MARGIN + GAP_SLACK):
    lim = HALF - margin
    return all(np.abs(rect_corners(p["cx"], p["cy"], p["w"], p["d"], p["ang"])).max() <= lim
               for p in world_parts(b))


def _wall_near_cell_centre(p, xc):
    r = np.hypot(p["w"], p["d"]) / 2 + 1.0
    X, Y = np.meshgrid(xc[np.abs(xc - p["cx"]) <= r], xc[np.abs(xc - p["cy"]) <= r])
    u, v = _local_uv(X, Y, p)
    tol = CELL_CENTRE_CLEARANCE
    du, dv = np.abs(np.abs(u) - p["w"] / 2), np.abs(np.abs(v) - p["d"] / 2)
    return bool(((du < tol) & (np.abs(v) <= p["d"] / 2 + tol)).any()
                or ((dv < tol) & (np.abs(u) <= p["w"] / 2 + tol)).any())


def clear_of_cell_centres(b, xc):
    """Move a building by at most 2 cm until no wall passes within CELL_CENTRE_CLEARANCE of a cell centre.
    Otherwise the height map pixel and a ray cast against the float32 mesh can disagree about which side
    of the wall the centre is on. GAP_SLACK keeps the moved building MIN_GAP from the others and inside."""
    def near():
        return any(_wall_near_cell_centre(p, xc) for p in world_parts(b))

    if not near():
        return
    cx0, cy0 = b["cx"], b["cy"]
    for r in NUDGE_RADII:
        for k in range(8):                       # 8 directions, none along the axes
            a = k * np.pi / 4 + 0.3
            b["cx"], b["cy"] = cx0 + r * float(np.cos(a)), cy0 + r * float(np.sin(a))
            if not near():
                return
    raise RuntimeError("could not move a building clear of the cell centres")


def footprint_grid(rects, key="h"):
    """Cut the building frame along every rectangle edge. Returns the cell edges xs, ys and the roof
    height of every cell (NaN where there is no building)."""
    xs = np.unique(np.round([v for r in rects for v in (r["x0"], r["x1"])], SNAP_DIGITS))
    ys = np.unique(np.round([v for r in rects for v in (r["y0"], r["y1"])], SNAP_DIGITS))
    xm, ym = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    top = np.full((len(ym), len(xm)), np.nan)
    for r in rects:
        ix = np.flatnonzero((xm > r["x0"]) & (xm < r["x1"]))
        iy = np.flatnonzero((ym > r["y0"]) & (ym < r["y1"]))
        top[np.ix_(iy, ix)] = np.fmax(top[np.ix_(iy, ix)], r[key])
    return xs, ys, top


def footprint_area(rects):
    xs, ys, top = footprint_grid(rects)
    return float((np.diff(ys)[:, None] * np.diff(xs)[None, :])[~np.isnan(top)].sum())


def building_mesh(b, z_bottom):
    """One closed, consistently oriented (outward) triangle mesh for the whole building.
    Roofs and floors are one quad per footprint cell; walls are put only where the height changes,
    and every vertical edge is split at all roof levels that meet there, so there are no T-junctions."""
    xs, ys, top = footprint_grid(b["rects"], key="z_top")
    nyc, nxc = top.shape
    zb = snap(z_bottom)

    def cell(i, j):
        if 0 <= i < nxc and 0 <= j < nyc and not np.isnan(top[j, i]):
            return snap(top[j, i])
        return None

    levels = {}
    for j in range(nyc + 1):
        for i in range(nxc + 1):
            hs = [cell(i + a, j + c) for a in (-1, 0) for c in (-1, 0)]
            hs = [h for h in hs if h is not None]
            if hs:
                levels[(i, j)] = sorted({zb, *hs})

    verts, vid, faces = [], {}, []

    def V(i, j, z):
        k = (i, j, z)
        if k not in vid:
            vid[k] = len(verts)
            verts.append((xs[i], ys[j], z))
        return vid[k]

    def quad(a, b_, c, d):          # corners counter clockwise seen from outside
        faces.append((a, b_, c))
        faces.append((a, c, d))

    def wall(pk, qk, lo, hi, flip):
        """Vertical wall between the vertical edges at grid vertices pk and qk, from lo to hi.
        Unflipped, the normal is (q - p) x z rotated: +x for a wall along +y, -y for a wall along +x."""
        P = [z for z in levels[pk] if lo <= z <= hi]
        Q = [z for z in levels[qk] if lo <= z <= hi]
        p = q = 0
        while p < len(P) - 1 or q < len(Q) - 1:
            if q < len(Q) - 1 and (p == len(P) - 1 or Q[q + 1] <= P[p + 1]):
                t = (V(*pk, P[p]), V(*qk, Q[q]), V(*qk, Q[q + 1]))
                q += 1
            else:
                t = (V(*pk, P[p]), V(*qk, Q[q]), V(*pk, P[p + 1]))
                p += 1
            faces.append(t[::-1] if flip else t)

    def wall_range(h1, h2):
        """Wall between a cell of height h1 and one of height h2 (None = no building).
        Returns (lo, hi, first side is higher) or None when there is no wall."""
        if h1 == h2:
            return None
        if h1 is None or h2 is None:
            return zb, (h1 if h2 is None else h2), h2 is None
        return min(h1, h2), max(h1, h2), h1 > h2

    for j in range(nyc):
        for i in range(nxc):
            t = cell(i, j)
            if t is not None:
                quad(V(i, j, t), V(i + 1, j, t), V(i + 1, j + 1, t), V(i, j + 1, t))      # roof, +z
                quad(V(i, j, zb), V(i, j + 1, zb), V(i + 1, j + 1, zb), V(i + 1, j, zb))  # floor, -z
    for i in range(nxc + 1):            # walls on the lines x = xs[i]
        for j in range(nyc):
            w = wall_range(cell(i - 1, j), cell(i, j))
            if w:
                wall((i, j), (i, j + 1), w[0], w[1], flip=not w[2])   # normal +x if the left cell is higher
    for j in range(nyc + 1):            # walls on the lines y = ys[j]
        for i in range(nxc):
            w = wall_range(cell(i, j - 1), cell(i, j))
            if w:
                wall((i, j), (i + 1, j), w[0], w[1], flip=w[2])       # normal +y if the lower cell is higher

    v = np.array(verts, float)
    v[:, :2] = v[:, :2] @ rot(b["ang"]).T + [b["cx"], b["cy"]]
    return v, np.array(faces, np.int64)


def settle_roofs(g_max, hs):
    """Absolute roof heights of one building from the heights hs above its ground g_max:
    moved away from the drone altitudes, then parts with almost the same height merged into one roof."""
    lo, hi = g_max + 1.0, g_max + HEIGHT_RANGE[1]
    z = []
    for h in hs:
        zz = g_max + h
        for a in DRONE_ALTITUDES:
            if abs(zz - a) < ROOF_CLEARANCE:
                near, far = (a - ROOF_CLEARANCE, a + ROOF_CLEARANCE) if zz < a else \
                            (a + ROOF_CLEARANCE, a - ROOF_CLEARANCE)
                zz = near if lo <= near <= hi else far
                break
        z.append(zz)
    z = np.array(z, float)
    out, group = z.copy(), []
    for k in np.argsort(z, kind="stable"):
        if group and z[k] - z[group[-1]] >= HEIGHT_MERGE:
            out[group] = z[group].max()      # the max is one of the moved roofs, so it is clear of the planes
            group = []
        group.append(k)
    out[group] = z[group].max()
    return [float(x) for x in out]


# ---------------------------------------------------------------- building shapes
def box_building(cx, cy, w, d, ang, h):
    return building(cx, cy, ang, [rect(-w / 2, w / 2, -d / 2, d / 2, h)])


def l_building(rng, cx, cy, w, d, ang, h):
    """L shape: a full width wing and a narrower wing next to it, the second one can be lower."""
    cw, cd = rng.uniform(0.35, 0.6) * w, rng.uniform(0.35, 0.6) * d
    sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
    h2 = h * rng.uniform(0.6, 1.0)
    return building(cx, cy, ang, [
        rect(-sx * w / 2, sx * w / 2, -sy * d / 2, sy * (d / 2 - cd), h),
        rect(-sx * w / 2, sx * (w / 2 - cw), sy * (d / 2 - cd), sy * d / 2, h2)])


def podium_tower(rng, cx, cy, w, d, ang, h):
    """Low wide base with a taller narrower tower on top."""
    h_p = rng.uniform(0.3, 0.55) * h
    tw, td = rng.uniform(0.35, 0.6) * w, rng.uniform(0.35, 0.6) * d
    ox = rng.uniform(-1, 1) * (w - tw) / 2 * 0.8
    oy = rng.uniform(-1, 1) * (d - td) / 2 * 0.8
    return building(cx, cy, ang, [rect(-w / 2, w / 2, -d / 2, d / 2, h_p),
                                  rect(ox - tw / 2, ox + tw / 2, oy - td / 2, oy + td / 2, h)])


def ring_building(rng, cx, cy, w, d, ang, h, t):
    """Four bars around a courtyard, sometimes one bar missing (U shape)."""
    t = max(min(t, (min(w, d) - MIN_COURTYARD) / 2), 3.0)   # courtyard stays at least MIN_COURTYARD wide
    bars = [rect(-w / 2, w / 2, -d / 2, -d / 2 + t, 0), rect(-w / 2, w / 2, d / 2 - t, d / 2, 0),
            rect(-w / 2, -w / 2 + t, -d / 2 + t, d / 2 - t, 0), rect(w / 2 - t, w / 2, -d / 2 + t, d / 2 - t, 0)]
    if rng.random() < 0.4:
        bars.pop(rng.integers(4))
    for r in bars:
        r["h"] = float(h * rng.uniform(0.85, 1.0))
    return building(cx, cy, ang, bars)


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
    """Random buildings at least MIN_GAP apart (exact rotated rectangle test). Used by blocks, downtown, hilly."""
    coverage = coverage if coverage is not None else rng.uniform(0.08, 0.35)
    theta = rng.uniform(0, np.pi / 2)
    buildings, outlines, area = [], [], 0.0
    for _ in range(3000):
        if area >= coverage * AREA_SIDE ** 2:
            break
        w, d = rng.uniform(8, 28), rng.uniform(8, 28)
        ang = rng.uniform(0, np.pi) if ang_mode == "random" else theta + rng.normal(0, 0.12)
        m = min(w, d) / 2
        cx, cy = rng.uniform(-HALF + m, HALF - m, 2)
        # both outlines grown by MIN_GAP / 2 on every side: not overlapping means at least MIN_GAP apart
        o = outline(cx, cy, w + MIN_GAP + GAP_SLACK, d + MIN_GAP + GAP_SLACK, ang)
        if any(outlines_overlap(o, o2) for o2 in outlines):
            continue
        h = height_fn(rng, cx, cy) if height_fn else random_height(rng)
        b = random_shape(rng, cx, cy, w, d, ang, h)
        if not fits(b):
            continue
        buildings.append(b)
        outlines.append(o)
        area += footprint_area(b["rects"])
    return buildings, dict(coverage_target=float(coverage), coverage_reached=float(area / AREA_SIDE ** 2),
                           theta=float(theta))


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


def row_range(c, s, v_edges, lim):
    """Interval [lo, hi] of positions u along a row (row axis = (c, s), across = (-s, c)) for which
    a box with its two long edges at across-positions v_edges has all corners inside the square
    [-lim, lim]^2. None if the row does not reach into the area."""
    lo, hi = -np.inf, np.inf
    for v in v_edges:
        for a, b in ((c, -s), (s, c)):          # world x = c*u - s*v,  world y = s*u + c*v
            if abs(a) < 1e-9:
                if abs(b * v) > lim:
                    return None
            else:
                t1, t2 = (-lim - b * v) / a, (lim - b * v) / a
                lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    return (lo, hi) if hi - lo > 0 else None


def clip_lot(c, s, u0, u1, v0, v1, lim):
    """Cut a lot [u0, u1] x [v0, v1] of a grid rotated by (c, s) so it fits in [-lim, lim]^2, either
    along u or along v (whichever keeps more area). Returns (u0, u1, v0, v1, was_cut) or None."""
    best = None
    ur = row_range(c, s, (v0, v1), lim)
    if ur is not None:
        a, b = max(u0, ur[0]), min(u1, ur[1])
        if b - a >= MIN_LOT_SIDE:
            best = (a, b, v0, v1)
    vr = row_range(-s, c, (-u0, -u1), lim)      # same test with the axes swapped (across position = -u)
    if vr is not None:
        a, b = max(v0, vr[0]), min(v1, vr[1])
        if b - a >= MIN_LOT_SIDE and (best is None or (b - a) * (u1 - u0) > (best[1] - best[0]) * (best[3] - best[2])):
            best = (u0, u1, a, b)
    if best is None:
        return None
    return (*best, best != (u0, u1, v0, v1))


def layout_grid(rng):
    """Street grid, rotated, every block cut into 1 to 4 lots. Lots crossing the border are cut, not dropped."""
    street = rng.uniform(8, 16)
    bw, bd = rng.uniform(26, 48, 2)
    theta = rng.uniform(-np.pi / 4, np.pi / 4)
    c, s = np.cos(theta), np.sin(theta)
    lim = HALF - BORDER_MARGIN - GAP_SLACK - 1e-6   # same margin as fits()
    ox, oy = rng.uniform(0, bw + street), rng.uniform(0, bd + street)
    n = int(HALF * 1.6 / min(bw, bd)) + 2
    buildings, n_cut = [], 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            bx, by = i * (bw + street) + ox, j * (bd + street) + oy
            if np.hypot(bx, by) > HALF * 1.5:
                continue
            block_h = random_height(rng, 18.0)
            gap = rng.uniform(MIN_GAP + GAP_SLACK, MIN_GAP + 2.0)
            u = rng.random()
            nx, ny = (1, 1) if u < 0.35 else ((2, 1) if bw >= bd else (1, 2)) if u < 0.75 else (2, 2)
            lw, ld = bw / nx, bd / ny
            for a in range(nx):
                for b in range(ny):
                    h = float(np.clip(block_h * rng.uniform(0.75, 1.25), *HEIGHT_RANGE))
                    u0 = bx - bw / 2 + lw * a + gap / 2
                    v0 = by - bd / 2 + ld * b + gap / 2
                    lot = clip_lot(c, s, u0, u0 + lw - gap, v0, v0 + ld - gap, lim)
                    if lot is None:
                        continue
                    u0_, u1_, v0_, v1_, cut = lot
                    wx, wy = rot(theta) @ np.array([(u0_ + u1_) / 2, (v0_ + v1_) / 2])
                    if cut:     # a building cut by the border stays a plain box
                        bld = box_building(wx, wy, u1_ - u0_, v1_ - v0_, theta, h)
                    else:
                        bld = random_shape(rng, wx, wy, u1_ - u0_, v1_ - v0_, theta, h)
                    if fits(bld):
                        buildings.append(bld)
                        n_cut += cut
    return buildings, dict(street=float(street), block=[float(bw), float(bd)], theta=float(theta),
                           lots_cut_at_border=int(n_cut))


def layout_canyon(rng):
    """Long parallel buildings with narrow streets: not seen in training.
    Rows are cut at the border of the area instead of being dropped when they stick out."""
    lim = HALF - BORDER_MARGIN - GAP_SLACK - 1e-6   # same margin as fits()
    theta = rng.uniform(0, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    street, depth = rng.uniform(10, 22), rng.uniform(10, 20)
    # the rotated square is at least 2*lim wide across the rows; use as many rows as fit (2 to 4)
    width = 2 * lim * (abs(c) + abs(s))
    n_max = max(2, int((width + street) // (depth + street)))
    n_rows = int(rng.integers(2, min(4, n_max) + 1))
    buildings = []
    for j in range(n_rows):
        ly = (j - (n_rows - 1) / 2) * (depth + street) + rng.uniform(-3, 3)
        u_rng = row_range(c, s, (ly - depth / 2, ly + depth / 2), lim)
        if u_rng is None:
            continue
        row_h = rng.uniform(12, 45)
        x = u_rng[0] - rng.uniform(0, 40)        # random phase: the first building is cut at the border
        while x < u_rng[1]:
            seg = rng.uniform(25, 70)
            u0, u1 = max(x, u_rng[0]), min(x + seg, u_rng[1])
            if u1 - u0 >= 12.0:                  # skip slivers
                h = float(np.clip(row_h * rng.uniform(0.8, 1.2), *HEIGHT_RANGE))
                wx, wy = rot(theta) @ np.array([(u0 + u1) / 2, ly])
                bld = box_building(wx, wy, u1 - u0, depth, theta, h)
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
    max_size = (2 * (HALF - BORDER_MARGIN - GAP_SLACK) / spread - (n_max - 1) * street) / n_max
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
    """Scene description for Sionna / Mitsuba. The material ids must NOT start with "mat-itu" / "itu": Sionna
    rewrites such materials when loading and keeps only their type and thickness (the scattering coefficient
    would be lost). The material name is the id without "mat-" and must differ from the shape ids."""
    lines = ['<scene version="2.1.0">']
    for mat_id, itu in [("mat-ground-surface", GROUND_MATERIAL), ("mat-building-walls", BUILDING_MATERIAL)]:
        lines += [f'    <bsdf type="itu-radio-material" id="{mat_id}">',
                  f'        <string name="type" value="{itu}"/>',
                  '        <float name="thickness" value="0.2"/>',
                  f'        <float name="scattering_coefficient" value="{SCATTERING_COEFFICIENT}"/>', '    </bsdf>']
    for shape_id, ply, mat_id in [("ground", "meshes/ground.ply", "mat-ground-surface"),
                                  ("buildings", "meshes/buildings.ply", "mat-building-walls")]:
        lines += [f'    <shape type="ply" id="{shape_id}">',
                  f'        <string name="filename" value="{ply}"/>',
                  f'        <ref id="{mat_id}" name="bsdf"/>', '    </shape>']
    lines.append("</scene>")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- stations
def enclosed_cells(free):
    """Free cells that cannot reach the border of the area (4-connected flood fill from the border)."""
    reach = np.zeros_like(free)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = free[0], free[-1], free[:, 0], free[:, -1]
    while True:
        grow = reach.copy()
        grow[1:] |= reach[:-1]
        grow[:-1] |= reach[1:]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= free
        if np.array_equal(grow, reach):
            return free & ~reach
        reach = grow


def place_stations(rng, parts, vx, vy, terrain_v, built):
    """Candidate ground stations: inside the area, away from the walls, from each other and from the
    lowest radio map plane. Every station gets its distance to the nearest wall and a class."""
    cand = rng.uniform(-HALF + EDGE_MARGIN, HALF - EDGE_MARGIN, (4000, 2))
    dist = np.full(len(cand), np.inf)
    for p in parts:
        dist = np.minimum(dist, rect_distance(cand[:, 0], cand[:, 1], p))
    z_ground = ground_height(cand[:, 0], cand[:, 1], vx, vy, terrain_v)
    h_max = np.minimum(STATION_HEIGHT_AGL[1], min(DRONE_ALTITUDES) - PLANE_CLEARANCE - z_ground)
    ok = (dist >= BUILDING_MARGIN) & (h_max >= STATION_HEIGHT_AGL[0])
    chosen = []
    for k in np.flatnonzero(ok):
        if all(np.hypot(*(cand[k] - cand[m])) >= STATION_MIN_SPACING for m in chosen):
            chosen.append(k)
            if len(chosen) == N_STATIONS:
                break
    chosen = np.array(chosen, int)
    xy, dist, z_ground = cand[chosen], dist[chosen], z_ground[chosen]
    h_agl = rng.uniform(STATION_HEIGHT_AGL[0], h_max[chosen])

    closed = enclosed_cells(~built)
    ix = np.clip(((xy[:, 0] + HALF) // CELL).astype(int), 0, N_CELLS - 1)
    iy = np.clip(((xy[:, 1] + HALF) // CELL).astype(int), 0, N_CELLS - 1)
    cls = np.where(closed[iy, ix], STATION_CLASSES.index("courtyard"),
                   np.where(dist >= OPEN_DISTANCE, STATION_CLASSES.index("open"),
                            STATION_CLASSES.index("street"))).astype(np.int8)
    return dict(ids=np.arange(len(xy)), positions=np.column_stack([xy, z_ground + h_agl]),
                height_agl=h_agl, ground_z=z_ground, wall_distance=dist, station_class=cls)


# ---------------------------------------------------------------- one scene
def build_scene(out_dir, name, family, rng):
    scene_dir = out_dir / name
    (scene_dir / "meshes").mkdir(parents=True, exist_ok=True)

    for _ in range(20):     # draw again if the layout came out too empty
        buildings, layout_info = LAYOUTS[family](rng)
        if len(buildings) >= MIN_BUILDINGS[family]:
            break
    else:
        raise RuntimeError(f"{name}: layout '{family}' gave fewer than {MIN_BUILDINGS[family]} buildings "
                           f"in 20 attempts, check the layout parameters")

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
    xs = (np.arange(N_CELLS * SUPERSAMPLE) + 0.5) * (CELL / SUPERSAMPLE) - HALF   # occupancy sub-samples

    surface = terrain_c.copy()
    occupied = np.zeros((len(xs), len(xs)), bool)
    all_vertices, all_faces, n_vert, parts, geometry, roofs_moved = [], [], 0, [], [], 0
    for b in buildings:
        clear_of_cell_centres(b, xc)
        bparts = world_parts(b)
        # ground reference of the whole building: the floor goes below the lowest ground, roofs are
        # measured from the highest ground under the building
        near = np.zeros(terrain_v.shape, bool)
        for p in bparts:
            rows, cols, m = inside_rect_on_grid(vx, vy, p, grow=1.0)
            near[rows, cols] |= m
        g = terrain_v[near] if near.any() else np.array([0.0])
        g_min, g_max = float(g.min()), float(g.max())
        z_bottom = g_min - 0.5
        tops = settle_roofs(g_max, [r["h"] for r in b["rects"]])
        for r, z in zip(b["rects"], tops):
            roofs_moved += abs(z - (g_max + r["h"])) > 1e-9
            r["z_top"] = z
        v, f = building_mesh(b, z_bottom)
        all_vertices.append(v)
        all_faces.append(f + n_vert)
        n_vert += len(v)
        for p, r in zip(bparts, b["rects"]):
            rows, cols, m = inside_rect_on_grid(xc, yc, p)
            win = surface[rows, cols]                     # a view: writing into it updates surface
            win[m] = np.maximum(win[m], r["z_top"])
            rows, cols, m = inside_rect_on_grid(xs, xs, p)
            occupied[rows, cols] |= m
        parts += bparts
        geometry.append(dict(cx=b["cx"], cy=b["cy"], ang=b["ang"], z_bottom=z_bottom, ground_max=g_max,
                             rects=b["rects"]))
    occupancy = occupied.reshape(N_CELLS, SUPERSAMPLE, N_CELLS, SUPERSAMPLE).mean(axis=(1, 3))

    write_ply(scene_dir / "meshes" / "ground.ply", *ground_mesh(vx, vy, terrain_v, flat))
    if all_vertices:
        write_ply(scene_dir / "meshes" / "buildings.ply",
                  np.vstack(all_vertices), np.vstack(all_faces))
    else:   # keep the file valid, one tiny box far below the ground
        write_ply(scene_dir / "meshes" / "buildings.ply",
                  *box_mesh(rect_corners(0, 0, 1, 1, 0), -50, -49))
    write_xml(scene_dir / f"{name}.xml")
    np.savez(scene_dir / "height_map.npz", x=xc, y=yc, terrain=terrain_c, surface=surface,
             occupancy=occupancy.astype(np.float32))

    built = surface - terrain_c > 0.5
    src = place_stations(rng, parts, vx, vy, terrain_v, built)
    np.savez(scene_dir / "sources.npz", **src)

    counts = np.bincount(src["station_class"], minlength=len(STATION_CLASSES))
    meta = dict(name=name, family=family, num_buildings=len(buildings), num_parts=len(parts),
                num_meshes=len(buildings),           # one closed mesh per building in buildings.ply
                built_fraction=float(built.mean()),
                occupancy_fraction=float(occupancy.mean()),
                max_height=float((surface - terrain_c).max()),
                mean_building_height=float((surface - terrain_c)[built].mean()) if built.any() else 0.0,
                roofs_moved_off_altitudes=int(roofs_moved),
                num_stations=int(len(src["ids"])),
                station_classes={c: int(k) for c, k in zip(STATION_CLASSES, counts)},
                layout=layout_info, terrain=terrain_info, buildings=geometry)
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta, surface - terrain_c, src["positions"][:, :2]


def make_one(job):
    """Build one scene and return its index.json entry. Every scene has its own seed (seed, family, number),
    so scenes can be built in any order, in parallel processes."""
    out_dir, name, family, split, seed, fam_id, i = job
    meta, _, _ = build_scene(out_dir, name, family, np.random.default_rng([seed, fam_id, i]))
    return dict(name=name, family=family, split=split, num_buildings=meta["num_buildings"],
                built_fraction=round(meta["built_fraction"], 3), max_height=round(meta["max_height"], 1),
                station_classes=meta["station_classes"])


# ---------------------------------------------------------------- overview figure
def overview(out_dir, index, per_family=4):
    families = ID_FAMILIES + OOD_FAMILIES
    fig, axes = plt.subplots(len(families), per_family, figsize=(2.6 * per_family, 2.6 * len(families)),
                             constrained_layout=True, squeeze=False)
    for r, fam in enumerate(families):
        names = [e["name"] for e in index if e["family"] == fam][:per_family]
        for c in range(per_family):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if c >= len(names):
                ax.axis("off")
                continue
            with np.load(out_dir / names[c] / "height_map.npz") as hm:
                h = hm["surface"] - hm["terrain"]
            with np.load(out_dir / names[c] / "sources.npz") as f:
                src, cls = f["positions"], f["station_class"]
            ax.imshow(np.where(h > 0.5, h, np.nan), origin="lower", cmap="magma", vmin=0, vmax=55,
                      extent=[-HALF, HALF, -HALF, HALF])
            ax.set_facecolor("#cfe3cf" if fam != "hilly" else "#d9d2b6")
            ax.scatter(src[:, 0], src[:, 1], s=5, c=np.array(["tab:blue", "tab:cyan", "tab:red"])[cls])
            if c == 0:
                ax.set_ylabel(fam, fontsize=11)
    fig.suptitle("Generated scenes: building height in m (colour), stations: open (blue), street (cyan), "
                 "courtyard (red)", fontsize=9)
    fig.savefig(out_dir / "overview.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "scenes_city"))
    ap.add_argument("--n-id", type=int, default=150, help="scenes per in distribution family")
    ap.add_argument("--n-ood", type=int, default=30, help="scenes per out of distribution family")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--clean", action="store_true",
                    help="delete scene folders in --out that are not part of this run")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="scenes built in parallel processes (1 = no process pool); the output is the same")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [(f, args.n_id) for f in ID_FAMILIES] + [(f, args.n_ood) for f in OOD_FAMILIES]

    expected = {f"{family}_{i:04d}" for family, n in plan for i in range(n)}
    stale = sorted(p.name for p in out_dir.iterdir()
                   if p.is_dir() and (p / "meta.json").exists() and p.name not in expected)
    if stale and args.clean:
        for s in stale:
            shutil.rmtree(out_dir / s)
        print(f"removed {len(stale)} old scene folders that are not part of this run")
    elif stale:
        print(f"WARNING: {len(stale)} old scene folders in {out_dir} are not part of this run "
              f"(e.g. {stale[:3]}); rerun with --clean to delete them, or the checker will complain")

    jobs = []
    for fam_id, (family, n) in enumerate(plan):
        for i in range(n):
            if family in OOD_FAMILIES:
                split = "ood"
            else:
                f = i / n
                split = "train" if f < ID_SPLIT[0] else "val" if f < ID_SPLIT[0] + ID_SPLIT[1] else "test"
            jobs.append((out_dir, f"{family}_{i:04d}", family, split, args.seed, fam_id, i))
    index = []
    for e in parallel_map(make_one, jobs, args.workers, min_jobs=20):     # ~0.1 s per scene
        index.append(e)
        print(f"      {len(index)}/{len(jobs)} {e['name']}        ", end="\r", flush=True)
    print(" " * 50, end="\r")

    for family, n in plan:
        sub = [e for e in index if e["family"] == family]
        if not sub:
            continue
        cls = {c: sum(e["station_classes"][c] for e in sub) for c in STATION_CLASSES}
        print(f"{family:10s} {n:4d} scenes, mean built fraction "
              f"{np.mean([e['built_fraction'] for e in sub]):.2f}, "
              f"mean max height {np.mean([e['max_height'] for e in sub]):.1f} m, stations {cls}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    config = {k: v for k, v in globals().items()
              if k.isupper() and k != "LAYOUTS" and isinstance(v, (int, float, str, tuple, list, dict))}
    config["SEED"] = args.seed
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    overview(out_dir, index)
    counts = {s: sum(e["split"] == s for e in index) for s in ["train", "val", "test", "ood"]}
    print(f"\n{len(index)} scenes written to {out_dir}  {counts}; check them with check_city_scenes.py")


if __name__ == "__main__":
    main()
