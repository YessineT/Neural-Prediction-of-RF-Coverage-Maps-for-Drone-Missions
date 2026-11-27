"""
Yessine's code, copied VERBATIM from rmtransformer_loss_and_size_study.ipynb (branch yessine), so that PMNet is
trained and measured exactly like his RMTransformer: data class, input channels (incl. line of sight), augmentation,
losses, treatment of no-signal cells, edge weighting, optimiser and schedule, training loop, metrics.
Only this header was added (imports, DEVICE, tqdm). Do not edit the functions below; if his notebook changes,
regenerate this file from it.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:              # progress bars are optional
    tqdm = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CFG = dict(
    # --- data
    repo_id="TYessine/airwise-rf-coverage-maps",
    revision=None,                # pin a commit hash or tag of the dataset repo for reproducibility
    data_dir=None,                # folder that already holds <split>.npz (None = download from the Hub)
    splits=("train", "val", "test", "ood"),
    study_dir="runs/loss_and_size_study",   # one sub-folder per experiment
    seed=0,
    # --- physics and normalisation
    freq_hz=2.4e9,
    floor_db=-150.0,              # dataset floor: -150 dB means no ray arrived
    max_db=-50.0,                 # top of the [0, 1] scale (highest gain in the data is about -54 dB)
    height_scale=60.0,            # m, heights, altitudes and station z are divided by this
    tx_sigma=1.0,                 # cells, width of the Gaussian station marker
    in_channels=("height", "rel_height", "valid", "tx", "fspl", "tx_z", "altitude"),
    # --- model
    model_size="tiny",            # "tiny": 32-256 channels, "small": 64-512, "paper": 128-1024 (as in the paper)
    window=8,                     # MaxViT block size P and grid size G
    head_dim=32,
    mbconv_expand=4,
    drop_path=0.1,
    # --- training (the RMTransformer paper uses 50 epochs and batch 8; all runs here use 20 and 16)
    epochs=20,
    batch_size=16,
    lr=1e-4,
    min_lr=1e-5,
    warmup_steps=500,
    weight_decay=0.0,             # 0 = plain Adam as in the paper
    grad_clip=1.0,
    augment=True,                 # random 90 degree rotations and mirroring (see the input section)
    amp=True,
    loss="mse",                   # "mse", "l1", "huber" or "combo" (set per experiment)
    huber_delta=0.05,             # Huber threshold in [0, 1] units: 0.05 = 5 dB
    loss_weights=dict(l1=0.4, mse=0.1, ssim=0.2, grad=0.3),   # combined loss (RadioMamba weights)
    floor_method="none",          # "none", "truncate", "censor", "two_output", "asymmetric" or "reweight" (section 10)
    floor_threshold_db=-130.0,    # path gain below which no drone link is possible (section 10)
    signal_loss_weight=0.1,       # weight of the signal / no-signal classification loss ("two_output")
    weak_db=-110.0,               # "asymmetric": cells weaker than this are weak-signal cells
    asym_factor=4.0,              # "asymmetric": an overestimation in a weak-signal cell counts this many times
    reweight_bin_db=5.0,          # "reweight": width of the path gain bins (dB)
    reweight_table=None,          # "reweight": weight of each bin, computed from the training data (section 10)
    select_metric="rmse_db",      # validation metric used to keep the best checkpoint of a run
    # --- edges of the coverage map (sections 11 and 12)
    edge_threshold_db=3.0,        # a cell is an edge where the ray traced map changes by >= this many dB per cell
    edge_width=2,                 # cells within this distance of an edge are "near an edge"
    edge_weight=1.0,              # weight of the cells near an edge in the loss (1 = no edge weighting)
    los_samples=96,               # points checked along each station-to-cell segment for the "los" channel
    eval_batch_size=32,
    max_val_maps=None,            # validate on an evenly spaced subset each epoch to save time (None = all)
    # --- evaluation
    coverage_thresholds_db=(-80.0, -100.0, -110.0),   # path gain needed for a link, e.g. -110 dB = 20 dBm Tx, -90 dBm Rx
    boundary_width=2,             # cells on each side of the coverage boundary counted as "boundary"
)


WAVELENGTH = 299_792_458.0 / CFG["freq_hz"]


class CoverageSplit:
    """One split held in CPU memory as torch tensors. Height maps are stored once per scene and paired with the
    maps through scene_idx, so the train split takes about 0.9 GB."""

    def __init__(self, path, name):
        self.name = name
        with np.load(path) as d:
            self.heights = torch.from_numpy(np.ascontiguousarray(d["heights"], dtype=np.float32))
            self.scene_idx = torch.from_numpy(np.ascontiguousarray(d["scene_idx"], dtype=np.int64))
            self.station = torch.from_numpy(np.ascontiguousarray(d["station_pos"], dtype=np.float32))
            self.altitude = torch.from_numpy(np.ascontiguousarray(d["altitude"], dtype=np.float32))
            self.target = torch.from_numpy(np.ascontiguousarray(d["target_db"], dtype=np.float32))
            self.rt_time = np.asarray(d["rt_time"], dtype=np.float32)
            self.scene_names = d["scene_names"].astype(str)
            self.scene_source = (d["scene_source"].astype(str) if "scene_source" in d.files
                                 else np.full(len(self.scene_names), "unknown"))
        si = self.scene_idx.numpy()
        self.family = np.array([n.rsplit("_", 1)[0] for n in self.scene_names])[si]   # per map
        self.source = self.scene_source[si]                                         # per map

    def __len__(self):
        return len(self.target)

    def batch(self, idx, device):
        """heights (B,H,W), station (B,3), altitude (B,), target_db (B,H,W) of the maps idx, on device."""
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return (self.heights[self.scene_idx[idx]].to(device, non_blocking=True),
                self.station[idx].to(device, non_blocking=True),
                self.altitude[idx].to(device, non_blocking=True),
                self.target[idx].to(device, non_blocking=True))

    def __repr__(self):
        fams, counts = np.unique(self.family, return_counts=True)
        return (f"{self.name}: {len(self.scene_names)} scenes, {len(self)} maps, "
                f"maps per family {dict(zip(fams.tolist(), counts.tolist()))}")


def db_to_unit(db, cfg=CFG):
    return (db - cfg["floor_db"]) / (cfg["max_db"] - cfg["floor_db"])


def unit_to_db(u, cfg=CFG):
    return u * (cfg["max_db"] - cfg["floor_db"]) + cfg["floor_db"]


def free_space_gain_db(d):
    return 20.0 * torch.log10(WAVELENGTH / (4.0 * math.pi * d))


def plane_geometry(heights, station, altitude):
    """dx, dy (B,H,W) from the station to every cell centre, and the 3D distance to the drone plane."""
    B, H, W = heights.shape
    dev = heights.device
    xs = torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2     # x of column j = j - 63.5
    ys = torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2     # y of row i    = i - 63.5
    dx = xs.view(1, 1, W) - station[:, 0].view(B, 1, 1)
    dy = ys.view(1, H, 1) - station[:, 1].view(B, 1, 1)
    dz = (altitude - station[:, 2]).view(B, 1, 1)
    d3 = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2).clamp_min(1.0)
    return dx.expand(B, H, W), dy.expand(B, H, W), d3


def line_of_sight(heights, station, altitude, cfg=CFG):
    """1 where the segment from the station to the centre of the cell at the drone altitude stays above the surface.
    The surface height is read (bilinear) at los_samples points along each segment."""
    B, H, W = heights.shape
    dev = heights.device
    t = torch.linspace(0, 1, cfg["los_samples"] + 2, device=dev)[1:-1].view(1, -1, 1, 1)     # (1, K, 1, 1)
    xs = (torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2).view(1, 1, 1, W)
    ys = (torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2).view(1, 1, H, 1)
    sx, sy, sz = (station[:, k].view(B, 1, 1, 1) for k in range(3))
    px = sx + t * (xs - sx)                                                 # (B, K, 1, W)
    py = sy + t * (ys - sy)                                                 # (B, K, H, 1)
    pz = sz + t * (altitude.view(B, 1, 1, 1) - sz)                         # (B, K, 1, 1)
    grid = torch.stack(torch.broadcast_tensors(px / (W / 2), py / (H / 2)), dim=-1)    # (B, K, H, W, 2) in [-1, 1]
    K = t.shape[1]
    surface = F.grid_sample(heights[:, None], grid.view(B, K * H, W, 2), mode="bilinear", padding_mode="border",
                            align_corners=False).view(B, K, H, W)
    return (pz > surface).all(dim=1).float()


def make_inputs(heights, station, altitude, cfg=CFG):
    """Model input (B, C, H, W) and validity mask (B, H, W) for one batch."""
    B, H, W = heights.shape
    dx, dy, d3 = plane_geometry(heights, station, altitude)
    alt = altitude.view(B, 1, 1)
    valid = heights < alt - 0.5
    hs = cfg["height_scale"]
    channels = dict(
        height=lambda: heights / hs,
        rel_height=lambda: (alt - heights) / hs,
        valid=lambda: valid.float(),
        tx=lambda: torch.exp(-(dx ** 2 + dy ** 2) / (2 * cfg["tx_sigma"] ** 2)),
        fspl=lambda: db_to_unit(free_space_gain_db(d3), cfg),
        tx_z=lambda: (station[:, 2] / hs).view(B, 1, 1).expand(B, H, W),
        altitude=lambda: (alt / hs).expand(B, H, W),
        los=lambda: line_of_sight(heights, station, altitude, cfg),
    )
    x = torch.stack([channels[c]() for c in cfg["in_channels"]], dim=1)
    return x, valid


def augment_d4(*tensors):
    """Apply the same random rotation (k * 90 deg) and optional mirror to every sample of each (B, ..., H, W) tensor."""
    B = tensors[0].shape[0]
    ks = torch.randint(0, 4, (B,)).tolist()
    flips = torch.randint(0, 2, (B,)).tolist()
    out = [t.clone() for t in tensors]
    for b in range(B):
        for o, t in zip(out, tensors):
            v = t[b].flip(-1) if flips[b] else t[b]
            o[b] = torch.rot90(v, ks[b], dims=(-2, -1))
    return out


def prepare_batch(heights, station, altitude, target_db, augment=False, cfg=CFG):
    x, valid = make_inputs(heights, station, altitude, cfg)
    y = db_to_unit(target_db.clamp_min(cfg["floor_db"]), cfg)
    if augment:
        x, y, valid = augment_d4(x, y, valid)
    return x, y, valid


def masked_mean(values, m):
    """Mean of `values` over the cells where the mask m is 1."""
    return (values * m).sum() / m.sum().clamp_min(1.0)


SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]]) / 8.0


def sobel(img):
    """Derivatives of a (B, H, W) map along x and y (Sobel filter, borders replicated)."""
    k = torch.stack([SOBEL_X, SOBEL_X.T])[:, None].to(img)                    # (2, 1, 3, 3)
    g = F.conv2d(F.pad(img[:, None], (1, 1, 1, 1), mode="replicate"), k)      # (B, 2, H, W)
    return g[:, 0], g[:, 1]


def gaussian_window(size=11, sigma=1.5, device=None):
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, size=11, sigma=1.5, data_range=1.0):
    """Local SSIM of every cell of two (B, H, W) maps."""
    w = gaussian_window(size, sigma, x.device)
    pad = size // 2
    blur = lambda t: F.conv2d(F.pad(t[:, None], (pad, pad, pad, pad), mode="replicate"), w)[:, 0]
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))


def masked_loss(pred, y, valid, cfg=CFG, weight=None):
    """Training loss over the valid cells. pred and y: (B, H, W) path gain on the [0, 1] scale.
    weight: optional (B, H, W) weight of every cell for the pixel terms (L1, MSE, Huber)."""
    m = valid.float()
    pixel_mean = lambda values: masked_mean(values if weight is None else values * weight, m)
    kind = cfg["loss"]
    if kind == "mse":
        return pixel_mean((pred - y) ** 2)
    if kind == "l1":
        return pixel_mean((pred - y).abs())
    if kind == "huber":
        return pixel_mean(F.huber_loss(pred, y, reduction="none", delta=cfg["huber_delta"]))
    if kind == "combo":
        w = cfg["loss_weights"]
        total = pred.new_zeros(())
        if w.get("l1", 0):
            total = total + w["l1"] * pixel_mean((pred - y).abs())
        if w.get("mse", 0):
            total = total + w["mse"] * pixel_mean((pred - y) ** 2)
        if w.get("ssim", 0):
            filled = torch.where(valid, pred, y)
            total = total + w["ssim"] * masked_mean(1.0 - ssim_map(filled, y), m)
        if w.get("grad", 0):
            inner = -F.max_pool2d(-m[:, None], 3, 1, 1)[:, 0]
            gx_p, gy_p = sobel(pred)
            gx_t, gy_t = sobel(y)
            total = total + w["grad"] * masked_mean((gx_p - gx_t).abs() + (gy_p - gy_t).abs(), inner)
        return total
    raise ValueError(f"unknown loss {kind!r}")


def no_signal(y, cfg=CFG):
    """Cells whose target is at the floor (-150 dB): no ray reached them."""
    return y <= db_to_unit(cfg["floor_db"] + 1e-3, cfg)


def combine(*weights):
    """Product of the per-cell weights that are not None (None when there is none)."""
    weights = [w for w in weights if w is not None]
    out = None
    for w in weights:
        out = w if out is None else out * w
    return out


def training_loss(out, y, valid, cfg=CFG):
    """Loss of one batch with the selected treatment of no-signal cells and the optional edge weighting (5.2).
    y: target on the [0, 1] scale."""
    method = cfg.get("floor_method", "none")
    tau = db_to_unit(cfg["floor_threshold_db"], cfg)                  # -130 dB on the [0, 1] scale
    edge_w = edge_weight_map(y, valid, cfg)                           # None unless edge_weight != 1
    if method == "none":
        return masked_loss(out, y, valid, cfg, edge_w)
    if method == "truncate":                                          # every target below tau becomes tau
        return masked_loss(out, y.clamp_min(tau), valid, cfg, edge_w)
    if method == "censor":                                            # no-signal cell: target "any value <= tau"
        target = torch.where(no_signal(y, cfg), torch.clamp(out.detach(), max=tau), y)
        return masked_loss(out, target, valid, cfg, edge_w)
    if method == "two_output":                                        # out[:, 0]: path gain, out[:, 1]: signal logit
        gain, logit = out[:, 0], out[:, 1]
        silent = no_signal(y, cfg)
        gain_loss = masked_loss(gain, torch.where(silent, gain.detach(), y), valid, cfg, edge_w)   # signal cells only
        signal_loss = masked_mean(F.binary_cross_entropy_with_logits(logit, (~silent).float(), reduction="none"),
                                  valid.float())
        return gain_loss + cfg["signal_loss_weight"] * signal_loss
    if method == "asymmetric":                                        # overestimating a weak signal costs more
        weak = y < db_to_unit(cfg["weak_db"], cfg)
        over = out.detach() > y
        weight = 1.0 + (cfg["asym_factor"] - 1.0) * (weak & over).float()
        return masked_loss(out, y, valid, cfg, combine(weight, edge_w))
    if method == "reweight":                                          # rare path gain values count more
        table = torch.as_tensor(cfg["reweight_table"], dtype=torch.float32, device=y.device)
        bins = (y * (cfg["max_db"] - cfg["floor_db"]) / cfg["reweight_bin_db"]).long().clamp(0, len(table) - 1)
        return masked_loss(out, y, valid, cfg, combine(table[bins], edge_w))
    raise ValueError(f"unknown floor_method {method!r}")


def near_edges(gain_db, valid, cfg=CFG):
    """Valid cells within edge_width cells of a sharp change (>= edge_threshold_db per cell) of a (B, H, W) path gain
    map in dB. The map is clipped at floor_threshold_db first, so the no-signal cells do not create edges."""
    gx, gy = sobel(gain_db.clamp_min(cfg["floor_threshold_db"]))
    inner = -F.max_pool2d(-valid.float()[:, None], 3, 1, 1)[:, 0] > 0           # valid cells with 8 valid neighbours
    edge = inner & (torch.sqrt(gx ** 2 + gy ** 2) >= cfg["edge_threshold_db"])
    w = cfg["edge_width"]
    return valid & (F.max_pool2d(edge.float()[:, None], 2 * w + 1, 1, w)[:, 0] > 0)


def edge_weight_map(y, valid, cfg=CFG):
    """Per-cell weight of the edge-weighted loss (None when edge_weight is 1). y: target on the [0, 1] scale."""
    if cfg.get("edge_weight", 1.0) == 1.0:
        return None
    near = near_edges(unit_to_db(y, cfg), valid, cfg)
    return 1.0 + (cfg["edge_weight"] - 1.0) * near.float()


def amp_autocast(cfg=CFG):
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=cfg["amp"] and DEVICE.type == "cuda")


def model_predictor(model, cfg=CFG):
    """predict(heights, station, altitude) -> path gain in dB (B, H, W)."""
    method = cfg.get("floor_method", "none")
    low = db_to_unit(cfg["floor_threshold_db"], cfg) if method == "truncate" else 0.0

    @torch.no_grad()
    def predict(heights, station, altitude):
        x, _ = make_inputs(heights, station, altitude, cfg)
        with amp_autocast(cfg):
            out = model(x).float()
        if method == "two_output":                                    # P(signal) < 0.5  ->  -150 dB
            gain = torch.where(out[:, 1] >= 0, out[:, 0].clamp(0.0, 1.0), torch.zeros_like(out[:, 0]))
        else:
            gain = out.clamp(low, 1.0)
        return unit_to_db(gain, cfg)
    return predict


def coverage_boundary(cov, valid, width):
    """Valid cells with both covered and uncovered valid cells within `width` cells (Chebyshev distance)."""
    k = 2 * width + 1
    hi = F.max_pool2d(torch.where(valid, cov, -1.0)[:, None], k, 1, width)[:, 0]      # max over valid neighbours
    lo = -F.max_pool2d(torch.where(valid, -cov, -2.0)[:, None], k, 1, width)[:, 0]    # min over valid neighbours
    return valid & (hi == 1) & (lo == 0)


@torch.no_grad()
def evaluate(predict, split, cfg=CFG, indices=None):
    """Per map error sums and metrics of predict() on the maps `indices` of a split (all maps by default)."""
    idx_all = np.arange(len(split)) if indices is None else np.asarray(indices)
    thr, w = cfg["coverage_thresholds_db"], cfg["boundary_width"]
    link = cfg.get("floor_threshold_db", -130.0)
    acc = {k: [] for k in ("n", "sse", "sae", "se", "n_sig", "sse_sig", "sse_link", "n_nosig", "nosig_link",
                           "nosig_low", "n_edge", "sse_edge")}
    for t in thr:
        acc.update({f"cov_wrong@{t:g}": [], f"bnd_n@{t:g}": [], f"bnd_sse@{t:g}": []})
    for s in range(0, len(idx_all), cfg["eval_batch_size"]):
        h, st, alt, tgt = split.batch(idx_all[s:s + cfg["eval_batch_size"]], DEVICE)
        pred = predict(h, st, alt)
        tgt = tgt.clamp_min(cfg["floor_db"])
        valid = h < alt.view(-1, 1, 1) - 0.5
        m = valid.float()
        err = (pred - tgt) * m
        sig = m * (tgt > cfg["floor_db"] + 1e-3).float()
        acc["n"].append(m.sum((1, 2)))
        acc["sse"].append((err ** 2).sum((1, 2)))
        acc["sae"].append(err.abs().sum((1, 2)))
        acc["se"].append(err.sum((1, 2)))
        acc["n_sig"].append(sig.sum((1, 2)))
        acc["sse_sig"].append((err ** 2 * sig).sum((1, 2)))
        err_link = (pred.clamp_min(link) - tgt.clamp_min(link)) * m
        nosig = m - sig
        acc["sse_link"].append((err_link ** 2).sum((1, 2)))
        acc["n_nosig"].append(nosig.sum((1, 2)))
        acc["nosig_link"].append((nosig * (pred >= thr[-1]).float()).sum((1, 2)))
        acc["nosig_low"].append((nosig * (pred <= link + 0.01).float()).sum((1, 2)))
        near = near_edges(tgt, valid, cfg).float()
        acc["n_edge"].append(near.sum((1, 2)))
        acc["sse_edge"].append((err_link ** 2 * near).sum((1, 2)))
        for t in thr:
            cov_t, cov_p = (tgt >= t).float(), (pred >= t).float()
            band = coverage_boundary(cov_t, valid, w).float()
            acc[f"cov_wrong@{t:g}"].append(((cov_t != cov_p).float() * m).sum((1, 2)))
            acc[f"bnd_n@{t:g}"].append(band.sum((1, 2)))
            acc[f"bnd_sse@{t:g}"].append((err ** 2 * band).sum((1, 2)))
    res = {k: torch.cat(v).double().cpu().numpy() for k, v in acc.items()}
    n = np.maximum(res["n"], 1)
    res.update(idx=idx_all, rmse_db=np.sqrt(res["sse"] / n), mae_db=res["sae"] / n,
               rmse_link_db=np.sqrt(res["sse_link"] / n),
               family=split.family[idx_all], altitude=split.altitude.numpy()[idx_all],
               source=split.source[idx_all], rt_time=split.rt_time[idx_all])
    return res


def summarize(res, cfg=CFG, mask=None):
    """Scalar metrics of an evaluate() result (optionally of the maps where mask is True)."""
    r = res if mask is None else {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                                  for k, v in res.items()}
    N = r["n"].sum()
    out = dict(maps=len(r["n"]), rmse_db=r["rmse_db"].mean(), rmse_db_pooled=np.sqrt(r["sse"].sum() / N),
               mae_db=r["sae"].sum() / N, bias_db=r["se"].sum() / N,
               rmse_signal_db=np.sqrt(r["sse_sig"].sum() / max(r["n_sig"].sum(), 1)))
    out["nrmse"] = out["rmse_db_pooled"] / (cfg["max_db"] - cfg["floor_db"])
    out["rmse_link_db"] = r["rmse_link_db"].mean()
    n_nosig = max(r["n_nosig"].sum(), 1)
    out["no_signal_false_link"] = r["nosig_link"].sum() / n_nosig
    out["no_signal_detected"] = r["nosig_low"].sum() / n_nosig
    n_edge = r["n_edge"].sum()
    out["edge_rmse_db"] = np.sqrt(r["sse_edge"].sum() / max(n_edge, 1))
    out["non_edge_rmse_db"] = np.sqrt((r["sse_link"].sum() - r["sse_edge"].sum()) / max(N - n_edge, 1))
    for t in cfg["coverage_thresholds_db"]:
        out[f"cov_err@{t:g}"] = r[f"cov_wrong@{t:g}"].sum() / N
        out[f"boundary_rmse@{t:g}"] = np.sqrt(r[f"bnd_sse@{t:g}"].sum() / max(r[f"bnd_n@{t:g}"].sum(), 1))
    return {k: float(v) for k, v in out.items()}


def group_rmse(res, cfg):
    """RMSE (dB, mean over maps) per city family and per altitude."""
    return dict(family={f: summarize(res, cfg, mask=res["family"] == f)["rmse_db"] for f in np.unique(res["family"])},
                altitude={f"{a:g} m": summarize(res, cfg, mask=res["altitude"] == a)["rmse_db"]
                          for a in np.unique(res["altitude"])})


def make_optimizer(model, cfg=CFG):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "rel_bias" in name else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg["weight_decay"]),
                              dict(params=no_decay, weight_decay=0.0)], lr=cfg["lr"])


def warmup_cosine(total_steps, cfg=CFG):
    lr, min_lr, warm = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"]

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return (min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t))) / lr
    return factor


def train(model, model_kwargs, train_set, val_set, cfg, run_dir, resume=True):
    bs = cfg["batch_size"]
    steps_per_epoch = len(train_set) // bs
    total = steps_per_epoch * cfg["epochs"]
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)
    start, best, history = 0, float("inf"), []
    last, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"], ck["best"], ck["history"]
        print(f"resumed from {last} after epoch {start} (best validation {cfg.get('select_metric', 'rmse_db')} "
              f"{best:.3f} dB)")
    (run_dir / "config.json").write_text(json.dumps(dict(cfg=cfg, model_kwargs=model_kwargs), indent=2, default=list))
    val_idx = None if cfg["max_val_maps"] is None else np.linspace(0, len(val_set) - 1, cfg["max_val_maps"]).astype(int)
    span = cfg["max_db"] - cfg["floor_db"]
    print(f"{len(train_set)} training maps, {steps_per_epoch} steps per epoch, {total} steps in total")

    for epoch in range(start, cfg["epochs"]):
        t0 = time.time()
        seed_everything(cfg["seed"] + epoch)                   # same order and augmentation when resuming
        perm = torch.randperm(len(train_set))
        model.train()
        loss_sum = torch.zeros((), device=DEVICE)
        steps = range(steps_per_epoch)
        if tqdm is not None:
            steps = tqdm(steps, desc=f"epoch {epoch + 1}/{cfg['epochs']}", leave=False)
        for b in steps:
            x, y, valid = prepare_batch(*train_set.batch(perm[b * bs:(b + 1) * bs], DEVICE),
                                        augment=cfg["augment"], cfg=cfg)
            with amp_autocast(cfg):
                out = model(x)
            loss = training_loss(out.float(), y, valid, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss is {loss.item()} at epoch {epoch + 1}, step {b}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach()
            if tqdm is not None and b % 50 == 0:
                steps.set_postfix(loss=f"{loss.item():.2e}", lr=f"{sched.get_last_lr()[0]:.1e}")
        train_loss = loss_sum.item() / steps_per_epoch

        model.eval()
        val = summarize(evaluate(model_predictor(model, cfg), val_set, cfg, val_idx), cfg)
        entry = dict(epoch=epoch + 1, train_loss=train_loss, lr=sched.get_last_lr()[0], time_s=time.time() - t0,
                     **{f"val_{k}": v for k, v in val.items() if k != "maps"})
        if cfg["loss"] == "mse":
            entry["train_rmse_db"] = math.sqrt(train_loss) * span   # on augmented training batches, in train mode
        history.append(entry)
        select = cfg.get("select_metric", "rmse_db")
        if val[select] < best:
            best = val[select]
            torch.save(dict(model=model.state_dict(), model_kwargs=model_kwargs, cfg=cfg, epoch=epoch + 1,
                            val=val), best_path)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch + 1, best=best, history=history,
                        model_kwargs=model_kwargs, cfg=cfg), last)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        t_cov = cfg["coverage_thresholds_db"][-1]
        cov = val[f"cov_err@{t_cov:g}"]
        print(f"epoch {epoch + 1:3d}/{cfg['epochs']} | train loss {train_loss:.3e} | val RMSE {val['rmse_db']:.3f} dB "
              f"(MAE {val['mae_db']:.3f}, cov err@{t_cov:g} {cov:.3%}) | lr {entry['lr']:.1e} | "
              f"{entry['time_s']:.0f} s" + ("  *best*" if val[select] == best else ""))
    return history


@torch.no_grad()
def prediction_time_ms(predict, split, batch_sizes=(1, 32), n_iter=30, warmup=5):
    """Milliseconds per map, from inputs on the GPU to the path gain map in dB."""
    out = {}
    for bs in batch_sizes:
        h, st, alt, _ = split.batch(np.arange(bs), DEVICE)
        for _ in range(warmup):
            predict(h, st, alt)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            predict(h, st, alt)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        out[f"batch {bs}"] = 1e3 * (time.perf_counter() - t0) / (n_iter * bs)
    return out