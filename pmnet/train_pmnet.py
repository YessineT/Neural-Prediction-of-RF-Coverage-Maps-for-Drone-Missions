"""
Trains and evaluates PMNet (https://github.com/abman23/PMNet, MIT license) on the AIRWISE coverage map dataset.

What it does
    - gets train.npz / val.npz / test.npz / ood.npz from Hugging Face (TYessine/airwise-rf-coverage-maps), downloaded once
      and cached on your disk (about 1 GB); or reads them from a local folder if you pass --data
    - builds the 5 input channels of every map itself (height, station, valid, altitude, station height), so this file
      does not need scene_images.py
    - builds PMNet from the pmnet_v1.py / pmnet_v3.py of the PMNet repository WITHOUT copying or editing them: it creates
      the model as in the PMNet repo (2 input channels), then replaces the two layers that depend on the number of input
      channels (the first convolution and the first convolution of the last decoder block)
    - target: path gain in dB scaled to 0..1 with (dB + 150) / 150, as PMNet does (-150 dB = no signal)
    - loss: mean squared error on VALID cells only (cells inside buildings taller than the drone are ignored)
    - reports errors in dB (RMSE per map, MAE), error per altitude and per family, coverage F1 / IoU at thresholds,
      and the inference time per map next to the ray tracing time of the dataset

Needs: pip install torch numpy huggingface_hub

Examples:
    python train_pmnet.py --pmnet-dir PMNet/network --smoke            # 1 minute check that everything runs
    python train_pmnet.py --pmnet-dir PMNet/network --overfit 8        # the loss must go close to 0
    python train_pmnet.py --pmnet-dir PMNet/network --epochs 30 --augment --out runs/v3
    python train_pmnet.py --pmnet-dir PMNet/network --eval-only runs/v3/best.pt
    python train_pmnet.py --pmnet-dir PMNet/network --epochs 30 --augment --out runs/v3 --resume runs/v3/last.pt
--pmnet-dir is the folder of your PMNet clone that contains pmnet_v3.py (called network/ or models/).
Add --data <folder> to read the .npz files from a local folder instead of Hugging Face.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


FLOOR_DB, RANGE_DB = -150.0, 150.0
HF_REPO = "TYessine/airwise-rf-coverage-maps"
HALF, CELL, N = 64.0, 1.0, 128           # scene of 128 m x 128 m, cells of 1 m, centred on (0, 0)
H_NORM = 64.0                            # m, scale of the heights
ALT_NORM = 60.0                          # m, highest drone altitude
DIST_NORM = 200.0                        # m, scale of the (optional) distance channel
STATION_SIGMA = 1.5                      # cells, width of the station marker
BASE_CHANNELS = ["height", "station", "valid", "altitude", "station_z"]
_xc = -HALF + CELL * (np.arange(N) + 0.5)            # cell centres; array index is [row = y, col = x]
_X, _Y = np.meshgrid(_xc, _xc)


def make_input(height, station, altitude, distance=False):
    """One map as (C, 128, 128) float32: height / 64, Gaussian marker at the station, valid mask (drone plane above
    the surface), altitude / 60 and station z / 64 as constant images, optionally the 3D distance to the station."""
    sx, sy, sz = (float(v) for v in station)
    d2 = (_X - sx) ** 2 + (_Y - sy) ** 2
    chans = [height / H_NORM,
             np.exp(-d2 / (2 * STATION_SIGMA ** 2)),
             (height < altitude - 0.5).astype(np.float32),
             np.full((N, N), altitude / ALT_NORM, np.float32),
             np.full((N, N), sz / H_NORM, np.float32)]
    if distance:
        chans.append(np.sqrt(d2 + (altitude - sz) ** 2) / DIST_NORM)
    return np.stack(chans).astype(np.float32)


def get_split_file(split, src):
    """Path of <split>.npz: from the local folder --data if given (None if the file is missing there), otherwise
    downloaded from Hugging Face on the first call and cached by huggingface_hub afterwards."""
    if src["data"]:
        f = Path(src["data"]) / f"{split}.npz"
        return f if f.exists() else None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    print(f"getting {split}.npz from Hugging Face ({src['repo']}); downloaded once, then cached")
    return Path(hf_hub_download(src["repo"], f"{split}.npz", repo_type="dataset", cache_dir=src["cache"] or None))


class CoverageMaps:
    """The maps of one split as (input image, target in dB, valid mask). Images are built when an item is read.
    Works with torch DataLoader and Subset (items are dicts of numpy arrays)."""

    def __init__(self, split, src, distance=False):
        f = get_split_file(split, src)
        if f is None:
            raise FileNotFoundError(f"{split}.npz not found in {src['data']}")
        with np.load(f) as z:
            self.z = {k: z[k] for k in z.files}
        self.distance = distance
        self.channels = BASE_CHANNELS + (["distance"] if distance else [])

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


def db_to_unit(db):
    return ((db - FLOOR_DB) / RANGE_DB).clamp(0, 1)


def unit_to_db(u):
    return u * RANGE_DB + FLOOR_DB


# ---------------------------------------------------------------- model
def load_pmnet_module(pmnet_dir, variant):
    path = Path(pmnet_dir) / f"pmnet_{variant}.py"
    if not path.exists():
        sys.exit(f"{path} not found: --pmnet-dir must be the folder of your PMNet clone that contains pmnet_{variant}.py")
    spec = importlib.util.spec_from_file_location(f"pmnet_{variant}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(pmnet_dir, variant, in_channels, bn_momentum):
    mod = load_pmnet_module(pmnet_dir, variant)
    model = mod.PMNet(n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4],
                      output_stride=8 if variant == "v3" else 16)       # exactly as train.py of PMNet
    # PMNet has 2 input channels (map, transmitter). Replace the two layers that depend on that number:
    #   - first convolution of the encoder
    #   - first convolution of the last decoder block, which also receives the raw input image
    # (the max pooling of the stem has kernel 2 and is not touched)
    old = model.layer1.conv1.conv
    model.layer1.conv1.conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv_up00[0] = nn.Conv2d(128 + in_channels, 64, kernel_size=3, padding=1)
    # PMNet uses a batch norm momentum of 0.001 (running statistics move very slowly); with short runs the statistics
    # used at evaluation would lag behind, so the default here is the usual 0.1
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = bn_momentum
    return model


# ---------------------------------------------------------------- data
def to_device(batch, device):
    x = batch["x"].to(device, non_blocking=True)
    y = db_to_unit(batch["y"].to(device, non_blocking=True))
    v = batch["valid"].to(device, non_blocking=True)
    return x, y, v


def augment(x, y, v):
    """Same random rotation (multiple of 90 degrees) and flip for the whole batch. The scene, the station and the
    target are turned together, so the physics stays consistent (isotropic antennas)."""
    k = int(np.random.randint(4))
    flip = np.random.rand() < 0.5
    x, y, v = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2)), torch.rot90(v, k, (1, 2))
    if flip:
        x, y, v = x.flip(3), y.flip(2), v.flip(2)
    return x.contiguous(), y.contiguous(), v.contiguous()


def masked_mse(pred, target, valid):
    v = valid.float()
    return (((pred - target) ** 2) * v).sum() / v.sum().clamp(min=1)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, thresholds, amp=False):
    """Per-map results over a loader: errors in dB over the valid cells, coverage counts for every threshold."""
    model.eval()
    rec = dict(se=[], ae=[], n=[], alt=[], fam=[], tp=[], fp=[], fn=[])
    thr = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1)
    for batch in loader:
        x, y, v = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            pred = model(x)
        pred = unit_to_db(pred.float().squeeze(1).clamp(0, 1))
        true = unit_to_db(y)
        vf = v.float()
        err = (pred - true) * vf
        rec["se"].append((err ** 2).sum((1, 2)).cpu())
        rec["ae"].append(err.abs().sum((1, 2)).cpu())
        rec["n"].append(vf.sum((1, 2)).cpu())
        rec["alt"].append(batch["altitude"].float().cpu())
        rec["fam"] += [str(n).split("_")[0] for n in batch["name"]]
        pc = (pred.unsqueeze(0) >= thr) & v.unsqueeze(0)          # (T, B, H, W)
        tc = (true.unsqueeze(0) >= thr) & v.unsqueeze(0)
        rec["tp"].append((pc & tc).sum((2, 3)).T.cpu())
        rec["fp"].append((pc & ~tc).sum((2, 3)).T.cpu())
        rec["fn"].append((~pc & tc).sum((2, 3)).T.cpu())
    out = {k: (np.concatenate([t.numpy() for t in v]) if k != "fam" else np.array(v)) for k, v in rec.items()}
    return out


def summarize(rec, thresholds):
    n = np.maximum(rec["n"], 1)
    rmse_map = np.sqrt(rec["se"] / n)
    s = dict(maps=int(len(n)), rmse_db=float(rmse_map.mean()), rmse_pooled_db=float(np.sqrt(rec["se"].sum() / n.sum())),
             mae_db=float(rec["ae"].sum() / n.sum()), rmse_median_db=float(np.median(rmse_map)),
             rmse_p90_db=float(np.percentile(rmse_map, 90)))
    for i, t in enumerate(thresholds):
        tp, fp, fn = rec["tp"][:, i].sum(), rec["fp"][:, i].sum(), rec["fn"][:, i].sum()
        s[f"f1@{t:g}"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        s[f"iou@{t:g}"] = float(tp / max(tp + fp + fn, 1))
    s["by_altitude"] = {int(a): float(np.sqrt(rec["se"][rec["alt"] == a].sum() / max(rec["n"][rec["alt"] == a].sum(), 1)))
                        for a in np.unique(rec["alt"])}
    s["by_family"] = {f: float(np.sqrt(rec["se"][rec["fam"] == f].sum() / max(rec["n"][rec["fam"] == f].sum(), 1)))
                      for f in np.unique(rec["fam"])}
    return s


def print_summary(name, s, thresholds):
    cov = "  ".join(f"F1@{t:g}: {s[f'f1@{t:g}']:.3f}" for t in thresholds)
    print(f"  {name:6s} {s['maps']:5d} maps  RMSE {s['rmse_db']:.2f} dB (per map, mean)  median {s['rmse_median_db']:.2f}  "
          f"p90 {s['rmse_p90_db']:.2f}  MAE {s['mae_db']:.2f}  | {cov}")
    print("         by altitude: " + ", ".join(f"{a} m: {v:.2f}" for a, v in s["by_altitude"].items())
          + "   by family: " + ", ".join(f"{f}: {v:.2f}" for f, v in s["by_family"].items()))


@torch.no_grad()
def time_inference(model, ds, device, batch_sizes=(1, 16), n_iter=30):
    """Milliseconds per map for a few batch sizes (after a warm-up, with GPU synchronisation)."""
    model.eval()
    res = {}
    for bs in batch_sizes:
        x = torch.stack([torch.from_numpy(ds[i % len(ds)]["x"]) for i in range(bs)]).to(device)
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        res[bs] = (time.perf_counter() - t0) / (n_iter * bs) * 1000
    return res


def evaluate_splits(model, args, device, src, thresholds, splits):
    results = {}
    for split in splits:
        try:
            ds = CoverageMaps(split, src, args.distance)
        except FileNotFoundError:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        s = summarize(evaluate(model, loader, device, thresholds, args.amp), thresholds)
        s["rt_time_median_ms"] = float(np.median(ds.z["rt_time"]) * 1000)
        results[split] = s
        print_summary(split, s, thresholds)
        print(f"         ray tracing time in the dataset (median): {s['rt_time_median_ms']:.0f} ms per map")
    return results


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="", help="local folder with train.npz, val.npz, test.npz, ood.npz "
                                               "(default: download them from Hugging Face)")
    ap.add_argument("--hf-repo", default=HF_REPO, help="Hugging Face dataset repository")
    ap.add_argument("--cache-dir", default="", help="where Hugging Face files are cached (default: the usual cache)")
    ap.add_argument("--pmnet-dir", required=True, help="folder of the PMNet clone that contains pmnet_v3.py")
    ap.add_argument("--variant", choices=["v1", "v3"], default="v3", help="v3: output stride 8, v1: output stride 16")
    ap.add_argument("--out", default="runs/pmnet")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=10)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--bn-momentum", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true", help="random rotations by 90 degrees and flips")
    ap.add_argument("--distance", action="store_true", help="add the distance channel to the input")
    ap.add_argument("--amp", action="store_true", help="bfloat16 mixed precision (less memory, faster)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[-110.0, -100.0, -90.0],
                    help="path gain thresholds (dB) of the coverage metrics")
    ap.add_argument("--max-train-maps", type=int, default=0, help="use only this many training maps (0 = all)")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on the same N maps (sanity check)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 64 maps, 1 epoch")
    ap.add_argument("--eval-only", default="", help="checkpoint to evaluate (no training)")
    ap.add_argument("--resume", default="", help="continue a stopped run from its last.pt (same --out and options)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found, PMNet will be very slow on the CPU")
    torch.backends.cudnn.benchmark = True
    if args.smoke:
        args.max_train_maps, args.epochs = 64, 1

    src = dict(data=args.data, repo=args.hf_repo, cache=args.cache_dir)
    train_ds = CoverageMaps("train", src, args.distance) if not args.eval_only else None
    probe = train_ds or CoverageMaps("val", src, args.distance)
    in_ch = len(probe.channels)
    model = build_model(args.pmnet_dir, args.variant, in_ch, args.bn_momentum).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"PMNet {args.variant}, {in_ch} input channels {probe.channels}, {n_par / 1e6:.1f} M parameters, device {device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        rng = np.random.default_rng(args.seed)
        if args.overfit:
            idx = np.arange(args.overfit)
        elif args.max_train_maps:
            idx = rng.permutation(len(train_ds))[:args.max_train_maps]
        else:
            idx = None
        tr = Subset(train_ds, idx) if idx is not None else train_ds
        val_ds = Subset(train_ds, idx) if args.overfit else CoverageMaps("val", src, args.distance)
        if args.max_train_maps and not args.overfit and len(val_ds) > 256:
            val_ds = Subset(val_ds, rng.permutation(len(val_ds))[:256])
        train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=len(tr) >= args.batch_size,
                                  num_workers=args.workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
        print(f"training on {len(tr)} maps, validating on {len(val_ds)} maps, {len(train_loader)} steps per epoch")

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
        start_epoch, best = 1, 1e9
        if args.resume:
            ck = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
            else:
                print("WARNING: this checkpoint has no optimizer state, the optimizer restarts from scratch")
            start_epoch = ck["epoch"] + 1
            best = ck.get("best", ck.get("val_rmse_db", 1e9))
            print(f"resuming from {args.resume}: epoch {ck['epoch']} done, best validation RMSE so far {best:.3f} dB, "
                  f"learning rate {opt.param_groups[0]['lr']:.2e}")
            if start_epoch > args.epochs:
                print(f"nothing to do: {ck['epoch']} epochs are already done (--epochs {args.epochs}); "
                      f"use a larger --epochs to train further")
        new_log = not (args.resume and (out / "log.csv").exists())
        log = open(out / "log.csv", "w" if new_log else "a", newline="")
        w = csv.writer(log)
        if new_log:
            w.writerow(["epoch", "train_loss", "train_rmse_db_approx", "val_rmse_db", "val_mae_db", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            model.train()
            tot, cnt = 0.0, 0
            for batch in train_loader:
                x, y, v = to_device(batch, device)
                if args.augment:
                    x, y, v = augment(x, y, v)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                    pred = model(x)
                loss = masked_mse(pred.float().squeeze(1), y, v)
                loss.backward()
                opt.step()
                tot += loss.item()
                cnt += 1
            sched.step()
            s = summarize(evaluate(model, val_loader, device, args.thresholds, args.amp), args.thresholds)
            train_loss = tot / max(cnt, 1)
            w.writerow([epoch, f"{train_loss:.6f}", f"{np.sqrt(train_loss) * RANGE_DB:.2f}", f"{s['rmse_db']:.3f}",
                        f"{s['mae_db']:.3f}", f"{opt.param_groups[0]['lr']:.2e}", f"{time.time() - t0:.0f}"])
            log.flush()
            print(f"epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.6f} (~{np.sqrt(train_loss) * RANGE_DB:.2f} dB)  "
                  f"val RMSE {s['rmse_db']:.2f} dB  MAE {s['mae_db']:.2f} dB  {time.time() - t0:.0f} s")
            improved = s["rmse_db"] < best
            best = min(best, s["rmse_db"])
            ckpt = dict(model=model.state_dict(), args=vars(args), epoch=epoch, val_rmse_db=s["rmse_db"], channels=probe.channels)
            if improved:
                torch.save(ckpt, out / "best.pt")
            # last.pt also holds what is needed to continue the run (optimizer, schedule, best value)
            torch.save(dict(ckpt, opt=opt.state_dict(), sched=sched.state_dict(), best=best), out / "last.pt")
        log.close()
        ckpt_path = out / "best.pt"
    else:
        ckpt_path = Path(args.eval_only)

    if not Path(ckpt_path).exists():
        sys.exit(f"{ckpt_path} not found")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"\nevaluating {ckpt_path} (epoch {ck.get('epoch', '?')}), errors in dB over valid cells")
    if args.overfit and not args.eval_only:
        # sanity check: the error on the SAME maps the network was trained on (it must become small)
        s = summarize(evaluate(model, val_loader, device, args.thresholds, args.amp), args.thresholds)
        print_summary("overfit", s, args.thresholds)
        results = {"overfit_maps": s}
    else:
        results = evaluate_splits(model, args, device, src, args.thresholds, ["val", "test", "ood"])
    t = time_inference(model, probe, device)
    print("inference time per map: " + ", ".join(f"batch {b}: {ms:.1f} ms" for b, ms in t.items()))
    (out / "results.json").write_text(json.dumps(dict(results=results, inference_ms_per_map={str(b): v for b, v in t.items()},
                                                      checkpoint=str(ckpt_path)), indent=1))
    print(f"results written to {out / 'results.json'}")


if __name__ == "__main__":
    main()