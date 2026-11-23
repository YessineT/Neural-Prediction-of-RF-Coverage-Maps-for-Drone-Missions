"""
AIRWISE · U-Net loss experiments (Mahdi) · to run on a PC with an NVIDIA GPU

Same U-Net and same 4 inputs as unet_v3 (height, distance, altitude, line of sight). Only the loss changes.
Trains 5 experiments one after the other (~15-40 min each depending on the GPU), then grades them all on test and OOD,
together with unet_v3 if its folder is in runs/.

    python run_loss_experiments.py                 # downloads the data from Hugging Face
    python run_loss_experiments.py D:/data_folder  # or: a folder that already has train/val/test/ood.npz

Everything is written next to this script, in runs/:
    runs/<experiment>/   best.pt, last.pt, history.json, config.json, done.json
    runs/loss_comparison/  test_summary.csv, ood_summary.csv, ..., val_curves.png, test_errors.png, ood_errors.png
If it stops (crash, power cut), run the same command again: finished experiments are skipped,
the current one continues from its last epoch.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================================================================================================
# 1 · Settings
# ====================================================================================================
BASE = dict(                   # everything shared by all experiments = the v3 settings
    epochs=60, batch_size=16, lr=1e-3, weight_decay=1e-4, lr_factor=0.3, lr_patience=3, early_stop_patience=10,
    value_loss="mse", edge_focus=0, edge_loss=True, edge_weight=0.5, ssim_weight=0,
    los_input=True, base_channels=32, augment=True,
)
EXPERIMENTS = {                # only what is different from v3
    "loss_nograd": dict(edge_loss=False),
    "loss_l1":     dict(value_loss="l1"),
    "loss_huber":  dict(value_loss="huber"),
    "loss_focus":  dict(edge_focus=4),
    "loss_ssim":   dict(ssim_weight=0.5),
}
ALREADY_TRAINED = ["unet_v3"]                  # the reference, shown in the final table too (not retrained)
OTHER = dict(num_workers=0 if os.name == "nt" else 4, seed=0, quick_test=False,
             data_dir=sys.argv[1] if len(sys.argv) > 1 else None)   # optional: folder with train/val/test/ood.npz
CONFIG = {**BASE, **OTHER}
HF_REPO = "TYessine/airwise-rf-coverage-maps"


# ====================================================================================================
# 2 · Setup: GPU, Drive
# ====================================================================================================
import os, json, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")   # "runs" folder next to this script
LOS_CACHE = f"{RUNS}/los_cache"          # line-of-sight pictures are computed once and reused by every run
COMPARE = f"{RUNS}/loss_comparison"      # the final tables and pictures
os.makedirs(LOS_CACHE, exist_ok=True); os.makedirs(COMPARE, exist_ok=True)


# ====================================================================================================
# 3 · Line of sight (idea B)
# ====================================================================================================
# For each square at the drone altitude: walk along the straight line antenna → square in 192 small steps.
# If at some step a building (or the ground) is higher than the line, the square is hidden (0); otherwise it sees the antenna (1).
@torch.no_grad()
def line_of_sight(heights, scene_idx, pos, alt, steps=192, batch=64):
    ii, jj = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing="ij")
    X, Y = (jj - 63.5).float(), (ii - 63.5).float()
    Hall = torch.tensor(heights, device=device).float()
    out = np.zeros((len(alt), 128, 128), np.uint8)
    for s in range(0, len(alt), batch):
        e = min(s + batch, len(alt))
        H = Hall[torch.as_tensor(scene_idx[s:e], device=device).long()].reshape(e - s, -1)
        p = torch.tensor(pos[s:e], device=device).float(); a = torch.tensor(alt[s:e], device=device).float()
        sx, sy, sz = (p[:, c, None, None] for c in range(3))
        seen = torch.ones(e - s, 128, 128, dtype=torch.bool, device=device)
        for t in torch.linspace(0, 1, steps + 2, device=device)[1:-1]:
            px, py, pz = sx + t * (X - sx), sy + t * (Y - sy), sz + t * (a[:, None, None] - sz)
            ci = (py + 63.5).round().clamp(0, 127).long(); cj = (px + 63.5).round().clamp(0, 127).long()
            seen &= ~(H.gather(1, (ci * 128 + cj).reshape(e - s, -1)).reshape(e - s, 128, 128) > pz)
        out[s:e] = seen.cpu().numpy()
    return out

def los_for(name, d):
    f = f"{LOS_CACHE}/{name}_{len(d['altitude'])}.npy"
    if os.path.exists(f):
        return np.unpackbits(np.load(f), axis=-1)
    t0 = time.time()
    los = line_of_sight(d["heights"], d["scene_idx"], d["station_pos"], d["altitude"])
    np.save(f, np.packbits(los, axis=-1))
    print(f"line of sight for {name}: {len(los)} maps in {time.time() - t0:.0f} s (saved for next runs)")
    return los


# ====================================================================================================
# 4 · Data
# ====================================================================================================
def split_path(name):
    if CONFIG["data_dir"]:
        return os.path.join(CONFIG["data_dir"], f"{name}.npz")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, f"{name}.npz", repo_type="dataset")

class CoverageDataset(Dataset):
    """One example = 4 input pictures (height, distance to antenna, altitude, line of sight), the answer (scaled dB) and the mask."""
    def __init__(self, name, augment=False):
        d = np.load(split_path(name))
        self.heights, self.idx = d["heights"], d["scene_idx"]
        self.pos, self.alt, self.target = d["station_pos"], d["altitude"], d["target_db"]
        self.names = d["scene_names"]
        self.family = np.array([str(n).split("_")[0] for n in self.names])[self.idx]   # town type of each map
        self.los = los_for(name, d)
        self.augment = augment
        ii, jj = np.mgrid[0:128, 0:128]
        self.x, self.y = (jj - 63.5).astype(np.float32), (ii - 63.5).astype(np.float32)

    def __len__(self):
        return len(self.alt)

    def __getitem__(self, k):
        H = self.heights[self.idx[k]]; sx, sy, sz = self.pos[k]; alt = self.alt[k]
        dist = np.sqrt((self.x - sx) ** 2 + (self.y - sy) ** 2 + (alt - sz) ** 2)
        x = torch.tensor(np.stack([H / 60, dist / 200, np.full((128, 128), alt / 60), self.los[k]]), dtype=torch.float32)
        y = torch.tensor(((self.target[k] + 150) / 100)[None], dtype=torch.float32)
        m = torch.tensor((H < alt - 0.5)[None], dtype=torch.float32)
        if self.augment:                                   # turn and/or mirror inputs, answer and mask together
            r = int(torch.randint(0, 4, (1,)))
            x, y, m = (torch.rot90(t, r, dims=(1, 2)) for t in (x, y, m))
            if torch.rand(1) < 0.5:
                x, y, m = (torch.flip(t, dims=(2,)) for t in (x, y, m))
        return x, y, m

def loader(ds, shuffle):
    return DataLoader(ds, batch_size=CONFIG["batch_size"] if shuffle else 32, shuffle=shuffle,
                      num_workers=CONFIG["num_workers"], pin_memory=(device == "cuda"))

train_ds = CoverageDataset("train", augment=True)
val_ds = CoverageDataset("val")
test_ds, ood_ds = CoverageDataset("test"), CoverageDataset("ood")
train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)
print(f"train {len(train_ds)} · val {len(val_ds)} · test {len(test_ds)} · ood {len(ood_ds)} maps")


# ====================================================================================================
# 5 · Model (same U-Net, with 3 or 4 inputs)
# ====================================================================================================
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, c=32, n_in=3):
        super().__init__()
        self.n_in = n_in
        self.down1, self.down2, self.down3 = block(n_in, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.bottom = block(4 * c, 8 * c)
        self.shrink = nn.MaxPool2d(2)
        self.up3, self.mix3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2), block(8 * c, 4 * c)
        self.up2, self.mix2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.up1, self.mix1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x = x[:, :self.n_in]                              # a 3-input model simply ignores the line-of-sight picture
        a = self.down1(x); b = self.down2(self.shrink(a)); c = self.down3(self.shrink(b))
        z = self.bottom(self.shrink(c))
        z = self.mix3(torch.cat([self.up3(z), c], 1))
        z = self.mix2(torch.cat([self.up2(z), b], 1))
        z = self.mix1(torch.cat([self.up1(z), a], 1))
        return self.out(z)



# ====================================================================================================
# 6 · The loss: grade 1 (values) + grade 2 (jumps) + grade 3 (patterns)
# ====================================================================================================
def masked_mse(pred, y, m):
    return (((pred - y) ** 2) * m).sum() / m.sum()

SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8      # 3×3 window: how much the value changes
SOBEL = torch.stack([SOBEL, SOBEL.T])[:, None].to(device)                    # left→right and bottom→top
ONES = torch.ones(1, 1, 3, 3, device=device)

def jumps(img):
    return F.conv2d(img, SOBEL, padding=1)

def inner_mask(m):
    """squares whose 8 neighbours are all valid (a jump next to a building would be meaningless)"""
    return (F.conv2d(m, ONES, padding=1) > 8.5).float()

def gradient_loss(pred, y, m):
    mi = inner_mask(m)
    return ((jumps(pred) - jumps(y)).abs() * mi).sum() / (2 * mi.sum() + 1e-6)

def value_loss(pred, y, m):
    """grade 1: error on each square, averaged (cliff squares can count more)"""
    d = pred - y
    kind = CONFIG["value_loss"]
    if kind == "mse":     e = d ** 2
    elif kind == "l1":    e = d.abs()
    elif kind == "huber": e = F.smooth_l1_loss(pred, y, reduction="none", beta=0.05)   # squared below 5 dB, absolute above
    w = m * (1 + CONFIG["edge_focus"] * edge_mask(y, m)) if CONFIG["edge_focus"] else m
    return (e * w).sum() / w.sum()

def ssim_loss(pred, y, m):
    """grade 3: 1 − SSIM. SSIM compares the average, the contrast and the pattern in each 7×7 window (1 = identical)"""
    p = pred * m + y * (1 - m)                                  # buildings: copy the truth so they never count
    mu_p, mu_y = (F.avg_pool2d(t, 7, 1, 3, count_include_pad=False) for t in (p, y))
    var_p = F.avg_pool2d(p * p, 7, 1, 3, count_include_pad=False) - mu_p ** 2
    var_y = F.avg_pool2d(y * y, 7, 1, 3, count_include_pad=False) - mu_y ** 2
    cov = F.avg_pool2d(p * y, 7, 1, 3, count_include_pad=False) - mu_p * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_y + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_y ** 2 + c1) * (var_p + var_y + c2))
    w = (F.avg_pool2d(m, 7, 1, 3, count_include_pad=False) > 0.999).float()   # windows fully outside buildings
    return ((1 - ssim) * w).sum() / (w.sum() + 1e-6)

def loss_fn(pred, y, m):
    g1 = value_loss(pred, y, m)
    g2 = CONFIG["edge_weight"] * gradient_loss(pred, y, m) if CONFIG["edge_loss"] else torch.zeros((), device=pred.device)
    g3 = CONFIG["ssim_weight"] * ssim_loss(pred, y, m) if CONFIG["ssim_weight"] else torch.zeros((), device=pred.device)
    return g1 + g2 + g3, g1, g2, g3

def edge_mask(y, m):
    """the 'cliff' squares of the truth: signal changes by more than 5 dB per metre (light smoothing first, to ignore noise)"""
    ys = F.avg_pool2d(y, 3, stride=1, padding=1, count_include_pad=False)
    g = jumps(ys) * 100                                                # dB per metre
    return ((g ** 2).sum(1, keepdim=True).sqrt() > 5).float() * inner_mask(m)


# ====================================================================================================
# 7 · Train one experiment (with checkpoint + resume)
# ====================================================================================================
@torch.no_grad()
def val_rmse(model, dl, max_batches=None):
    model.eval(); se = n = 0.0
    for b, (x, y, m) in enumerate(dl):
        if max_batches and b >= max_batches: break
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            p = model(x)
        se += (((p.float() - y) ** 2) * m).sum().item(); n += m.sum().item()
    return (se / n) ** 0.5 * 100                                     # back to dB

def train(name):
    global CONFIG
    CONFIG = {**BASE, **EXPERIMENTS[name], **OTHER, "run_name": name}   # the loss functions read CONFIG
    out = f"{RUNS}/{name}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/done.json"):
        print(f"{name}: already finished, skipped"); return
    json.dump(CONFIG, open(f"{out}/config.json", "w"), indent=2)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model = UNet(CONFIG["base_channels"], 4 if CONFIG["los_input"] else 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=CONFIG["lr_factor"], patience=CONFIG["lr_patience"])
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    ckpt, best_path = f"{out}/last.pt", f"{out}/best.pt"
    start, best, history, waited = 1, float("inf"), [], 0
    if os.path.exists(ckpt):                                         # resume after a disconnection
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, history, waited = ck["epoch"] + 1, ck["best"], ck["history"], ck["waited"]
        print(f"{name}: resuming at epoch {start}")
    print(f"\n==================== {name}: {EXPERIMENTS[name]} ====================")
    quick = CONFIG["quick_test"]
    for epoch in range(start, CONFIG["epochs"] + 1):
        if waited >= CONFIG["early_stop_patience"]: break
        t0 = time.time(); model.train(); total = s1 = s2 = s3 = nb = 0
        for b, (x, y, m) in enumerate(train_loader):
            if quick and b >= 3: break
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                p = model(x)
            loss, g1, g2, g3 = loss_fn(p.float(), y, m)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            with torch.no_grad(): total += masked_mse(p.float(), y, m).item()   # plain MSE, to compare with v1/v2/v3
            s1 += g1.item(); s2 += g2.item(); s3 += g3.item(); nb += 1
        train_rmse = (total / nb) ** 0.5 * 100
        v = val_rmse(model, val_loader, 2 if quick else None)
        scheduler.step(v); lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_rmse=train_rmse, val_rmse=v, lr=lr, grade1=s1 / nb, grade2=s2 / nb, grade3=s3 / nb))
        msg = f"epoch {epoch:2d} | train {train_rmse:5.2f} dB | val {v:5.2f} dB | lr {lr:.0e} | grades {s1/nb:.4f} + {s2/nb:.4f} + {s3/nb:.4f} | {time.time()-t0:3.0f} s"
        if v < best:
            best, waited = v, 0; torch.save(model.state_dict(), best_path); msg += "  ← best"
        else:
            waited += 1
        print(msg, flush=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best=best, history=history, waited=waited), ckpt)
        json.dump(history, open(f"{out}/history.json", "w"), indent=1)
        if quick and epoch >= start + 1: break
    json.dump(dict(best_val_rmse=best, epochs=len(history)), open(f"{out}/done.json", "w"))
    print(f"{name}: finished, best validation RMSE {best:.2f} dB")


# ====================================================================================================
# 8 · Train all the experiments, one after the other
# ====================================================================================================
for name in EXPERIMENTS:
    train(name)


# ====================================================================================================
# 9 · Grade every network on the same test and OOD maps
# ====================================================================================================
@torch.no_grad()
def evaluate(net, ds, max_batches=None):
    S = {k: [] for k in ["se", "ae", "n", "ese", "eae", "en"]}                # per map
    for b, (x, y, m) in enumerate(loader(ds, False)):
        if max_batches and b >= max_batches: break
        x, y, m = x.to(device), y.to(device), m.to(device)
        d = (net(x).float() - y) * 100                                     # error in dB
        e = edge_mask(y, m)
        for key, val in [("se", (d ** 2) * m), ("ae", d.abs() * m), ("n", m), ("ese", (d ** 2) * e), ("eae", d.abs() * e), ("en", e)]:
            S[key].append(val.sum((1, 2, 3)).cpu())
    S = {k: torch.cat(v).numpy() for k, v in S.items()}
    k = len(S["n"]); fam, alt = ds.family[:k], ds.alt[:k]
    def row(sel):
        return dict(rmse=float(np.sqrt(S["se"][sel].sum() / S["n"][sel].sum())), mae=float(S["ae"][sel].sum() / S["n"][sel].sum()),
                    edge_rmse=float(np.sqrt(S["ese"][sel].sum() / S["en"][sel].sum())), edge_mae=float(S["eae"][sel].sum() / S["en"][sel].sum()))
    res = dict(ALL=row(np.ones(k, bool)))
    res.update({f: row(fam == f) for f in sorted(set(fam))})
    res.update({f"{a:.0f} m": row(alt == a) for a in sorted(set(alt))})
    return res

nets = {}
for run in ALREADY_TRAINED + list(EXPERIMENTS):
    path = f"{RUNS}/{run}/best.pt"
    if not os.path.exists(path):
        print("skip", run, "(not in Drive)"); continue
    cfg = json.load(open(f"{RUNS}/{run}/config.json"))
    net = UNet(cfg.get("base_channels", 32), 4 if cfg.get("los_input", False) else 3).to(device)
    net.load_state_dict(torch.load(path, map_location=device)); net.eval()
    nets[run] = net

quick = OTHER["quick_test"]
results = {split: {run: evaluate(net, ds, 2 if quick else None) for run, net in nets.items()}
           for split, ds in [("test", test_ds), ("ood", ood_ds)]}
json.dump(results, open(f"{COMPARE}/results.json", "w"), indent=2)


# ====================================================================================================
# 10 · THE RESULTS: one table with every network (dB, lower = better)
# ====================================================================================================
import pandas as pd
for split in ["test", "ood"]:
    r = results[split]
    summary = pd.DataFrame({run: {"MAE": v["ALL"]["mae"], "EDGE MAE": v["ALL"]["edge_mae"], "RMSE": v["ALL"]["rmse"],
                                  "EDGE RMSE": v["ALL"]["edge_rmse"]} for run, v in r.items()}).T.round(2)
    groups = pd.DataFrame({run: {g: v[g]["mae"] for g in v if g != "ALL"} for run, v in r.items()}).T.round(2)
    edges = pd.DataFrame({run: {g: v[g]["edge_mae"] for g in v if g != "ALL"} for run, v in r.items()}).T.round(2)
    summary.to_csv(f"{COMPARE}/{split}_summary.csv"); groups.to_csv(f"{COMPARE}/{split}_mae_per_group.csv")
    edges.to_csv(f"{COMPARE}/{split}_edge_mae_per_group.csv")
    print(f"\n==================== {split.upper()} ===================="); print("overall:"); print(summary.to_string())
    print("MAE per town type and altitude:"); print(groups.to_string())
    print("EDGE MAE per town type and altitude:"); print(edges.to_string())


# ====================================================================================================
# 11 · Validation curves of all the experiments
# ====================================================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for run in nets:
    hp = f"{RUNS}/{run}/history.json"
    if os.path.exists(hp):
        h = json.load(open(hp))
        ax.plot([r["epoch"] for r in h], [r["val_rmse"] for r in h], label=run)
ax.set_xlabel("epoch"); ax.set_ylabel("validation RMSE (dB)"); ax.set_ylim(top=12); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{COMPARE}/val_curves.png", dpi=120); plt.close()


# ====================================================================================================
# 12 · Pictures: Sionna and the error of each network on the same maps
# ====================================================================================================
def show(ds, ks, title):
    cols = 1 + len(nets)
    fig, ax = plt.subplots(len(ks), cols, figsize=(3.8 * cols, 3.7 * len(ks)), squeeze=False)
    for r, k in enumerate(ks):
        x, y, m = ds[k]
        ok = m[0].numpy() > 0
        t = np.where(ok, y[0].numpy() * 100 - 150, np.nan)
        panels = [(t, f"Sionna · {ds.names[ds.idx[k]]} · {ds.alt[k]:.0f} m", "viridis", -150, -50)]
        for run, net in nets.items():
            with torch.no_grad():
                q = np.where(ok, net(x[None].to(device))[0, 0].float().cpu().numpy() * 100 - 150, np.nan)
            panels.append((q - t, f"error {run}", "RdBu", -20, 20))
        for c, (img, ttl, cm, lo, hi) in enumerate(panels):
            ax[r, c].imshow(img, origin="lower", cmap=cm, vmin=lo, vmax=hi)
            ax[r, c].set_title(ttl, fontsize=9); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle(title + "   (red = too weak, blue = too strong, white = right)", fontsize=13); plt.tight_layout()
    plt.savefig(f"{COMPARE}/{title.split()[0].lower()}_errors.png", dpi=100); plt.close()

show(test_ds, [10, 700, 1500] if not quick else [0, 1], "TEST towns (known types)")
show(ood_ds, [10, 1500, 3000] if not quick else [0, 1], "OOD towns (unseen types)")
