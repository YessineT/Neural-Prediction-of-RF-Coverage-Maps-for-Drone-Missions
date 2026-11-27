"""
PMNet trained and measured EXACTLY like Yessine's RMTransformer study (his code is in yessine_port.py, copied
verbatim from his notebook): same data class, input channels, line-of-sight channel, augmentation, losses,
no-signal treatments, optimiser and schedule, training loop and metrics. Only the network is different (PMNet).
The results.json has the same layout as his, so his comparison tables work on it.

Needs in the same folder: yessine_port.py, train_pmnet.py, pmnet_slim.py.

Examples:
    python train_yessine_setup.py --loss mse --out runs/pm_mse_nolos_s0                      # baseline, 7 channels
    python train_yessine_setup.py --loss combo --los --out runs/pm_combo_los_s0              # best of his study
    python train_yessine_setup.py --loss combo --los --floor-method asymmetric --out runs/pm_combo_los_asym_s0
    python train_yessine_setup.py --loss huber --los --seed 1 --out runs/pm_huber_los_s1
    python train_yessine_setup.py --eval-only runs/pm_combo_los_s0 --los --loss combo
Smaller PMNet: add --width 0.5 (and/or --n-blocks 3 3 9 3). Interrupted run: start the same command again, it resumes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_pmnet as T          # PMNet builder and split downloader of your own code
import yessine_port as Y


class Squeezed(nn.Module):
    """PMNet returns (B, 1, H, W); Yessine's code expects (B, H, W)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)


def to_json(obj):
    return obj.item() if hasattr(obj, "item") else list(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="", help="local folder with the .npz files (default: Hugging Face, needs a token)")
    ap.add_argument("--hf-repo", default=T.HF_REPO)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--pmnet-dir", default="", help="empty = pmnet_slim.py")
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--variant", choices=["v1", "v3"], default="v3")
    ap.add_argument("--n-blocks", type=int, nargs=4, default=[3, 3, 27, 3])
    ap.add_argument("--bn-momentum", type=float, default=0.1)
    ap.add_argument("--out", default="runs/pmnet_yessine_setup")
    ap.add_argument("--loss", choices=["mse", "l1", "huber", "combo"], default="combo")
    ap.add_argument("--combo-weights", type=float, nargs=4, default=[0.4, 0.1, 0.2, 0.3],
                    metavar=("L1", "MSE", "SSIM", "GRAD"))
    ap.add_argument("--floor-method", choices=["none", "truncate", "censor", "asymmetric"], default="none",
                    help="treatment of the no-signal cells (his study 2); the checkpoint is then chosen on rmse_link_db")
    ap.add_argument("--los", action="store_true", help="add the line-of-sight input channel (his study 3)")
    ap.add_argument("--edge-weight", type=float, default=1.0, help="his edge-weighted loss uses 3.0 (1 = off)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-maps", type=int, default=0, help="testing only")
    ap.add_argument("--max-val-maps", type=int, default=None, help="validate on an evenly spaced subset (faster)")
    ap.add_argument("--eval-only", default="", help="run folder with best.pt to evaluate (no training)")
    args = ap.parse_args()

    cfg = {**Y.CFG, "model_size": "pmnet", "loss": args.loss, "floor_method": args.floor_method, "seed": args.seed,
           "epochs": args.epochs, "batch_size": args.batch_size, "edge_weight": args.edge_weight,
           "max_val_maps": args.max_val_maps,
           "loss_weights": dict(zip(["l1", "mse", "ssim", "grad"], args.combo_weights)),
           "in_channels": list(Y.CFG["in_channels"]) + (["los"] if args.los else []),
           "select_metric": "rmse_db" if args.floor_method == "none" else "rmse_link_db"}
    src = dict(data=args.data, repo=args.hf_repo, cache=args.cache_dir)
    sets = {}
    for split in (["train"] if not args.eval_only else []) + ["val", "test", "ood"]:
        sets[split] = Y.CoverageSplit(T.get_split_file(split, src), split)
        print(sets[split])
    if args.max_train_maps:                                        # testing only: the first maps of the split
        tr, n = sets["train"], args.max_train_maps
        tr.station, tr.altitude, tr.target, tr.scene_idx = tr.station[:n], tr.altitude[:n], tr.target[:n], tr.scene_idx[:n]
        tr.family, tr.source, tr.rt_time = tr.family[:n], tr.source[:n], tr.rt_time[:n]

    model_kwargs = dict(in_channels=len(cfg["in_channels"]), variant=args.variant, n_blocks=args.n_blocks,
                        width=args.width)
    Y.seed_everything(cfg["seed"])                                 # same initial weights for every run with this seed
    net = T.build_model(args.pmnet_dir, args.variant, len(cfg["in_channels"]), args.bn_momentum, args.n_blocks,
                        args.width)
    model = Squeezed(net).to(Y.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    name = Path(args.eval_only or args.out).name
    print(f"=== {name}: PMNet {args.variant}, {n_params / 1e6:.2f} M parameters, channels {cfg['in_channels']}, "
          f"loss {args.loss}, no-signal cells: {args.floor_method}, seed {args.seed}")

    run_dir = Path(args.eval_only or args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        hf = run_dir / "history.json"
        history = json.loads(hf.read_text()) if hf.exists() else []
    else:
        history = Y.train(model, model_kwargs, sets["train"], sets["val"], cfg, run_dir)

    ck = torch.load(run_dir / "best.pt", map_location=Y.DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    predict = Y.model_predictor(model, cfg)
    summary, groups = {}, {}
    for split in ("val", "test", "ood"):
        res = Y.evaluate(predict, sets[split], cfg)
        summary[split] = Y.summarize(res, cfg)
        groups[split] = Y.group_rmse(res, cfg)
        s = summary[split]
        print(f"  {split:5s} RMSE {s['rmse_db']:.3f} dB  MAE {s['mae_db']:.3f}  link RMSE {s['rmse_link_db']:.3f}  "
              f"edge RMSE {s['edge_rmse_db']:.3f}  cov err@-110 {100 * s['cov_err@-110']:.2f}%  "
              f"false links {100 * s['no_signal_false_link']:.2f}%")
        print("        by altitude: " + ", ".join(f"{k}: {v:.2f}" for k, v in groups[split]["altitude"].items())
              + "   by family: " + ", ".join(f"{k}: {v:.2f}" for k, v in groups[split]["family"].items()))
    result = dict(name=name, model_size="pmnet", loss=args.loss, floor_method=args.floor_method, parameters=n_params,
                  best_epoch=ck["epoch"],
                  train_time_min=sum(h["time_s"] for h in history) / 60 if history else None,
                  epoch_time_s=float(np.median([h["time_s"] for h in history])) if history else None,
                  ms_per_map=Y.prediction_time_ms(predict, sets["test"]), summary=summary, groups=groups,
                  history=history, cfg=cfg, model_kwargs=model_kwargs)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, default=to_json))
    print(f"results written to {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()