#!/usr/bin/env python
"""Track J train.py: GEARS on Jurkat, official recipe, per-epoch checkpoints.

Official GEARS optimizer/recipe: Adam(lr=1e-3, weight_decay=5e-4),
StepLR(step_size=1, gamma=0.5), grad clip value 1.0, GEARS weighted loss with
direction_lambda=1e-3. Architecture params per spec: hidden 32, train batch 8,
eval batch 16, 8 epochs, seed 42.

--benchmark_batches N: run N training batches + 1 val pass, record timing/RAM,
project full runtime, and exit without training.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

from gears import GEARS
from gears.inference import compute_metrics, evaluate
from gears.pertdata import PertData
from gears.utils import loss_fct


def ram_gb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
    return (info["MemTotal"] - info["MemAvailable"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/workspace/jurkat/gears_data")
    ap.add_argument("--dataset", default="nadig25jurkat")
    ap.add_argument("--out_dir", default="/mnt/results/jurkat_gears")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--benchmark_batches", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pd_ = PertData(data_path=args.data_dir)
    pd_.load(data_path=os.path.join(args.data_dir, args.dataset))
    # Track J: use the biology-informed, response-blind split from prepare_data
    # (custom split pickle written from split_manifest.json), NOT GEARS's random
    # 'single' split.
    custom_split = os.path.join(args.data_dir, args.dataset, "splits", "custom_trackJ.pkl")
    pd_.prepare_split(split="custom", split_dict_path=custom_split)
    pd_.get_dataloader(batch_size=args.batch_size, test_batch_size=args.eval_batch_size)

    model = GEARS(pd_, device="cpu")
    model.model_initialize(hidden_size=args.hidden_size)
    n_params = sum(p.numel() for p in model.model.parameters())

    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_loader = model.dataloader["train_loader"]
    val_loader = model.dataloader["val_loader"]

    # ---- benchmark mode -------------------------------------------------------
    if args.benchmark_batches:
        model.model.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model.model(batch)
            loss = loss_fct(pred, batch.y, batch.pert, ctrl=model.ctrl_expression,
                            dict_filter=model.dict_filter,
                            direction_lambda=model.config["direction_lambda"])
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.model.parameters(), clip_value=1.0)
            optimizer.step()
            if step == 9:
                t0 = time.time()  # reset after warmup
            if step >= 9 + args.benchmark_batches:
                break
        per_batch = (time.time() - t0) / args.benchmark_batches
        tv = time.time()
        evaluate(val_loader, model.model, model.config["uncertainty"], "cpu")
        val_time = time.time() - tv
        n_train = len(train_loader.dataset)
        est = per_batch * n_train * args.epochs + val_time * args.epochs
        bench = {"n_params": n_params, "ram_used_gb": round(ram_gb(), 2),
                 "s_per_train_batch": round(per_batch, 3),
                 "train_batches_per_epoch": n_train,
                 "val_eval_s": round(val_time, 1),
                 "estimated_total_train_h": round(est / 3600, 2)}
        print("[benchmark] " + json.dumps(bench), flush=True)
        with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
            json.dump(bench, f, indent=2)
        return

    # ---- training loop (official recipe) --------------------------------------
    history = []
    best_val = np.inf
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.model.train()
        te = time.time()
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model.model(batch)
            loss = loss_fct(pred, batch.y, batch.pert, ctrl=model.ctrl_expression,
                            dict_filter=model.dict_filter,
                            direction_lambda=model.config["direction_lambda"])
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.model.parameters(), clip_value=1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 200 == 0:
                print(f"[train] epoch {epoch+1} step {step} loss {loss.item():.4f} "
                      f"({ram_gb():.1f} GB)", flush=True)
        scheduler.step()
        val_res = evaluate(val_loader, model.model, model.config["uncertainty"], "cpu")
        val_metrics, _ = compute_metrics(val_res)
        rec = {"epoch": epoch + 1,
               "train_loss_mean": float(np.mean(losses)),
               "train_loss_last": float(losses[-1]),
               "val_mse": float(val_metrics["mse"]),
               "val_mse_de": float(val_metrics["mse_de"]),
               "val_pearson": float(val_metrics["pearson"]),
               "val_pearson_de": float(val_metrics["pearson_de"]),
               "lr": float(scheduler.get_last_lr()[0]),
               "epoch_time_s": round(time.time() - te, 1)}
        history.append(rec)
        print("[epoch] " + json.dumps(rec), flush=True)
        pd.DataFrame(history).to_csv(
            os.path.join(args.out_dir, "training_history.csv"), index=False)
        torch.save({"state_dict": model.model.state_dict(), "epoch": epoch + 1,
                    "val_mse_de": val_metrics["mse_de"], "args": vars(args)},
                   os.path.join(ckpt_dir, f"epoch{epoch+1}_checkpoint.pt"))
        if float(val_metrics["mse_de"]) < best_val:
            best_val = float(val_metrics["mse_de"])
            torch.save({"state_dict": model.model.state_dict(), "epoch": epoch + 1,
                        "val_mse_de": best_val, "args": vars(args)},
                       os.path.join(args.out_dir, "best_checkpoint.pt"))
    # epoch-8 checkpoint == last epoch
    torch.save({"state_dict": model.model.state_dict(), "epoch": args.epochs,
                "val_mse_de": history[-1]["val_mse_de"], "args": vars(args)},
               os.path.join(args.out_dir, "epoch8_checkpoint.pt"))
    print(f"[done] best val mse_de {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
