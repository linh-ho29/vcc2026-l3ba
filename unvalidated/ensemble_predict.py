#!/usr/bin/env python
"""
Ensemble combination of perturbation-response predictions across models
(scGPT + GEARS + custom baselines), all sharing the standard prediction
contract: test_pred.npy, test_truth.npy, test_pertcat.npy, test_de_idx.npy,
ctrl_mean.npy in each model dir, with test conditions in IDENTICAL order
(guaranteed when every model is evaluated on the same dataloader/split).

Combination operates on predicted DELTAS (pred - ctrl_mean), which is the
right scale for averaging: absolute profiles are dominated by the shared
control mean, deltas carry the actual perturbation signal.

Strategies:
  mean      - uniform average of member deltas
  weighted  - average weighted by fixed per-model weights (--weights)
  median    - per-gene median of member deltas (robust to one bad member)

Outputs are written in the SAME standard contract, so the ensemble plugs
directly into aggregate_metrics.py, compare_models.py, make_figures_combined.py
and build_report_k562.py as if it were another model.

Example:
  python ensemble_predict.py \
    --model_dirs gears=/mnt/shared-workspace/preds/gears \
                 scgpt=/mnt/shared-workspace/preds/scgpt \
                 meanshift=/mnt/shared-workspace/preds/meanshift \
                 ridge=/mnt/shared-workspace/preds/ridge \
    --strategies mean,weighted,median \
    --weights gears=0.4,scgpt=0.4,meanshift=0.1,ridge=0.1 \
    --out_dir /mnt/shared-workspace/preds/ensemble

To generate the scGPT member first:
  predict.py --model scgpt --dataset custom --data_path ... --ft_ckpt <finetuned.pt> \
    --out_dir /mnt/shared-workspace/preds/scgpt
(train_scgpt.py produces the fine-tuned checkpoint; see its --help)
"""
import argparse, json, os
from pathlib import Path
import numpy as np

REQUIRED = ["test_pred.npy", "test_truth.npy", "test_pertcat.npy",
            "test_de_idx.npy", "ctrl_mean.npy"]


def load_member(name, path):
    """Load one member's arrays; squeeze truth to (cells, genes)."""
    for f in REQUIRED:
        if not Path(path, f).exists():
            raise FileNotFoundError(f"[{name}] missing {f} in {path}")
    pred = np.load(Path(path, "test_pred.npy"))
    truth = np.load(Path(path, "test_truth.npy"))
    if truth.ndim == 3:
        truth = truth[:, 0, :]
    pert = np.load(Path(path, "test_pertcat.npy"), allow_pickle=True)
    de_idx = np.load(Path(path, "test_de_idx.npy"))
    ctrl = np.load(Path(path, "ctrl_mean.npy")).astype(np.float32)
    return {"name": name, "pred": pred.astype(np.float32), "truth": truth,
            "pert": pert, "de_idx": de_idx, "ctrl": ctrl}


def check_alignment(members):
    """Members must describe the exact same test cells in the same order."""
    ref = members[0]
    if ref["pred"].shape != ref["truth"].shape:
        raise ValueError(f"[{ref['name']}] pred shape {ref['pred'].shape} != truth {ref['truth'].shape}")
    for m in members[1:]:
        if m["pred"].shape != ref["pred"].shape:
            raise ValueError(f"shape mismatch: {m['name']} {m['pred'].shape} vs {ref['name']} {ref['pred'].shape}")
        if not np.array_equal(m["pert"], ref["pert"]):
            raise ValueError(f"test_pertcat order differs between {m['name']} and {ref['name']} - "
                             "members were not evaluated on the same dataloader order")
        if not np.array_equal(m["de_idx"], ref["de_idx"]):
            raise ValueError(f"test_de_idx differs between {m['name']} and {ref['name']}")
        if not np.allclose(m["ctrl"], ref["ctrl"], atol=1e-5):
            raise ValueError(f"ctrl_mean differs between {m['name']} and {ref['name']}")


def combine(members, strategy, weights=None):
    """Combine member deltas; return combined absolute predictions."""
    ref = members[0]
    deltas = np.stack([m["pred"] - m["ctrl"][None, :] for m in members])  # (M, cells, genes)
    if strategy == "mean":
        comb_delta = deltas.mean(0)
    elif strategy == "median":
        comb_delta = np.median(deltas, axis=0)
    elif strategy == "weighted":
        if not weights:
            raise ValueError("--weights required for the weighted strategy")
        names = [m["name"] for m in members]
        missing = set(weights) - set(names)
        if missing:
            raise ValueError(f"weights reference unknown members: {missing}")
        w = np.array([float(weights[m["name"]]) for m in members], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / w.sum()
        comb_delta = np.tensordot(w, deltas, axes=1).astype(np.float32)
        print(f"[weighted] normalized weights: {dict(zip(names, np.round(w, 4)))}")
    else:
        raise ValueError(f"unknown strategy '{strategy}'")
    return comb_delta + ref["ctrl"][None, :]


def save_contract(out_dir, pred, ref):
    """Write the standard contract (truth/de_idx/pertcat copied from the reference)."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(Path(out_dir, "test_pred.npy"), pred.astype(np.float32))
    np.save(Path(out_dir, "test_truth.npy"), ref["truth"][:, None, :])
    np.save(Path(out_dir, "test_pertcat.npy"), ref["pert"])
    np.save(Path(out_dir, "test_de_idx.npy"), ref["de_idx"])
    np.save(Path(out_dir, "ctrl_mean.npy"), ref["ctrl"])
    meta = {"members": [m["name"] for m in ref.get("_members", [])],
            "note": "ensemble; truth/de_idx/pertcat copied from reference member"}
    json.dump(meta, open(Path(out_dir, "meta.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dirs", nargs="+", required=True,
                    help="name=path pairs, e.g. gears=/mnt/shared-workspace/preds/gears")
    ap.add_argument("--strategies", default="mean",
                    help="comma list from {mean,weighted,median}")
    ap.add_argument("--weights", nargs="*", default=[],
                    help="name=float pairs for the weighted strategy, e.g. gears=0.4 scgpt=0.4")
    ap.add_argument("--out_root", default="/mnt/shared-workspace/preds",
                    help="ensembles are written to <out_root>/ensemble_<strategy>")
    args = ap.parse_args()

    weights = {}
    for w in args.weights:
        k, v = w.split("=", 1)
        weights[k] = float(v)

    members = [load_member(name, path) for name, path in
               (s.split("=", 1) for s in args.model_dirs)]
    check_alignment(members)
    print(f"[align] {len(members)} members, {members[0]['pred'].shape[0]} test cells, "
          f"{members[0]['pred'].shape[1]} genes - all aligned")

    for strat in args.strategies.split(","):
        strat = strat.strip()
        pred = combine(members, strat, weights)
        # members list for meta
        for m in members:
            m["_members"] = members
        out_dir = Path(args.out_root) / f"ensemble_{strat}"
        save_contract(str(out_dir), pred, members[0])
        print(f"[saved] {strat} -> {out_dir} (contract files + meta.json)")

    print("[done] evaluate with: aggregate_metrics.py --preds_dir <ensemble_dir> ... "
          "or compare_models.py --preds ensemble_mean=<ensemble_dir> ...")


if __name__ == "__main__":
    main()
