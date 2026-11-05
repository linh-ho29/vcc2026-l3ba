"""Budget-conscious hyperparameter sweep runner.

Runs each trial in ``SweepConfig.trials`` for a short, fixed number of
epochs (``epochs_per_trial`` - enough to rank configs, not to fully
converge), tracks wall-clock time against ``max_total_minutes``, and
stops early (keeping whatever trials finished) if the budget would be
exceeded. Ranks trials by held-out validation loss, not training loss.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional

from train import PerturbSeqDataset, evaluate, train
from src.config import ModelConfig, SweepConfig, TrainingConfig


def _apply_overrides(base, overrides: dict):
    if not overrides:
        return base
    return replace(base, **overrides)


def run_sweep(
    base_model_cfg: ModelConfig,
    base_train_cfg: TrainingConfig,
    train_ds: PerturbSeqDataset,
    val_ds: PerturbSeqDataset,
    sweep_cfg: Optional[SweepConfig] = None,
) -> Dict:
    sweep_cfg = sweep_cfg or SweepConfig()
    results: List[dict] = []
    t_start = time.monotonic()

    for trial in sweep_cfg.trials:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        if elapsed_min >= sweep_cfg.max_total_minutes:
            print(f"[tune] budget exhausted ({elapsed_min:.1f}/{sweep_cfg.max_total_minutes:.0f} min) "
                  f"- stopping sweep before trial {trial['name']!r}.")
            break

        model_cfg = _apply_overrides(base_model_cfg, trial.get("model", {}))
        train_cfg = _apply_overrides(base_train_cfg, trial.get("train", {}))
        train_cfg = replace(train_cfg, max_epochs=sweep_cfg.epochs_per_trial)

        t0 = time.monotonic()
        print(f"\n[tune] trial {trial['name']!r}: model_overrides={trial.get('model', {})} "
              f"train_overrides={trial.get('train', {})}")
        try:
            model, history = train(model_cfg, train_cfg, train_ds, val_dataset=val_ds, verbose=False)
            final_val = history["val_loss"][-1]
            best_val = min(history["val_loss"])
        except Exception as exc:  # a bad config shouldn't kill the whole sweep
            print(f"[tune] trial {trial['name']!r} FAILED: {exc}")
            results.append({"name": trial["name"], "error": str(exc), "seconds": time.monotonic() - t0})
            continue

        seconds = time.monotonic() - t0
        print(f"[tune] trial {trial['name']!r}: final_val_loss={final_val:.4f} "
              f"best_val_loss={best_val:.4f} ({seconds:.1f}s)")
        results.append({
            "name": trial["name"], "final_val_loss": final_val, "best_val_loss": best_val,
            "seconds": seconds, "model_overrides": trial.get("model", {}),
            "train_overrides": trial.get("train", {}),
        })

    successful = [r for r in results if "best_val_loss" in r]
    if not successful:
        raise RuntimeError("Every sweep trial failed - nothing to select from.")
    best = min(successful, key=lambda r: r["best_val_loss"])

    total_minutes = (time.monotonic() - t_start) / 60.0
    print(f"\n[tune] sweep done in {total_minutes:.1f} min. Best trial: {best['name']!r} "
          f"(val_loss={best['best_val_loss']:.4f})")

    best_model_cfg = _apply_overrides(base_model_cfg, best["model_overrides"])
    best_train_cfg = _apply_overrides(base_train_cfg, best["train_overrides"])
    return {
        "results": results, "best": best,
        "best_model_cfg": best_model_cfg, "best_train_cfg": best_train_cfg,
        "total_minutes": total_minutes,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from train import load_dataset_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--num-genes", type=int, required=True)
    parser.add_argument("--d-latent", type=int, default=128)
    parser.add_argument("--max-total-minutes", type=float, default=90.0)
    parser.add_argument("--epochs-per-trial", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("sweep_results.json"))
    args = parser.parse_args()

    train_ds = load_dataset_from_disk(args.train_dir)
    val_ds = load_dataset_from_disk(args.val_dir)

    base_model_cfg = ModelConfig(num_genes=args.num_genes, d_latent=args.d_latent)
    base_train_cfg = TrainingConfig()
    sweep_cfg = SweepConfig(max_total_minutes=args.max_total_minutes, epochs_per_trial=args.epochs_per_trial)

    outcome = run_sweep(base_model_cfg, base_train_cfg, train_ds, val_ds, sweep_cfg)
    with open(args.out, "w") as f:
        json.dump({
            "results": outcome["results"], "best": outcome["best"],
            "best_model_cfg": asdict(outcome["best_model_cfg"]),
            "best_train_cfg": asdict(outcome["best_train_cfg"]),
            "total_minutes": outcome["total_minutes"],
        }, f, indent=2, default=str)
    print(f"\nSaved sweep results to {args.out}")
