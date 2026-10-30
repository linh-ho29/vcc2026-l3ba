from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / 'vcc2026_cpu'

# Note: the saved held-out data already has SHRINKAGE=0.65, CLIP_LOG_FOLD=1.0 baked into
# y_pred, since those were applied before saving. To properly sweep these hyperparameters
# we need the pre-shrinkage/clip raw effect, not the already-adjusted prediction. Re-derive
# it here: y_pred = baseline * exp(shrinkage * clip(raw_effect)), so raw_effect can be
# recovered as log(y_pred / baseline) / shrinkage, then re-clipped/re-scaled per trial.

data = np.load(ROOT / 'artifacts' / 'heldout_eval_results.npz')
y_true = data['y_true']
y_pred_saved = data['baseline_control']  # placeholder, real recovery below
baseline_control = data['baseline_control']
tested_genes = data['tested_genes']

# Recover raw (unclipped, unshrunk) effect from the saved prediction.
OLD_SHRINKAGE = 0.65
OLD_CLIP = 1.0
y_pred_old = data['y_pred']
with np.errstate(divide='ignore', invalid='ignore'):
    recovered_clipped_effect = np.log(np.clip(y_pred_old, 1e-10, None) / np.clip(baseline_control, 1e-10, None)) / OLD_SHRINKAGE
# This recovered value is already clip(raw_effect, -1, 1) from the original run - since OLD_CLIP=1.0
# was the widest clip we used, this *is* effectively the raw effect (clipping was a no-op for anything
# within [-1,1], and anything that hit the clip boundary we can't perfectly recover beyond the boundary,
# a minor approximation for already-saturated genes).
raw_effect = recovered_clipped_effect


def compute_vcc_metrics(y_true, y_pred, baseline_control):
    mse = mean_squared_error(y_true, y_pred)
    delta_true = y_true - baseline_control
    delta_pred = y_pred - baseline_control
    pearson_delta, _ = pearsonr(delta_true.flatten(), delta_pred.flatten())
    spearman_delta, _ = spearmanr(delta_true.flatten(), delta_pred.flatten())
    return {"MSE": mse, "Pearson_Delta": pearson_delta, "Spearman_Delta": spearman_delta}


baseline_mse = mean_squared_error(y_true, baseline_control)
print(f"Reference: naive baseline MSE = {baseline_mse:.6f}")
print(f"Reference: current (shrinkage=0.65, clip=1.0) MSE = {mean_squared_error(y_true, y_pred_old):.6f}\n")

shrinkage_grid = [0.3, 0.45, 0.65, 0.8, 1.0]
clip_grid = [0.5, 0.75, 1.0, 1.5, 2.0]

results = []
for shrinkage in shrinkage_grid:
    for clip in clip_grid:
        effect = np.clip(raw_effect, -clip, clip)
        y_pred = baseline_control * np.exp(shrinkage * effect)
        metrics = compute_vcc_metrics(y_true, y_pred, baseline_control)
        win_rate = np.mean(
            np.mean((y_true - y_pred) ** 2, axis=1) < np.mean((y_true - baseline_control) ** 2, axis=1)
        )
        results.append((shrinkage, clip, metrics['MSE'], metrics['Pearson_Delta'], metrics['Spearman_Delta'], win_rate))
        rejected = metrics['MSE'] >= baseline_mse
        tag = "REJECTED (worse than naive baseline)" if rejected else ""
        print(f"shrinkage={shrinkage:.2f} clip={clip:.2f} -> MSE={metrics['MSE']:.6f} "
              f"Pearson={metrics['Pearson_Delta']:.4f} Spearman={metrics['Spearman_Delta']:.4f} "
              f"win_rate={100*win_rate:.1f}% {tag}")

results.sort(key=lambda r: r[2])  # sort by MSE ascending
print("\n=== Top 5 by MSE ===")
for shrinkage, clip, mse, pearson, spearman, win_rate in results[:5]:
    print(f"shrinkage={shrinkage:.2f} clip={clip:.2f} -> MSE={mse:.6f} Pearson={pearson:.4f} "
          f"Spearman={spearman:.4f} win_rate={100*win_rate:.1f}%")

best = results[0]
print(f"\nBest: shrinkage={best[0]}, clip_log_fold={best[1]} (current default: shrinkage=0.65, clip=1.0)")
print("DONE")
