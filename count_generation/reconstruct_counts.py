"""Optional heuristic count sampler, separate from the GEARS checkpoint.

Generated counts are model predictions, never measured data or official scores.
"""
import numpy as np


def reconstruct_counts(predicted_log_effect, fitting_control_counts, n_cells=400,
                       seed=42, background_fraction=1e-4):
    controls=np.asarray(fitting_control_counts)
    delta=np.asarray(predicted_log_effect,dtype=np.float64)
    if controls.ndim!=2 or delta.shape!=(controls.shape[1],):
        raise ValueError("Effect and control gene axes must match exactly")
    if not np.isfinite(controls).all() or np.any(controls<0) or np.any(controls%1):
        raise ValueError("Controls must be measured nonnegative integral counts")
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite log effects")
    depths=controls.sum(axis=1)
    if np.any(depths<=0) or np.any(depths>1_000_000):
        raise ValueError("Invalid library totals")
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(controls),size=n_cells,replace=True)
    sampled_depths=rng.choice(depths.astype(np.int64),size=n_cells,replace=True)
    background=controls.sum(axis=0).astype(np.float64)
    background/=background.sum()
    result=np.zeros((n_cells,controls.shape[1]),np.int32)
    for i,index in enumerate(indices):
        probability=controls[index].astype(np.float64)/depths[index]
        probability=(1-background_fraction)*probability+background_fraction*background
        # Stable exponentiation does not change normalized probabilities.
        weights=probability*np.exp(delta-delta.max())
        if weights.sum()<=0:
            raise ValueError("Degenerate reconstructed profile")
        weights/=weights.sum()
        result[i]=rng.multinomial(int(sampled_depths[i]),weights)
    return dict(
        counts=result,control_indices=indices,library_depths=sampled_depths,
        seed=seed,background_fraction=background_fraction,
        method="heuristic control resampling + exp(log-effect) reweighting + multinomial sampling",
        count_model_trained=False,
    )
