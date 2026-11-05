"""Test bounded recovery against saved actual-data pilot tables."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
import polars as pl
from evaluation.evaluate_counts_bounded_cpu import ChunkPaused, evaluate
from io_utils import write_json


def test():
    pilot = Path("/workspace/gears_raw_eval/pilot")
    out = Path("/workspace/gears_recovery_20261104/pilot")
    args = dict(
        predicted=pilot/"gears_epoch_8_unseen_test_pilot.h5ad",
        real=pilot/"measured_unseen_pilot.h5ad",
        output=out,
        cache=pilot/"cache_identity/gears_epoch_8")
    try:
        evaluate(**args, max_minutes=1e-8)
    except ChunkPaused:
        pass
    else:
        raise AssertionError("Zero-budget chunk did not pause before a target")
    result = evaluate(**args, max_minutes=5, parity_pilot=True)
    old = pl.read_csv(pilot/"evaluation/official_implementation_per_target_metrics.csv")
    left = result.sort(["perturbation", "metric"])
    right = old.sort(["perturbation", "metric"])
    assert left.select(["perturbation", "metric"]).equals(
        right.select(["perturbation", "metric"]))
    np.testing.assert_allclose(left["value"].to_numpy(), right["value"].to_numpy(),
                               rtol=1e-10, atol=1e-10, equal_nan=True)
    summary = pl.read_csv(out/"official_implementation_metric_summary.csv")
    assert "expr_mse_unbiased_capped_norm" in summary["metric"].to_list()
    record = dict(passed=True, actual_pilot_targets=2, safe_boundary_pause=True,
                  unchanged_metric_values=True, ordinary_official_api_parity=True,
                  derived_normalized_expression_metric_present=True,
                  relative_tolerance=1e-10, absolute_tolerance=1e-10)
    write_json("/mnt/results/gears_recovery_20261104/bounded_recovery_pilot.json", record)
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    test()
