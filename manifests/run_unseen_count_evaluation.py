"""Resume-safe first full count evaluation; existing CPU only, no training."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import polars as pl
from evaluation.evaluate_counts_bounded_cpu import ChunkPaused, evaluate
from io_utils import digest, timestamp, write_json


def main():
    base = Path("/workspace/gears_raw_eval/full_unseen")
    destination = Path("/mnt/results/gears_adapter_ready/count_evaluation_v2/gears_epoch_8/unseen_test")
    checkpoint = Path("/mnt/shared-workspace/shared/gears_rpe1_official_de_cache")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"completion_receipt.json").exists():
        receipt = json.loads((destination/"completion_receipt.json").read_text())
        for name, record in receipt["files"].items():
            assert digest(destination/name) == record["sha256"]
        print("ALREADY_COMPLETED_VERIFIED_NO_REPEATED_EVALUATION", flush=True)
        return
    lock = (base/"official_evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pred = json.loads((base/"gears_epoch_8_unseen_test_manifest.json").read_text())
    real = json.loads((base/"measured_unseen_test.json").read_text())
    assert len(pred["conditions"]) == real["unique_targets"] == 384
    assert pred["cells_per_target"] == 400 and pred["background_fraction"] == 1e-4
    assert real["independent_reference_controls"] == 2297
    for record, name in ((pred, pred["filename"]), (real, real["filename"])):
        assert digest(base/name) == record["sha256"]
    available = psutil.virtual_memory().available
    # Raw CSR plus conservative transformation copies and DE-table/runtime overhead.
    budget = 3*(pred["sparse_resident_bytes_estimate"] +
                real["sparse_resident_bytes_estimate"]) + 2*1024**3
    if available < budget*1.15:
        raise MemoryError(f"Safety gate failed: estimated {budget} bytes, available {available}")
    attempt = str(time.time_ns())
    write_json(destination/("resource_preflight_"+attempt+".json"),
               dict(verified_at=timestamp(), available_bytes=available,
                    conservative_budget_bytes=budget, device="cpu", threads=4,
                    new_machine_or_paid_compute_requested=False))
    print("FULL_UNSEEN_CPU_PREFLIGHT_PASSED",
          json.dumps(dict(available_gib=available/1024**3,
                          conservative_budget_gib=budget/1024**3)), flush=True)
    started = time.monotonic()
    progress = Path("/mnt/shared-workspace/shared/gears_rpe1_count_progress")/attempt
    progress.parent.mkdir(parents=True, exist_ok=True)
    recovery_out = Path("/mnt/results/gears_recovery_20261104")
    recovery_out.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(
            base/pred["filename"], base/real["filename"], base/"evaluation",
            base/"cache"/"gears_epoch_8", checkpoint=checkpoint,
            max_minutes=45, progress=progress)
    except ChunkPaused as error:
        write_json(recovery_out/("epoch08_unseen_chunk_"+attempt+".json"),
                   dict(status="paused_between_targets", message=str(error),
                        completed_at=timestamp(), seconds=time.monotonic()-started,
                        progress_prefix=str(progress), resumable=True))
        print("COUNT_EVALUATION_CHUNK_PAUSED_SAFELY", attempt, flush=True)
        return
    actual_metrics = sorted(result["metric"].unique().to_list())
    assert result["perturbation"].n_unique() == 384
    canonical_metrics = [
        "pds_cosine", "expr_mse_unbiased_capped_norm",
        "de_wilcoxon_direction_fidelity_yield_raw",
        "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
        "de_wilcoxon_lfc_nmae"]
    summary = pl.read_csv(base/"evaluation"/"official_implementation_metric_summary.csv")
    assert set(canonical_metrics).issubset(set(summary["metric"].to_list()))
    for path in (base/"evaluation").glob("*"):
        if path.is_file():
            shutil.copyfile(path, destination/path.name)
    for name in ("gears_epoch_8_unseen_test_manifest.json", "measured_unseen_test.json"):
        shutil.copyfile(base/name, destination/name)
    write_json(destination/"completion_receipt.json",
               dict(completed_at=timestamp(), targets=384, methods=["gears_epoch_8"],
                    exposure="unseen_test", canonical_metrics=canonical_metrics,
                    emitted_per_target_metrics=actual_metrics, metric_rows=result.height,
                    seconds=time.monotonic()-started,
                    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
                           for p in sorted(destination.iterdir()) if p.is_file()}))
    print("FULL_UNSEEN_COUNT_EVALUATION_COMPLETE", str(destination), flush=True)


if __name__ == "__main__":
    main()
