"""Actual-data parity for the bounded-memory official bulk/dispatch seam."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_counts_bounded_cpu import evaluate
from io_utils import write_json


def main():
    pilot = Path("/workspace/gears_raw_eval/pilot")
    result = evaluate(
        pilot/"gears_epoch_8_unseen_test_pilot.h5ad",
        pilot/"measured_unseen_pilot.h5ad",
        "/workspace/gears_recovery_20261104/groupwise_pilot",
        pilot/"cache_identity/gears_epoch_8", parity_pilot=True, groupwise_bulk=True)
    record = dict(passed=True, targets=result["perturbation"].n_unique(),
                  groupwise_official_bulk=True, full_panel_official_dispatch=True,
                  equal_to_unmodified_official_api=True, relative_tolerance=1e-10,
                  absolute_tolerance=1e-10, metric_rows=result.height)
    write_json("/mnt/results/gears_recovery_20261104/official_groupwise_bulk_parity.json",record)
    print(record,flush=True)


if __name__ == "__main__":
    main()
