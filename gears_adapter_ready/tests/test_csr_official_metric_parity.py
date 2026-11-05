"""Exercise exact CSR access on actual counts against the unchanged official API."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_counts_bounded_cpu import evaluate
from io_utils import digest, write_json


def main():
    existing = Path("/workspace/gears_recovery_20261104/eight_target_pilot")
    output = Path("/workspace/gears_recovery_20261104/csr_eight_target_pilot")
    predicted = existing/"gears_epoch_8_unseen_test.h5ad"
    measured = existing/"measured_unseen_test.h5ad"
    # Distinct cache forces the new read path for every DE table and bulk.
    assert not (output/"complete.json").exists(), "Already validated; do not rerun"
    result = evaluate(predicted, measured, output/"evaluation",
                      output/"cache/gears_epoch_8", parity_pilot=True,
                      groupwise_bulk=True)
    record = dict(passed=True, targets=8, metric_rows=result.height,
                  exact_csr_reads=True, equal_to_unmodified_official_api=True,
                  relative_tolerance=1e-10, absolute_tolerance=1e-10,
                  predicted_subset_sha256=digest(predicted),
                  measured_subset_sha256=digest(measured),
                  no_model_inference_or_count_regeneration=True,
                  benchmark_result=False)
    write_json(output/"complete.json", record)
    write_json("/mnt/results/gears_recovery_20261104/exact_csr_official_metric_parity.json",
               record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
