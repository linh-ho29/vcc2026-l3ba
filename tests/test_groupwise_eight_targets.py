"""Parity on eight frozen-order targets taken from existing complete predictions."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anndata as ad
from evaluation.evaluate_counts_bounded_cpu import evaluate
from io_utils import digest, write_json


def main():
    base = Path("/workspace/gears_raw_eval/full_unseen")
    out = Path("/workspace/gears_recovery_20261104/eight_target_pilot")
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((base/"gears_epoch_8_unseen_test_manifest.json").read_text())
    targets = [x.removesuffix("+ctrl") for x in manifest["conditions"][:8]]
    paths = []
    for name in ["gears_epoch_8_unseen_test.h5ad","measured_unseen_test.h5ad"]:
        destination = out/name
        if not destination.exists():
            data = ad.read_h5ad(base/name,backed="r")
            try:
                selected = data.obs.target.astype(str).isin(targets+["non-targeting"]).to_numpy()
                subset = data[selected].to_memory()
                subset.write_h5ad(destination,compression="gzip",compression_opts=1)
            finally:
                data.file.close()
        paths.append(destination)
    result = evaluate(
        paths[0],paths[1],out/"evaluation",out/"cache/gears_epoch_8",
        parity_pilot=True,groupwise_bulk=True)
    record = dict(passed=True,targets=8,selection="first eight frozen-order unseen targets",
                  metric_rows=result.height,groupwise_official_bulk=True,
                  full_panel_official_dispatch=True,
                  equal_to_unmodified_official_api=True,
                  relative_tolerance=1e-10,absolute_tolerance=1e-10,
                  predicted_subset_sha256=digest(paths[0]),
                  measured_subset_sha256=digest(paths[1]),
                  no_model_inference_or_count_regeneration=True,
                  benchmark_result=False)
    write_json("/mnt/results/gears_recovery_20261104/official_groupwise_bulk_eight_target_parity.json",record)
    print(record,flush=True)


if __name__ == "__main__":
    main()
