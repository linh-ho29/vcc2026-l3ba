"""Recover approved sampler draws without regenerating any count predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import anndata as ad
import h5py
import numpy as np

from io_utils import digest, timestamp, write_json


def csr_row_totals(path, rows_per_chunk=2048):
    """One sequential pass over CSR values; avoid repeated backed matrix slices."""
    with h5py.File(path,"r") as handle:
        matrix = handle["X"]
        pointers = matrix["indptr"][:]
        totals = np.zeros(len(pointers)-1,dtype=np.int64)
        for start in range(0,len(totals),rows_per_chunk):
            stop = min(start+rows_per_chunk,len(totals))
            low, high = int(pointers[start]),int(pointers[stop])
            values = matrix["data"][low:high]
            cumulative = np.empty(len(values)+1,dtype=np.int64)
            cumulative[0] = 0
            np.cumsum(values,dtype=np.int64,out=cumulative[1:])
            relative = pointers[start:stop+1]-low
            totals[start:stop] = cumulative[relative[1:]]-cumulative[relative[:-1]]
        return totals


def export(matched, counts, manifest, output):
    output = Path(output)
    receipt_path = output.with_suffix(".json")
    if output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert digest(output) == receipt["sha256"]
        return receipt
    record = json.loads(Path(manifest).read_text())
    assert digest(counts) == record["sha256"]
    source = ad.read_h5ad(matched, backed="r")
    predicted = ad.read_h5ad(counts, backed="r")
    try:
        assert source.var_names.equals(predicted.var_names)
        fit = source.obs.role.astype(str).to_numpy() == "fit_control"
        control_ids = source.obs_names[fit].to_numpy().astype(str)
        assert len(control_ids) == 9188
        depths = csr_row_totals(matched)[fit]
        actual_totals = csr_row_totals(counts)
        predicted_conditions = predicted.obs.condition.astype(str).to_numpy()
        conditions = record["conditions"]
        all_indices, all_depths, seeds = [], [], []
        for condition in conditions:
            seed = int.from_bytes(hashlib.sha256(
                f"42:heuristic_counts:{condition}".encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(depths), size=400, replace=True)
            sampled_depths = rng.choice(depths, size=400, replace=True)
            selected = predicted_conditions == condition
            assert int(selected.sum()) == 400
            actual_depths = actual_totals[selected]
            np.testing.assert_array_equal(sampled_depths, actual_depths)
            seeds.append(seed)
            all_indices.append(indices.astype(np.int32))
            all_depths.append(sampled_depths.astype(np.int32))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial.npz")
        np.savez_compressed(
            temporary, conditions=np.asarray(conditions),
            seeds=np.asarray(seeds, dtype=np.uint32),
            fitting_control_cell_ids=control_ids,
            sampled_fitting_control_indices=np.stack(all_indices),
            sampled_library_depths=np.stack(all_depths))
        temporary.replace(output)
        receipt = dict(completed_at=timestamp(), filename=output.name, sha256=digest(output),
                       count_source_sha256=record["sha256"],
                       matched_counts_sha256=digest(matched), cells_per_target=400,
                       seed_base=42, targets=len(conditions),
                       fitting_control_cells=9188,
                       all_sampled_depths_match_generated_cell_totals=True,
                       count_predictions_regenerated=False,
                       derivation="Recover the sampler's first two RNG choices; no multinomial draws")
        write_json(receipt_path, receipt)
        return receipt
    finally:
        source.file.close()
        predicted.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ["matched", "counts", "manifest", "output"]:
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.matched, args.counts, args.manifest, args.output),indent=2))
