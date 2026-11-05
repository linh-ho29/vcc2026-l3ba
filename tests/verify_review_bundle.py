"""Read-only review checks: hashes, exported-array schemas and metric coverage.

No checkpoint deserialization, model inference, training, or count generation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

METHODS = ["gears_epoch_2", "gears_epoch_4", "gears_epoch_6",
           "gears_epoch_8", "statistical_mean_effect", "control_only"]
CANONICAL = ["pds_cosine", "expr_mse_unbiased_capped_norm",
             "de_wilcoxon_lfc_nmae", "de_wilcoxon_direction_fidelity_yield_raw",
             "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard"]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_hashes(root):
    manifest = json.loads((root/"review_bundle_manifest.json").read_text())
    assert manifest["completed_epochs"] == 8 and not manifest["adapter_trained"]
    for name, record in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root/relative
        assert path.is_file() and path.stat().st_size == record["bytes"], name
        assert sha256(path) == record["sha256"], name
    for epoch in range(1, 9):
        assert f"checkpoints/epoch_{epoch:02d}.pt" in manifest["files"]
    return len(manifest["files"])


def verify_arrays(root):
    import numpy as np
    sys.path.insert(0, str(root))
    from adapter_interface.validation import validate_bundle
    for folder in ["embeddings", "embeddings_best_epoch_03"]:
        metadata = json.loads((root/folder/"embedding_manifest.json").read_text())
        with np.load(root/folder/"representations.npz", allow_pickle=False) as arrays:
            for name, record in metadata["representations"].items():
                values = arrays[name]
                assert list(values.shape) == record["shape"], (folder, name)
                assert values.ndim == 2 and values.shape[1] == 32
                assert np.isfinite(values).all()
    reference_genes = reference_conditions = None
    for epoch in [2, 4, 6, 8]:
        with np.load(root/f"predictions/epoch_{epoch:02d}_profiles.npz",
                     allow_pickle=False) as arrays:
            assert arrays["predicted_effects_mean"].shape == (1419, 5000)
            assert np.isfinite(arrays["predicted_effects_mean"]).all()
            genes, conditions = arrays["gene_ids"], arrays["conditions"]
            assert len(set(genes)) == 5000 and len(set(conditions)) == 1419
            if reference_genes is None:
                reference_genes, reference_conditions = genes.copy(), conditions.copy()
            else:
                assert np.array_equal(genes, reference_genes)
                assert np.array_equal(conditions, reference_conditions)
    with np.load(root/"adapter_interface/unseen_target_adapter_bundle.npz",
                 allow_pickle=False) as arrays:
        bundle = json.loads((root/"adapter_interface/adapter_manifest.json").read_text())
        bundle.update({key: arrays[key] for key in arrays.files})
        assert not bundle["adapter_trained"] and not bundle["measured_test_outcomes_used"]
        assert validate_bundle(bundle)
    return dict(embedding_exports=2, normalized_prediction_exports=4,
                adapter_schema_valid=True, model_inference=False)


def verify_panel(folder, expected_targets):
    receipt = json.loads((folder/"completion_receipt.json").read_text())
    assert receipt["targets"] == expected_targets
    for name, record in receipt["files"].items():
        assert (folder/name).stat().st_size == record["bytes"]
        assert sha256(folder/name) == record["sha256"]
    with (folder/"official_implementation_metric_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    metrics = {r["metric"]: r for r in rows}
    assert len(metrics) == len(rows) and set(CANONICAL).issubset(metrics)
    # Undefined results are retained as such, never filled or silently dropped.
    availability = {}
    for name in CANONICAL:
        row = metrics[name]
        mean = float(row["mean"]) if row["mean"] else None
        availability[name] = dict(mean=mean if mean is not None and math.isfinite(mean) else None,
                                  n_used=int(row["n_used"]), n_rows=int(row["n_rows"]),
                                  n_nan=int(row["n_nan"]), n_null=int(row["n_null"]),
                                  derived=row["derived"].lower() == "true")
        if name == "expr_mse_unbiased_capped_norm":
            assert availability[name]["derived"] and row["agg"] == "ratio_of_sums"
        else:
            assert int(row["n_rows"]) == expected_targets
    with (folder/"official_implementation_per_target_metrics.csv").open() as handle:
        tidy = list(csv.DictReader(handle))
    targets = {r["perturbation"] for r in tidy}
    assert len(targets) == expected_targets
    for metric in CANONICAL:
        if metric == "expr_mse_unbiased_capped_norm":
            continue
        selected = [r["perturbation"] for r in tidy if r["metric"] == metric]
        assert len(selected) == expected_targets and set(selected) == targets
    return availability


def verify_panels(root):
    return {f"{method}/{exposure}": verify_panel(
                root/"count_evaluation_v2"/method/exposure, targets)
            for method in METHODS
            for exposure, targets in [("seen_test", 1035), ("unseen_test", 384)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = dict(hashed_files=verify_hashes(root), schemas=verify_arrays(root),
                  count_panels=verify_panels(root), no_training=True,
                  no_model_inference=True, no_checkpoint_deserialization=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
