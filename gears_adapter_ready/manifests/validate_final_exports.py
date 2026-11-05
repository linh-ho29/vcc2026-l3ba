"""Validate completed artifacts without retraining or changing test settings."""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from adapter_interface.validation import validate_bundle
from common import digest, timestamp, write_json


def validate(root, output):
    root, output = Path(root), Path(output)
    verification_path = root / "reports/completed_checkpoint_verification.json"
    if not verification_path.exists():
        verification_path = root / "training_resume_20261103/reports/completed_checkpoint_verification.json"
    verified = json.loads(verification_path.read_text())
    checkpoint_records = []
    identities = None
    final = None
    for record in verified["checkpoints"]:
        path = root / "checkpoints" / Path(record["filename"]).name
        if not path.exists():
            path = root / "training_resume_20261103/checkpoints" / path.name
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["epoch"] == record["epoch"] and state["cursor"] == 0
        assert state["global_step"] == state["epoch"] * 11306
        assert all(torch.isfinite(v).all() for v in state["model"].values())
        keys = ["source_sha256", "split_sha256", "cell_manifest_sha256", "fit_row_sha256",
                "gene_order_sha256", "target_order_sha256", "graph_sha256"]
        current_identity = {k: state["dataset"][k] for k in keys}
        if identities is None:
            identities = current_identity
        assert identities == current_identity
        checkpoint_records.append(record)
        if state["epoch"] == 8:
            final = state
        else:
            del state
    assert final is not None and final["epoch"] == 8
    assert final["best_epoch"] == 3
    assert final["scheduler"]["last_epoch"] == 8
    assert final["batch_size"] == 8 and final["graph"]["model_config"]["hidden_size"] == 32
    genes = final["dataset"]["gene_ids"]
    symbols = final["dataset"]["gene_symbols"]
    conditions = None
    controls = None
    predictions = []
    for epoch in [2, 4, 6, 8]:
        path = root / "predictions" / f"epoch_{epoch:02d}_profiles.npz"
        manifest = json.loads(path.with_name(f"epoch_{epoch:02d}_prediction_manifest.json").read_text())
        assert manifest["checkpoint_sha256"] == checkpoint_records[epoch-1]["sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert data["gene_ids"].tolist() == genes
            assert data["gene_symbols"].tolist() == symbols
            assert data["predicted_expression_mean"].shape == (1419, 5000)
            if conditions is None:
                conditions, controls = data["conditions"].copy(), data["control_cell_ids"].copy()
            else:
                np.testing.assert_array_equal(conditions, data["conditions"])
                np.testing.assert_array_equal(controls, data["control_cell_ids"])
            for key in data.files:
                if data[key].dtype.kind in "fci":
                    assert np.isfinite(data[key]).all(), (epoch, key)
            np.testing.assert_allclose(data["predicted_effects_mean"],
                                       data["predicted_expression_mean"] - data["control_expression_mean"],
                                       atol=5e-6, rtol=1e-5)
        predictions.append(dict(epoch=epoch, shape=[1419, 5000], sha256=digest(path),
                                seconds=manifest["seconds"]))
        gc.collect()
    embedding_records = []
    for folder in ["embeddings", "embeddings_best_epoch_03"]:
        manifest = json.loads((root / folder / "embedding_manifest.json").read_text())
        with np.load(root / folder / "representations.npz", allow_pickle=False) as data:
            for name in data.files:
                values = data[name]
                expected = (9853 if name.startswith("target_") else 5000, 32)
                assert values.shape == expected and np.isfinite(values).all()
                singular = np.linalg.svd(values.astype(np.float64), compute_uv=False)
                probabilities = singular[singular > 0] / singular.sum()
                effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
                assert abs(effective_rank-manifest["representations"][name]["effective_rank"]) < 1e-8
                embedding_records.append(dict(checkpoint_epoch=manifest["checkpoint_epoch"],
                                              representation=name, rows=len(values), columns=32,
                                              effective_rank=effective_rank,
                                              zero_norm_rows=int((np.linalg.norm(values,axis=1)==0).sum())))
    adapter_dir = root / "adapter_interface"
    adapter_manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text())
    with np.load(adapter_dir / "unseen_target_adapter_bundle.npz", allow_pickle=False) as data:
        bundle = {**adapter_manifest, **{k:data[k] for k in data.files}}
        validate_bundle(bundle)
        assert bundle["predicted_effects"].shape == (1,64,5000)
        assert not bundle["adapter_trained"] and not bundle["measured_test_outcomes_used"]
        assert bundle["checkpoint_sha256"] == checkpoint_records[-1]["sha256"]
    table = pd.read_csv(root / "evaluation/comparison_normalized_metrics.csv")
    per_target = pd.read_csv(root / "evaluation/per_target_normalized_metrics.csv")
    assert len(table) == 12 and len(per_target) == 1419 * 6
    assert set(table["n_targets"]) == {384, 1035}
    for name in ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
                 "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
                 "de_wilcoxon_sig_jaccard", "official_aggregate"]:
        assert table[name].isna().all()
    pd.DataFrame(embedding_records).to_csv(output.parent / "embedding_diagnostics.csv", index=False)
    pd.DataFrame(final["history"]).to_csv(output.parent / "eight_epoch_validation_history.csv", index=False)
    result = dict(verified_at=timestamp(), status="primary_artifacts_valid", completed_epochs=8,
                  global_step=90448, best_validation_epoch=3, checkpoint_records=checkpoint_records,
                  immutable_identity=identities, predictions=predictions,
                  embedding_records=embedding_records, adapter_bundle_valid=True,
                  adapter_trained=False, trained_interface_smoke=json.loads(
                      (root / "reports/trained_epoch08_interface_smoke.json").read_text()),
                  normalized_metric_rows=len(table), per_target_metric_rows=len(per_target),
                  training_elapsed_seconds_committed=final["elapsed_training_seconds"],
                  peak_training_rss_gib=max(r["peak_rss_gib"] for r in final["history"]),
                  official_count_metrics_complete=False, github_uploaded=False)
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in [
        "checkpoint_records", "immutable_identity", "embedding_records", "trained_interface_smoke"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate(args.root, args.output)
