"""Post-interruption read-only audit; no training, inference, or artifact overwrite."""
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import polars as pl
from evaluation.evaluate_counts_bounded_cpu import config
from io_utils import digest, timestamp, write_json

RESULTS = Path("/mnt/results")
ROOT = RESULTS/"gears_adapter_ready"
OUT = RESULTS/"gears_recovery_20261104"
BASE = Path("/workspace/gears_raw_eval/full_unseen")
REVISION = "d86be3f4cd47015cd41046e9cc6a834777453594"


def audit_archive(receipt_name, manifest_name):
    receipt = json.loads((RESULTS/receipt_name).read_text())
    path = RESULTS/receipt["filename"]
    assert path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"]
    found, files = set(), {}
    with tarfile.open(path, "r:gz") as handle:
        manifest = json.load(handle.extractfile("gears_adapter_ready/"+manifest_name))
        entries = manifest["files"]
        for member in handle:
            assert member.isfile() and member.name.startswith("gears_adapter_ready/")
            name = member.name.removeprefix("gears_adapter_ready/")
            assert ".." not in Path(name).parts and name not in found
            found.add(name)
            actual = hashlib.file_digest(handle.extractfile(member), "sha256").hexdigest()
            if name != manifest_name:
                assert member.size == entries[name]["bytes"]
                assert actual == entries[name]["sha256"], name
                files[name] = entries[name]
    assert found == set(entries) | {manifest_name}
    return dict(filename=path.name, sha256=receipt["sha256"], verified_members=len(found)), files


def validate_table(path, source_hash, targets, genes):
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    record = json.loads(sidecar.read_text())
    target = record["target"]
    assert target in targets and record["source_sha256"] == source_hash
    assert path.stem == hashlib.sha256(target.encode()).hexdigest()[:20]
    assert path.stat().st_size > 0 and digest(path) == record["sha256"], str(path)
    frame = pl.read_parquet(path)
    assert frame.height == record["rows"] and frame.height > 0
    expected_columns = {"target", "feature", "log2_fold_change", "p_value",
                        "p_adj", "abs_log2_fold_change"}
    assert set(frame.columns) == expected_columns
    assert set(frame["target"].to_list()) == {target}
    assert frame["feature"].n_unique() == frame.height
    assert set(frame["feature"].to_list()).issubset(genes)
    return dict(target=target, filename=path.name, sha256=record["sha256"],
                bytes=path.stat().st_size, rows=frame.height)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", "/workspace/cell_eval2_source", "rev-parse", "HEAD"], text=True).strip()
    assert revision == REVISION
    env_record = json.loads((ROOT/"manifests/cell_eval_cpu_environment.json").read_text())
    assert env_record["source_commit"] == revision
    for package, expected in env_record["packages"].items():
        assert metadata.version(package) == expected, package
    primary_receipt, entries = audit_archive(
        "gears_rpe1_8epoch_primary_v2_checksums.json", "artifact_manifest.json")
    patch_receipt, _ = audit_archive(
        "gears_rpe1_primary_v2_reproducibility_patch_v3_checksums.json",
        "reproducibility_patch_v3_manifest.json")
    immutable = []
    for name, expected in entries.items():
        if Path(name).parts[0] not in {
            "data", "checkpoints", "embeddings", "embeddings_best_epoch_03",
            "predictions", "adapter_interface", "reports", "evaluation"
        } or Path(name).suffix in {".py", ".pyc"}:
            continue
        original = ROOT/name
        alternate = ROOT/"training_resume_20261103"/name
        path = alternate if alternate.exists() else original
        assert path.is_file() and path.stat().st_size == expected["bytes"], name
        assert digest(path) == expected["sha256"], name
        immutable.append(dict(archive_path=name, sha256=expected["sha256"]))
    # Current count files must retain their verified bytes and exact paired axes.
    counts = []
    for item in json.loads((ROOT/"reports/full_unseen_count_checkpoint_verification.json").read_text()):
        path = BASE/item["filename"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        counts.append(item)
    alignment = json.loads((ROOT/"reports/raw_alignment_manifest.json").read_text())
    matched = Path("/workspace/gears_raw_eval")/alignment["artifact_filename"]
    assert digest(matched) == alignment["artifact_sha256"]
    real = ad.read_h5ad(BASE/"measured_unseen_test.h5ad", backed="r")
    pred = ad.read_h5ad(BASE/"gears_epoch_8_unseen_test.h5ad", backed="r")
    try:
        assert real.var.equals(pred.var)
        assert real.n_vars == pred.n_vars == 5000
        assert real.n_obs == 39037 and pred.n_obs == 153664
        controls = real.obs_names[real.obs.target.astype(str) == "non-targeting"]
        assert len(controls) == 2297
        control_hash = hashlib.sha256("\n".join(controls).encode()).hexdigest()
        targets = set(real.obs.target.astype(str)) - {"non-targeting"}
        assert targets == set(pred.obs.target.astype(str)) - {"non-targeting"}
        assert len(targets) == 384
        genes = set(real.var_names)
    finally:
        real.file.close()
        pred.file.close()
    source_hash = next(r["sha256"] for r in counts if r["filename"] == "measured_unseen_test.h5ad")
    assert source_hash == digest(BASE/"measured_unseen_test.h5ad")
    expected_identity = dict(source_sha256=source_hash, control_cell_ids_sha256=control_hash,
                             config=asdict(config()))
    directories = [
        BASE/"cache/real_shared"/source_hash[:20],
        Path("/mnt/shared-workspace/shared/gears_rpe1_official_de_cache/real")/source_hash[:20]]
    records, locations, incomplete = {}, [], []
    for directory in directories:
        assert json.loads((directory/"identity.json").read_text()) == expected_identity
        validated = []
        for path in sorted(directory.glob("*.parquet")):
            record = validate_table(path, source_hash, targets, genes)
            if record is None:
                incomplete.append(str(path))
                continue
            if record["target"] in records:
                assert records[record["target"]]["sha256"] == record["sha256"]
            records[record["target"]] = record
            validated.append(record["target"])
        locations.append(dict(directory=str(directory), valid_tables=len(validated)))
    result = dict(verified_at=timestamp(), status="passed", source_revision=revision,
                  evaluation_packages_verified=len(env_record["packages"]),
                  archives=[primary_receipt, patch_receipt],
                  immutable_artifacts_verified=immutable, count_files=counts,
                  matched_counts_sha256=alignment["artifact_sha256"],
                  saved_reference_de_valid=len(records), remaining_reference_de=384-len(records),
                  remaining_prediction_de=384, cache_locations=locations,
                  validated_reference_tables=list(records.values()), incomplete_candidates=incomplete,
                  model_training_run=False, predictions_or_embeddings_regenerated=False,
                  original_artifacts_overwritten=False)
    write_json(OUT/"recovery_integrity_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in {
        "immutable_artifacts_verified", "validated_reference_tables"}}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
