"""Fail-closed complete review packaging; never alter historical releases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tests"))
from io_utils import digest, timestamp, write_json
from verify_review_bundle import verify_arrays, verify_panels


def collect(results, report, recovery):
    files = {}
    for path in ROOT.rglob("*"):
        if (path.is_file() and "__pycache__" not in path.parts
                and ".git" not in path.parts and path.suffix != ".pyc"):
            files[str(path.relative_to(ROOT))] = path
    for folder in ["data", "reports", "embeddings", "embeddings_best_epoch_03",
                   "predictions", "adapter_interface", "evaluation"]:
        for path in (results/folder).iterdir():
            if path.is_file() and path.suffix not in {".py", ".pyc", ".h5ad"}:
                files[str(path.relative_to(results))] = path
    for path in (results/"training_resume_20261103/reports").iterdir():
        if path.is_file():
            files["reports/"+path.name] = path
    receipt = json.loads((results/"training_resume_20261103/reports/"
                         "completed_checkpoint_verification.json").read_text())
    assert len(receipt["checkpoints"]) == 8
    for record in receipt["checkpoints"]:
        path = Path(record["filename"])
        assert digest(path) == record["sha256"]
        for source in [path, path.with_suffix(".json")]:
            files["checkpoints/"+source.name] = source
    for path in (results/"count_evaluation_v2").rglob("*"):
        if path.is_file() and path.suffix != ".h5ad":
            files[str(path.relative_to(results))] = path
    # Only completed audit artifacts, never volatile progress or process logs.
    names = [
        "primary_artifact_validation.json", "recovery_integrity_audit.json",
        "embedding_diagnostics.csv", "eight_epoch_validation_history.csv",
        "exact_csr_row_read_parity.json", "exact_csr_official_metric_parity.json",
        "official_groupwise_bulk_parity.json",
        "official_groupwise_bulk_eight_target_parity.json",
        "epoch246_unseen_count_checkpoint_verification.json",
        "immutable_progress_storage_test.json",
        "source_checkpoint_verification_v2.json",
    ]
    for name in names:
        path = recovery/name
        if path.is_file():
            files["reports/recovery/"+name] = path
    for folder in recovery.iterdir():
        if folder.is_dir() and folder.name.endswith("_completed"):
            for path in folder.iterdir():
                if path.is_file() and path.name != "checkpoint_audit_progress.json":
                    files["reports/recovery/"+folder.name+"/"+path.name] = path
    files["reports/final_count_report.md"] = report
    files["report_gears_eight_epoch_primary.md"] = Path(
        "/mnt/results/report_gears_eight_epoch_primary.md")
    files["requirements_resolved.txt"] = Path(
        "/workspace/gears_primary_release/requirements_resolved.txt")
    files["evaluation/cpu_constraints.txt"] = Path("/workspace/cell_eval_cpu_constraints.txt")
    files["manifests/cell_eval_cpu_environment.json"] = (
        results/"manifests/cell_eval_cpu_environment.json")
    return files


def build(files, destination):
    with tarfile.open(destination, "w:gz", compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname="gears_adapter_ready/"+name, recursive=False)


def verify(archive, entries):
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            assert member.isfile() and member.name.startswith("gears_adapter_ready/")
            name = member.name.removeprefix("gears_adapter_ready/")
            assert name in entries and name not in seen and ".." not in Path(name).parts
            assert member.size == entries[name]["bytes"]
            assert hashlib.file_digest(handle.extractfile(member), "sha256").hexdigest() == entries[name]["sha256"]
            seen.add(name)
    assert seen == set(entries)


def package(results, report, recovery, work, destination):
    results, report, recovery = Path(results), Path(report), Path(recovery)
    work, destination = Path(work), Path(destination)
    assert report.is_file(), "A final scientific report is required"
    # This is the publication gate: missing panels stop before any archive write.
    panels = verify_panels(results)
    assert len(panels) == 12
    schemas = verify_arrays(results)
    files = collect(results, report, recovery)
    assert "vendor/cell-eval2/SOURCE_COMMIT" in files
    assert "vendor/cell-eval2/src/cell_eval2/__init__.py" in files
    assert "CODEX_REVIEW_GUIDE.md" in files
    work.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/"gears_rpe1_complete_review_v4.tar.gz"
    if target.exists():
        raise FileExistsError("Never overwrite an existing release; use a new version")
    entries = {name: dict(bytes=p.stat().st_size, sha256=digest(p))
               for name, p in sorted(files.items())}
    manifest = work/"review_bundle_manifest.json"
    write_json(manifest, dict(
        created_at=timestamp(), completed_epochs=8, validation_best_epoch=3,
        adapter_trained=False, count_panels_complete=12,
        calibrated_aggregate_available=False, challenge_leaderboard_result=False,
        github_uploaded=False, array_schema_checks=schemas,
        metric_coverage=panels, files=entries))
    files[manifest.name] = manifest
    entries[manifest.name] = dict(bytes=manifest.stat().st_size, sha256=digest(manifest))
    pilot_names = ["CODEX_REVIEW_GUIDE.md", "checkpoints/epoch_08.pt",
                   "tests/verify_review_bundle.py"]
    pilot = work/"review_packaging_pilot.tar.gz"
    build({n: files[n] for n in pilot_names}, pilot)
    verify(pilot, {n: entries[n] for n in pilot_names})
    local = work/target.name
    build(files, local)
    shutil.copyfile(local, target)
    verify(target, entries)
    # Extract only our just-verified regular members, then run read-only smoke.
    extracted = work/"extracted_smoke"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(target, "r:gz") as handle:
        handle.extractall(extracted, filter="data")
    subprocess.run([sys.executable, str(extracted/"gears_adapter_ready/tests/verify_review_bundle.py"),
                    "--root", str(extracted/"gears_adapter_ready")], check=True)
    receipt = dict(created_at=timestamp(), filename=target.name,
                   bytes=target.stat().st_size, sha256=digest(target),
                   regular_files=len(entries), completed_epochs=8,
                   completed_count_panels=12, all_member_hashes_verified=True,
                   extracted_read_only_smoke_passed=True, no_training=True,
                   no_model_inference=True, github_uploaded=False)
    write_json(destination/"gears_rpe1_complete_review_v4_checksums.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["results", "report", "recovery", "work", "destination"]:
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    package(args.results, args.report, args.recovery, args.work, args.destination)
