"""Create a portable, checksummed primary release without raw expression data."""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, digest, timestamp


def collect(result_root, report):
    result_root = Path(result_root)
    files = {}
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(p in {"__pycache__", ".git"} for p in path.parts):
            files[str(path.relative_to(ROOT))] = path
    # Overlay frozen data and final exports; do not duplicate older source variants.
    for folder in ["data", "reports", "embeddings", "embeddings_best_epoch_03",
                   "predictions", "adapter_interface", "evaluation"]:
        for path in (result_root / folder).iterdir():
            if path.is_file() and path.suffix not in {".py", ".pyc"}:
                files[str(path.relative_to(result_root))] = path
    for path in (result_root / "training_resume_20261103/reports").iterdir():
        if path.is_file():
            files["reports/" + path.name] = path
    verification = json.loads((result_root / "training_resume_20261103/reports/completed_checkpoint_verification.json").read_text())
    for record in verification["checkpoints"]:
        path = Path(record["filename"])
        files["checkpoints/" + path.name] = path
        files["checkpoints/" + path.with_suffix(".json").name] = path.with_suffix(".json")
    files["report_gears_eight_epoch_primary.md"] = Path(report)
    return files


def build(files, archive):
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname="gears_adapter_ready/" + name, recursive=False)


def verify(archive, expected):
    found = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith("gears_adapter_ready/")
            name = member.name.removeprefix("gears_adapter_ready/")
            assert name in expected and name not in found and ".." not in Path(name).parts
            found.add(name)
            source = tar.extractfile(member)
            actual = hashlib.file_digest(source, "sha256").hexdigest()
            assert actual == expected[name]["sha256"]
            assert member.size == expected[name]["bytes"]
    assert found == set(expected)


def package(result_root, report, work, destination):
    work, destination = Path(work), Path(destination)
    work.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    files = collect(result_root, report)
    entries = {name:dict(bytes=path.stat().st_size, sha256=digest(path)) for name,path in files.items()}
    # Actual-data pilot uses the README and final checkpoint, not synthetic model data.
    pilot_names = ["README.md", "checkpoints/epoch_08.pt"]
    pilot = work / "release_packaging_pilot.tar.gz"
    build({name:files[name] for name in pilot_names}, pilot)
    verify(pilot, {name:entries[name] for name in pilot_names})
    manifest_path = work / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(dict(created_at=timestamp(), files=entries,
                                            completed_epochs=8, adapter_trained=False,
                                            official_count_evaluation_complete=False,
                                            github_uploaded=False), indent=2))
    files["artifact_manifest.json"] = manifest_path
    entries["artifact_manifest.json"] = dict(bytes=manifest_path.stat().st_size, sha256=digest(manifest_path))
    name = "gears_rpe1_8epoch_primary_v2.tar.gz"
    local = work / name
    build(files, local)
    target = destination / name
    shutil.copyfile(local, target)
    # Read the durable copy, not the workspace archive.
    verify(target, entries)
    receipt = dict(created_at=timestamp(), filename=name, bytes=target.stat().st_size,
                   sha256=digest(target), regular_files=len(entries),
                   every_member_hash_verified_from_durable_archive=True,
                   completed_epochs=8, official_count_metrics_complete=False)
    (destination / "gears_rpe1_8epoch_primary_v2_checksums.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["result-root", "report", "work", "destination"]:
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    package(args.result_root, args.report, args.work, args.destination)
