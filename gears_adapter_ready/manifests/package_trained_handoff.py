"""Build a verified trained-project snapshot independently of count evaluation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import digest, timestamp, write_json


def build(files, path):
    with tarfile.open(path, "w:gz", compresslevel=1) as archive:
        for name, source in sorted(files.items()):
            archive.add(source, arcname="gears_adapter_ready/"+name, recursive=False)


def verify(path, expected):
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            assert member.isfile() and member.name.startswith("gears_adapter_ready/")
            name = member.name.removeprefix("gears_adapter_ready/")
            assert name not in seen and ".." not in Path(name).parts
            record = expected[name]
            assert member.size == record["bytes"]
            assert hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() == record["sha256"]
            seen.add(name)
    assert seen == set(expected)


def main():
    started = time.monotonic()
    work = Path("/workspace/gears_recovery_20261105/handoff")
    work.mkdir(parents=True, exist_ok=True)
    target = Path("/mnt/results/gears_rpe1_trained_project_handoff_v4.tar.gz")
    if target.exists():
        raise FileExistsError("Preserve existing releases; do not overwrite")
    files = {}
    for source in ROOT.rglob("*"):
        if (source.is_file() and "__pycache__" not in source.parts
                and ".git" not in source.parts and source.suffix != ".pyc"):
            name = str(source.relative_to(ROOT))
            if name == "CODEX_REVIEW_GUIDE.md":
                name = "documentation/future_complete_evaluation_review_guide.md"
            files[name] = source
    entries = {name: dict(bytes=p.stat().st_size, sha256=digest(p))
               for name, p in sorted(files.items())}
    original = json.loads((ROOT/"reports/completed_checkpoint_verification.json").read_text())
    assert len(original["checkpoints"]) == 8
    for record in original["checkpoints"]:
        assert entries[f"checkpoints/epoch_{record['epoch']:02d}.pt"]["sha256"] == record["sha256"]
    manifest = work/"trained_handoff_manifest.json"
    write_json(manifest, dict(
        created_at=timestamp(), purpose="Standalone preservation before remaining evaluation",
        completed_epochs=8, validation_best_epoch=3, final_epoch=8,
        adapter_trained=False, normalized_prediction_exports=4, embedding_exports=2,
        complete_count_evaluation=False, previously_saved_count_panel_receipts=2,
        count_panel_results_included=False, new_model_inference=False,
        retraining_performed=False, github_uploaded=False,
        challenge_leaderboard_result=False, files=entries))
    files[manifest.name] = manifest
    all_entries = {**entries, manifest.name:dict(bytes=manifest.stat().st_size, sha256=digest(manifest))}
    pilot_names = ["HANDOFF_START_HERE.md", "checkpoints/epoch_08.pt",
                   "adapter_interface/adapter_manifest.json", "tests/verify_trained_handoff.py"]
    pilot = work/"actual_artifact_packaging_pilot.tar.gz"
    build({name:files[name] for name in pilot_names}, pilot)
    verify(pilot, {name:all_entries[name] for name in pilot_names})
    local = work/target.name
    build(files, local)
    shutil.copyfile(local, target)
    print("DURABLE_HANDOFF_WRITTEN", target.name, flush=True)
    verify(target, all_entries)
    extracted = work/"readback_smoke"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(target, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    completed = subprocess.run([
        sys.executable, str(extracted/"gears_adapter_ready/tests/verify_trained_handoff.py"),
        "--root", str(extracted/"gears_adapter_ready")],
        check=True, capture_output=True, text=True)
    smoke = json.loads(completed.stdout)
    receipt = dict(created_at=timestamp(), filename=target.name,
                   bytes=target.stat().st_size, sha256=digest(target),
                   regular_files=len(all_entries), all_member_hashes_verified=True,
                   extracted_read_only_smoke=smoke, completed_epochs=8,
                   independent_of_remaining_count_evaluation=True,
                   no_training=True, no_model_inference=True,
                   complete_count_evaluation=False, github_uploaded=False,
                   seconds=time.monotonic()-started)
    write_json("/mnt/results/gears_rpe1_trained_project_handoff_v4_checksums.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
