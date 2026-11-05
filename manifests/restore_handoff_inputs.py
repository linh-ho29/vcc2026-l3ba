"""Verify saved releases and restore their authorized overlays, without inference."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

WORK = Path("/workspace/gears_recovery_20261105")
ROOT = Path("/workspace/gears_adapter_ready")
OUT = Path("/mnt/results/gears_recovery_20261105")
SOURCE = Path("/mnt/shared-workspace/shared/gears_rpe1_recovery_sources/"
              "gears_recovery_source_1793809066847364595.tar.gz")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate(saved, receipt_path, manifest_name=None):
    receipt = json.loads(receipt_path.read_text())
    local = WORK/saved.name
    if local.exists():
        assert local.stat().st_size == receipt["bytes"]
        assert digest(local) == receipt["sha256"]
    else:
        shutil.copyfile(saved, local)
        assert local.stat().st_size == receipt["bytes"]
        assert digest(local) == receipt["sha256"]
    with tarfile.open(local, "r:gz") as archive:
        records = (json.loads(archive.extractfile(manifest_name).read())["files"]
                   if manifest_name else receipt["files"])
        seen = set()
        for member in archive:
            assert member.isfile() and not Path(member.name).is_absolute()
            assert ".." not in Path(member.name).parts
            name = (member.name.removeprefix("gears_adapter_ready/")
                    if manifest_name else member.name)
            assert name not in seen
            seen.add(name)
            if manifest_name and member.name == manifest_name:
                continue
            record = records[name]
            assert member.size == record["bytes"], name
            assert hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() == record["sha256"], name
        expected = set(records)
        if manifest_name:
            expected.add(manifest_name.removeprefix("gears_adapter_ready/"))
        assert seen == expected
    print("ARCHIVE_VERIFIED", local.name, len(seen), flush=True)
    return local, dict(filename=local.name, sha256=receipt["sha256"],
                       bytes=receipt["bytes"], verified_members=len(seen))


def restore(local, prefixed):
    with tarfile.open(local, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("gears_adapter_ready/") if prefixed else member.name
            target = ROOT/name
            assert ROOT.resolve() in target.resolve().parents
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are code/config overlays, never an authorization to change
            # any trained arrays or checkpoint in a previously restored tree.
            immutable = Path(name).parts[0] in {
                "checkpoints", "embeddings", "embeddings_best_epoch_03",
                "predictions", "adapter_interface"}
            if target.exists() and immutable and target.suffix not in {".py", ".md"}:
                expected = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()
                assert digest(target) == expected, name
                continue
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8*1024*1024)


def main():
    started = time.monotonic()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if ROOT.exists():
        raise FileExistsError("Restoration tree already exists; preserve and inspect before rerun")
    specifications = [
        (Path("/mnt/results/gears_rpe1_8epoch_primary_v2.tar.gz"),
         Path("/mnt/results/gears_rpe1_8epoch_primary_v2_checksums.json"),
         "gears_adapter_ready/artifact_manifest.json"),
        (Path("/mnt/results/gears_rpe1_primary_v2_reproducibility_patch_v3.tar.gz"),
         Path("/mnt/results/gears_rpe1_primary_v2_reproducibility_patch_v3_checksums.json"),
         "gears_adapter_ready/reproducibility_patch_v3_manifest.json"),
        (SOURCE, SOURCE.with_suffix(".json"), None)]
    checked = [validate(*spec) for spec in specifications]
    for (local, record), (_, _, manifest) in zip(checked, specifications):
        restore(local, prefixed=manifest is not None)
    for epoch in range(1, 9):
        assert (ROOT/f"checkpoints/epoch_{epoch:02d}.pt").is_file()
    result = dict(passed=True, archives=[record for _, record in checked],
                  restored_checkpoints=8, no_training=True, no_model_inference=True,
                  seconds=time.monotonic()-started)
    (OUT/"trained_project_restoration.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
