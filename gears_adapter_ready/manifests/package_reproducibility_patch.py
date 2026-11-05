"""Package the versioned source-only overlay; never alter trained artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import digest, timestamp


def package():
    root = Path(__file__).resolve().parents[1]
    work = Path("/workspace/gears_primary_release/reproducibility_patch_v3")
    work.mkdir(parents=True, exist_ok=True)
    source_names = [
        "README.md", "requirements.txt", "REPRODUCIBILITY_PATCH_V3.md",
        "io_utils.py", "data/download_raw_counts.py", "data/prepare_raw_counts.py",
        "data/export_count_references.py", "count_generation/export_count_panel.py",
        "count_generation/reconstruct_counts.py",
        "evaluation/evaluate_vcc2026.py", "evaluation/evaluate_counts_bounded_cpu.py",
        "tests/verify_count_export_equivalence.py",
        "manifests/package_reproducibility_patch.py"]
    files = {n:root/n for n in source_names}
    files["requirements_resolved.txt"] = Path("/workspace/gears_primary_release/requirements_resolved.txt")
    files["evaluation/cpu_constraints.txt"] = Path("/workspace/cell_eval_cpu_constraints.txt")
    results = Path("/mnt/results/gears_adapter_ready")
    files["manifests/cell_eval_cpu_environment.json"] = results/"manifests/cell_eval_cpu_environment.json"
    for name in [
        "raw_download_verified.json", "raw_alignment_manifest.json",
        "cpu_precomputation_parity.json", "matched_counts_checkpoint_verification.json",
        "full_unseen_count_checkpoint_verification.json", "direct_csr_export_parity.json"
    ]:
        files["reports/"+name] = results/"reports"/name
    evaluator = Path("/workspace/cell_eval2_source")
    for name in ["LICENSE", "README.md", "pyproject.toml"]:
        files["vendor/cell-eval2/"+name] = evaluator/name
    for path in (evaluator/"src").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files["vendor/cell-eval2/"+str(path.relative_to(evaluator))] = path
    entries = {name:dict(bytes=p.stat().st_size, sha256=digest(p))
               for name,p in sorted(files.items())}
    manifest = work/"reproducibility_patch_v3_manifest.json"
    manifest.write_text(json.dumps(dict(
        created_at=timestamp(), base_archive="gears_rpe1_8epoch_primary_v2.tar.gz",
        base_sha256="6034f052738296afbff3d255d9a069993f1ddf3a7bbb5fbe52c06522b67f0480",
        checkpoint_or_model_artifacts_changed=False, files=entries),indent=2)+"\n")
    files[manifest.name] = manifest
    entries[manifest.name] = dict(bytes=manifest.stat().st_size,sha256=digest(manifest))
    archive = work/"gears_rpe1_primary_v2_reproducibility_patch_v3.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=6) as handle:
        for name,p in sorted(files.items()):
            handle.add(p,arcname="gears_adapter_ready/"+name,recursive=False)
    durable = Path("/mnt/results")/archive.name
    shutil.copyfile(archive,durable)
    receipt = dict(created_at=timestamp(),filename=durable.name,
                   bytes=durable.stat().st_size,sha256=digest(durable),
                   expected_regular_files=len(entries))
    seen = set()
    with tarfile.open(durable,"r:gz") as handle:
        for member in handle:
            assert member.isfile() and ".." not in Path(member.name).parts
            name = member.name.removeprefix("gears_adapter_ready/")
            assert name in entries and name not in seen
            assert member.size == entries[name]["bytes"]
            assert hashlib.file_digest(handle.extractfile(member),"sha256").hexdigest() == entries[name]["sha256"]
            seen.add(name)
    assert seen == set(entries)
    receipt["all_member_hashes_verified_from_durable_archive"] = True
    (durable.parent/"gears_rpe1_primary_v2_reproducibility_patch_v3_checksums.json").write_text(
        json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


if __name__ == "__main__":
    package()
