"""Verify the standalone trained-project snapshot without training or inference."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_review_bundle import verify_arrays


def main(root):
    manifest = json.loads((root/"trained_handoff_manifest.json").read_text())
    assert manifest["completed_epochs"] == 8
    assert not manifest["adapter_trained"] and not manifest["complete_count_evaluation"]
    for name, record in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = root/relative
        assert path.is_file() and path.stat().st_size == record["bytes"], name
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"], name
    checkpoint_receipt = json.loads(
        (root/"reports/completed_checkpoint_verification.json").read_text())
    assert [r["epoch"] for r in checkpoint_receipt["checkpoints"]] == list(range(1, 9))
    for record in checkpoint_receipt["checkpoints"]:
        name = f"checkpoints/epoch_{record['epoch']:02d}.pt"
        assert manifest["files"][name]["sha256"] == record["sha256"]
    schemas = verify_arrays(root)
    result = dict(passed=True, hashed_files=len(manifest["files"]),
                  completed_epochs=8, array_schemas=schemas, no_training=True,
                  no_model_inference=True, no_checkpoint_deserialization=True,
                  count_evaluation_independent=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root.resolve())
