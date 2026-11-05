"""One locked, checkpointed continuation, followed by frozen exports/evaluation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from common import ROOT, digest, publish, timestamp, write_json


def verified_states(directory):
    candidates = []
    for path in sorted(Path(directory).glob("*.pt")):
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if path.stat().st_size != sidecar["bytes"] or digest(path) != sidecar["sha256"]:
            raise ValueError(f"Checkpoint checksum failure: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not all(torch.isfinite(v).all() for v in state["model"].values()):
            raise ValueError(f"Nonfinite model state: {path}")
        candidates.append((state["global_step"], path, state))
    if not candidates:
        raise ValueError(f"No checkpoint in {directory}")
    return candidates


def run(args):
    run_dir, prepared, output = map(Path, [args.run_dir, args.prepared, args.output])
    publish_root, resume_root = map(Path, [args.publish_root, args.resume_root])
    env = dict(os.environ, PYTHONHASHSEED="42", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (run_dir / "active_training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = verified_states(Path(args.resume).parent)
        original = next(s for _, p, s in initial if p == Path(args.resume))
        resume = Path(args.resume)
        for chunk in range(1, 7):
            subprocess.run([
                sys.executable, str(ROOT / "training/train.py"), "train",
                "--prepared", str(prepared), "--out", str(run_dir),
                "--threads", "4", "--batch-size", "8", "--resume", str(resume),
                "--publish-dir", str(resume_root / "checkpoints"),
                "--chunk-minutes", "100", "--checkpoint-seconds", "120",
                "--attempt-id", f"writable_resume_20261103_chunk{chunk}",
            ], env=env, check=True)
            states = verified_states(run_dir / "checkpoints")
            _, resume, latest = max(states, key=lambda item: item[0])
            for key in ["source_sha256", "split_sha256", "cell_manifest_sha256",
                        "fit_row_sha256", "gene_order_sha256", "target_order_sha256", "graph_sha256"]:
                if latest["dataset"][key] != original["dataset"][key]:
                    raise ValueError(f"Changed immutable training identity: {key}")
            if latest["epoch"] == 8:
                break
        else:
            raise RuntimeError("Continuation chunk limit reached; no automatic restart")
        expected = [run_dir / "checkpoints" / f"epoch_{i:02d}.pt" for i in range(1, 9)]
        if not all(p.exists() for p in expected):
            raise ValueError("Missing completed epoch checkpoint")
        verified = []
        for epoch, path in enumerate(expected, 1):
            durable = (publish_root / "checkpoints" if epoch <= 6 else resume_root / "checkpoints") / path.name
            sidecar = json.loads(durable.with_suffix(".json").read_text())
            if digest(durable) != digest(path) or digest(durable) != sidecar["sha256"]:
                raise ValueError(f"Durable checkpoint mismatch: {epoch}")
            verified.append(dict(epoch=epoch, filename=str(durable),
                                 sha256=sidecar["sha256"], bytes=durable.stat().st_size))
        write_json(resume_root / "reports/completed_checkpoint_verification.json",
                   dict(verified_at=timestamp(), completed_epochs=8, checkpoints=verified))
        subprocess.run([
            sys.executable, str(ROOT / "manifests/post_training.py"),
            "--run-dir", str(run_dir), "--prepared", str(prepared),
            "--output", str(output), "--publish-root", str(publish_root),
        ], env=env, check=True)
        subprocess.run([
            sys.executable, str(ROOT / "evaluation/evaluate_gears.py"),
            "--prepared", str(prepared), "--prediction-dir", str(output / "predictions"),
            "--output", str(output / "evaluation"),
            "--publish-dir", str(publish_root / "evaluation"),
        ], env=env, check=True)
        for name in ["evaluation_settings_frozen.json", "primary_exports_complete.json"]:
            publish(output / name, resume_root / "reports" / name)
        write_json(resume_root / "reports/continuation_complete.json",
                   dict(completed_at=timestamp(), completed_epochs=8,
                        primary_exports=True, normalized_evaluation=True,
                        adapter_trained=False, official_count_evaluation=False,
                        github_uploaded=False, paid_compute_requested=False))
        print("RESUMED_TRAINING_EXPORTS_NORMALIZED_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["run-dir", "prepared", "output", "publish-root", "resume-root", "resume"]:
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
