"""Prepare only missing seen count inputs; never train or rerun model inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from io_utils import digest, timestamp, write_json

METHODS = ["gears_epoch_8", "statistical_mean_effect", "control_only",
           "gears_epoch_2", "gears_epoch_4", "gears_epoch_6"]


def main(max_minutes):
    root = Path(__file__).resolve().parents[1]
    base = Path("/workspace/gears_raw_eval/full_seen")
    base.mkdir(parents=True, exist_ok=True)
    lock = (base/"count_preparation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared = Path("/mnt/shared-workspace/shared/gears_rpe1_official_de_cache/count_files/seen_test")
    shared.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max_minutes*60
    durations, prepared = [], []
    for method in METHODS:
        manifest = base/(method+"_seen_test_manifest.json")
        path = base/(method+"_seen_test.h5ad")
        if not manifest.exists():
            if path.exists():
                raise RuntimeError("Uncommitted existing count file; preserve and inspect")
            estimated = max(durations)*1.15 if durations else 15*60
            if deadline-time.monotonic() < estimated:
                break
            if psutil.virtual_memory().available < 8*1024**3:
                print("PREPARATION_PAUSED_MEMORY_HEADROOM", flush=True)
                break
            epoch = int(method.rsplit("_", 1)[1]) if method.startswith("gears_epoch_") else 8
            before = time.monotonic()
            subprocess.run([
                "/workspace/gears_env/bin/python",
                str(root/"count_generation/export_count_panel.py"),
                "--matched", "/workspace/gears_raw_eval/matched_measured_counts.h5ad",
                "--profiles", f"/mnt/results/gears_adapter_ready/predictions/epoch_{epoch:02d}_profiles.npz",
                "--statistics", "/workspace/gears_run/prepared/training_statistics.npz",
                "--method", method, "--exposure", "seen_test", "--output", str(base)
            ], check=True)
            durations.append(time.monotonic()-before)
        record = json.loads(manifest.read_text())
        assert len(record["conditions"]) == 1035 and record["cells"] == 414064
        assert digest(path) == record["sha256"]
        for source in [path, manifest]:
            destination = shared/source.name
            if destination.exists():
                assert digest(destination) == digest(source)
            else:
                shutil.copyfile(source, destination)
            assert digest(destination) == digest(source)
        prepared.append(dict(method=method, bytes=path.stat().st_size,
                             sha256=record["sha256"], manifest=manifest.name))
        print("SEEN_COUNT_INPUT_PREPARED", method, flush=True)
    receipt = dict(completed_at=timestamp(), seconds=time.monotonic()-started,
                   prepared=prepared, count_generation_seconds=durations,
                   no_training=True, no_model_inference=True, existing_cpu_only=True,
                   threads=4, separate_post_copy_readback_pending=True)
    write_json("/mnt/results/gears_recovery_20261104/seen_count_preparation_"+
               str(time.time_ns())+".json", receipt)
    print("SEEN_PREPARATION_CHUNK_END", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-minutes", type=float, default=35)
    main(parser.parse_args().max_minutes)
