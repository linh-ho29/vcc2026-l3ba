"""Validate recovery without optimizer updates or measured held-out responses."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import pandas as pd
import torch

from common import digest, timestamp, write_json
from training.checkpoint_utils import load_checkpoint
from training.train import make_training


def validate(checkpoint_root, prepared, output, local_checkpoint_root=None,
             previous_background_status="Terminated by platform daily credit limit; not paused or active."):
    root, prepared = Path(checkpoint_root), Path(prepared)
    verified = []
    selected = None
    selected_path = None
    for path in sorted(root.glob("*.pt")):
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            raise ValueError(f"Checkpoint has no integrity sidecar: {path.name}")
        record = json.loads(sidecar.read_text())
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Checkpoint integrity failure: {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        verified.append(dict(filename=path.name, epoch=state["epoch"],
                             cursor=state.get("cursor", 0),
                             global_step=state["global_step"],
                             bytes=path.stat().st_size, sha256=record["sha256"]))
        if selected is None or state["global_step"] > selected["global_step"]:
            selected, selected_path = state, path
    if selected is None:
        raise ValueError("No checkpoint available")
    meta, graph, roles, model, optimizer, scheduler, pairs, filters, ctrl, de, loss = make_training(
        prepared, selected["threads"])
    saved = selected["dataset"]
    for key, name in [
        ("source_sha256", meta["h5ad"]),
        ("split_sha256", prepared / "split.json"),
        ("cell_manifest_sha256", prepared / "cell_manifest.csv.gz"),
        ("gene_order_sha256", prepared / "gene_index.csv"),
        ("target_order_sha256", prepared / "target_index.csv"),
        ("graph_sha256", prepared / "graphs.pt"),
    ]:
        assert digest(name) == saved[key] == meta[key], key
    assert saved["gene_ids"] == meta["gene_ids"]
    assert saved["gene_symbols"] == meta["gene_symbols"]
    assert saved["target_symbols"] == meta["target_symbols"]
    assert saved["frozen_split"] == meta["frozen_split"]
    for key, value in saved["cell_role_rows"].items():
        np.testing.assert_array_equal(value, roles[key])
    for key, value in graph["model_config"].items():
        other = selected["graph"]["model_config"][key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    assert selected["batch_size"] == 8
    assert graph["model_config"]["hidden_size"] == 32
    assert selected["threads"] == 4
    loaded = load_checkpoint(selected_path, model, optimizer, scheduler)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    for parameter, entry in optimizer.state.items():
        for key in ["exp_avg", "exp_avg_sq"]:
            assert entry[key].shape == parameter.shape
            assert torch.isfinite(entry[key]).all()
        assert 0 < float(entry["step"]) <= loaded["global_step"]
    assert scheduler.last_epoch == loaded["epoch"]
    expected_lr = 0.001 * 0.5 ** loaded["epoch"]
    assert all(group["lr"] == expected_lr for group in optimizer.param_groups)
    assert random.getstate() == loaded["rng"]["python"]
    actual_numpy = np.random.get_state()
    for a, b in zip(actual_numpy, loaded["rng"]["numpy"]):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    assert torch.equal(torch.get_rng_state(), loaded["rng"]["torch"])
    generator = torch.Generator()
    generator.set_state(loaded["permutation_rng"])
    assert torch.equal(generator.get_state(), loaded["permutation_rng"])
    permutation = loaded["permutation"]
    cursor = loaded["cursor"]
    batches = len(roles["fit"]) // loaded["batch_size"]
    assert len(np.unique(permutation)) == len(roles["fit"])
    np.testing.assert_array_equal(np.sort(permutation), np.sort(roles["fit"]))
    assert 0 <= cursor <= batches
    assert loaded["global_step"] == loaded["epoch"] * batches + cursor
    # Validate restored immutable memory-map arrays against their HDF5 source.
    with h5py.File(meta["h5ad"], "r") as handle:
        for key in ["data", "indices", "indptr"]:
            cached = np.load(Path(meta["matrix_cache"]) / f"X_{key}.npy", mmap_mode="r")
            source = handle[f"X/{key}"]
            assert cached.shape == source.shape and cached.dtype == source.dtype
            sample = np.unique(np.linspace(0, len(source)-1, 256, dtype=np.int64))
            np.testing.assert_array_equal(cached[sample], source[sample])
    if local_checkpoint_root:
        local = Path(local_checkpoint_root) / selected_path.name
        assert local.exists() and digest(local) == digest(selected_path)
    log = Path(meta["h5ad"]).parents[2] / "unused.jsonl"
    if local_checkpoint_root:
        log = Path(local_checkpoint_root).parent / "training_log.jsonl"
    entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    latest_attempt = entries[-1].get("attempt_id") if entries else None
    recent = [r for r in entries if r.get("attempt_id") == latest_attempt][-30:]
    median_seconds = float(np.median([r["step_seconds"] for r in recent])) if recent else None
    latest_logged = max((r["global_step"] for r in entries), default=loaded["global_step"])
    last_logged_elapsed = max((r["elapsed_training_seconds"] for r in entries),
                              default=loaded["elapsed_training_seconds"])
    remaining = 8 * batches - loaded["global_step"]
    validation_seconds = float(np.mean([r["validation_seconds"] for r in loaded["history"][-2:]]))
    report = dict(
        verified_at=timestamp(), status="valid_recovery_no_training_updates_performed",
        checkpoint=str(selected_path), checkpoint_sha256=digest(selected_path),
        verified_checkpoint_files=verified, completed_epochs=loaded["epoch"],
        current_epoch=loaded["epoch"]+1, completed_batches_in_current_epoch=cursor,
        next_batch_number_1_based=cursor+1, batches_per_epoch=batches,
        global_step=loaded["global_step"], remaining_training_batches=remaining,
        model_weights_finite=True, optimizer_state_valid=True, scheduler_valid=True,
        python_numpy_torch_rng_restored=True, data_loader_permutation_and_rng_valid=True,
        dataset_gene_target_split_and_graph_identity_valid=True,
        restored_matrix_cache_samples_match_source=True,
        learning_rate=expected_lr, training_threads=loaded["threads"],
        source_training_log_latest_step=latest_logged,
        uncheckpointed_logged_steps_requiring_replay=max(0, latest_logged-loaded["global_step"]),
        interrupted_attempt_uncheckpointed_elapsed_lower_bound_seconds=max(
            0., last_logged_elapsed-loaded["elapsed_training_seconds"]),
        recent_median_batch_seconds=median_seconds,
        estimated_remaining_training_and_validation_seconds=(
            remaining*median_seconds+(8-loaded["epoch"])*validation_seconds
        ) if median_seconds is not None else None,
        allowance_status="Worker restored and responsive; no direct credit-balance endpoint available. User reports allowance reset.",
        previous_background_status=previous_background_status,
        paid_resources_requested=False,
    )
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-checkpoint-root")
    parser.add_argument("--previous-background-status", default="Terminated by platform daily credit limit; not paused or active.")
    args = parser.parse_args()
    validate(args.checkpoint_root, args.prepared, args.output, args.local_checkpoint_root,
             args.previous_background_status)
