"""Exact-preset CPU metrics with per-perturbation official DE precomputation.

Per-perturbation precomputation is valid only with the frozen per_pert FDR rule
and fixed per-cell normalization. A measured small-panel parity test is required
before full scoring. No metric is reimplemented here.
"""
import argparse
import gc
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import cell_eval2
import numpy as np
import pandas as pd
import polars as pl
import psutil

from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
from cell_eval2.de_compute import compute_de
from cell_eval2.run import metric_cohorts, metric_output_names
from data.csr_rows import CSRRowReader
from io_utils import digest, timestamp, write_json, write_progress_snapshot

EVALUATOR_REVISION = "d86be3f4cd47015cd41046e9cc6a834777453594"


class ChunkPaused(RuntimeError):
    """The bounded chunk completed safely between targets; caches remain reusable."""


def verify_evaluator_revision():
    source_root = Path(cell_eval2.__file__).resolve().parents[2]
    if not (source_root/".git").exists():
        vendor = Path(__file__).resolve().parents[1]/"vendor/cell-eval2"
        source_root = vendor if (vendor/"SOURCE_COMMIT").exists() else Path("/workspace/cell_eval2_source")
    if (source_root/".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(source_root), "diff", "--quiet", "--", "src"],
                       check=True)
    else:
        revision = (source_root/"SOURCE_COMMIT").read_text().strip()
    assert revision == EVALUATOR_REVISION
    installed = Path(cell_eval2.__file__).resolve().parent
    for path in (source_root/"src/cell_eval2").rglob("*.py"):
        relative = path.relative_to(source_root/"src/cell_eval2")
        assert digest(path) == digest(installed/relative), str(relative)


def config():
    cfg = EvalConfig.from_preset("vcc2026")
    return replace(cfg, device="cpu", num_threads=4,
                   de=replace(cfg.de, backend="scanpy"))


def compute_one(perturbed, controls, cfg):
    combined = ad.concat([perturbed, controls], axis=0, join="inner", merge="same")
    combined.obs[cfg.pert_col] = pd.Categorical(combined.obs[cfg.pert_col])
    return compute_de(
        combined, backend=cfg.de.backend, groupby=cfg.pert_col, reference=cfg.control,
        mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, input_type=cfg.input_type,
        target_sum=cfg.target_sum, clip_value=cfg.de.clip_value,
        filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
        fdr_scope=cfg.de.fdr_scope, threads=cfg.num_threads, device="cpu")


def precompute(source, controls, cfg, cache, checkpoint=None, deadline=None, progress=None):
    if cfg.de.fdr_scope != "per_pert" or cfg.target_sum is None:
        raise ValueError("Independent DE precomputation would change this configuration")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if checkpoint:
        checkpoint = Path(checkpoint)/source_hash[:20]
        checkpoint.mkdir(parents=True, exist_ok=True)
    control_hash = hashlib.sha256("\n".join(controls.obs_names).encode()).hexdigest()
    identity = dict(source_sha256=source_hash, control_cell_ids_sha256=control_hash,
                    config=asdict(cfg))
    if checkpoint:
        durable_identity = checkpoint/"identity.json"
        if durable_identity.exists():
            if json.loads(durable_identity.read_text()) != identity:
                raise ValueError("Durable DE cache configuration or control identity differs")
        else:
            write_json(durable_identity, identity)
    identity_path = cache/"identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        write_json(identity_path, identity)
    provenance = dict(**identity, evaluator_revision=EVALUATOR_REVISION,
                      evaluator_version="0.20.0", protocol="per-target official DE v2")
    for folder in [cache] + ([checkpoint] if checkpoint else []):
        provenance_path = folder/"verified_identity_v2.json"
        if provenance_path.exists():
            if json.loads(provenance_path.read_text()) != provenance:
                raise ValueError("Verified DE provenance differs")
        else:
            write_json(provenance_path, provenance)
    backed = ad.read_h5ad(source, backed="r")
    row_reader = CSRRowReader(source)
    frames = []
    labels = backed.obs[cfg.pert_col].astype(str).to_numpy()
    targets = sorted(set(labels)-{cfg.control})
    started = time.monotonic()
    reused = created = restored = 0

    def report(status, processed):
        record = dict(updated_at=timestamp(), status=status, source=Path(source).name,
                      source_sha256=source_hash, total_targets=len(targets),
                      processed=processed, reused=reused, created=created, restored=restored,
                      remaining=len(targets)-processed, seconds=time.monotonic()-started,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        if progress:
            write_progress_snapshot(str(progress)+"_"+source_hash[:20], record)
        print("CPU_DE_PROGRESS", json.dumps(record), flush=True)

    try:
        for i,target in enumerate(targets):
            if deadline is not None and time.monotonic() >= deadline:
                report("paused_at_chunk_boundary", i)
                raise ChunkPaused(f"Time-bounded pause after {i}/{len(targets)} targets")
            if psutil.Process().memory_info().rss > 12*1024**3:
                raise MemoryError("12 GiB DE-stage RSS safety ceiling reached")
            name = hashlib.sha256(target.encode()).hexdigest()[:20] + ".parquet"
            path = cache/name
            sidecar = path.with_suffix(".json")
            durable = checkpoint/name if checkpoint else None
            if not path.exists() and durable and durable.exists():
                if durable.with_suffix(".json").exists():
                    temporary = path.with_suffix(".restore.parquet")
                    shutil.copyfile(durable, temporary)
                    record = json.loads(durable.with_suffix(".json").read_text())
                    assert digest(temporary) == record["sha256"]
                    temporary.replace(path)
                    shutil.copyfile(durable.with_suffix(".json"), sidecar)
                    restored += 1
            if path.exists() and sidecar.exists():
                record = json.loads(sidecar.read_text())
                if record["target"] != target or record["source_sha256"] != source_hash or digest(path) != record["sha256"]:
                    raise ValueError(f"Invalid cached DE table for {target}")
                frame = pl.read_parquet(path)
                reused += 1
            else:
                # A table without its commit sidecar was not completed. Preserve it.
                if path.exists():
                    path.rename(path.with_suffix(".incomplete_"+str(time.time_ns())+".parquet"))
                perturbed = row_reader.read(np.flatnonzero(labels==target),
                                            backed.obs, backed.var)
                frame = compute_one(perturbed, controls, cfg)
                del perturbed
                temporary = path.with_suffix(".partial.parquet")
                frame.write_parquet(temporary)
                temporary.replace(path)
                sidecar_temp = sidecar.with_suffix(".partial.json")
                write_json(sidecar_temp, dict(target=target, source_sha256=source_hash,
                                              sha256=digest(path), rows=frame.height))
                sidecar_temp.replace(sidecar)
                created += 1
            if durable:
                expected_hash = json.loads(sidecar.read_text())["sha256"]
                if not durable.exists():
                    shutil.copyfile(path, durable)
                elif digest(durable) != expected_hash:
                    raise ValueError("Durable DE object differs; preserving both copies")
                if not durable.with_suffix(".json").exists():
                    shutil.copyfile(sidecar, durable.with_suffix(".json"))
                else:
                    assert json.loads(durable.with_suffix(".json").read_text()) == json.loads(sidecar.read_text())
                assert digest(durable) == expected_hash
            frames.append(frame)
            if i % 20 == 0:
                print("CPU_DE_TARGET", Path(source).name, i, len(targets),
                      f"{time.monotonic()-started:.1f}s", flush=True)
                report("running", i+1)
            gc.collect()
    finally:
        row_reader.close()
        backed.file.close()
    report("complete", len(targets))
    return pl.concat(frames, how="vertical")


def evaluate(predicted, real, output, cache, checkpoint=None, parity_pilot=False,
             max_minutes=None, progress=None, groupwise_bulk=False):
    started = time.monotonic()
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    cfg = config()
    deadline = started + max_minutes*60 if max_minutes else None
    verify_evaluator_revision()
    real_backed = ad.read_h5ad(real, backed="r")
    try:
        control_rows = real_backed.obs[cfg.pert_col].astype(str).to_numpy() == cfg.control
        controls = real_backed[control_rows].to_memory()
    finally:
        real_backed.file.close()
    de_real = precompute(real, controls, cfg, cache.parent/"real_shared"/digest(real)[:20],
                         Path(checkpoint)/"real" if checkpoint else None,
                         deadline=deadline, progress=progress)
    de_pred = precompute(predicted, controls, cfg, cache/"pred",
                         Path(checkpoint)/"pred" if checkpoint else None,
                         deadline=deadline, progress=progress)
    # Per-target DE is now small table input; the heavy count-space pseudobulk
    # materializes only one complete sparse side at a time in the official API.
    if groupwise_bulk:
        from evaluation.official_groupwise_bulk import compute_groupwise_metrics
        result = compute_groupwise_metrics(predicted, real, cfg, de_pred, de_real,
                                           cache/"official_groupwise_bulk")
    else:
        result = compute_metrics(str(predicted), str(real), config=cfg,
                                 de_real=de_real, de_pred=de_pred)
    if parity_pilot:
        ordinary = compute_metrics(str(predicted), str(real), config=cfg)
        left = result.sort(["perturbation","metric"])
        right = ordinary.sort(["perturbation","metric"])
        assert left.select(["perturbation","metric"]).equals(right.select(["perturbation","metric"]))
        np.testing.assert_allclose(left["value"].to_numpy(), right["value"].to_numpy(),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)
        write_json(output/"cpu_precomputation_parity.json",
                   dict(passed=True, comparison="ordinary official compute_metrics versus per-target official DE precomputation",
                        tolerance_absolute=1e-10, tolerance_relative=1e-10,
                        predicted_sha256=digest(predicted), real_sha256=digest(real)))
    result.write_csv(output/"official_implementation_per_target_metrics.csv")
    # Include requested derived metrics that have no tidy per-target rows.
    selected_metrics = metric_output_names(cfg)
    aggregate_metrics(result,metrics=selected_metrics).join(
        metric_cohorts(result,metrics=selected_metrics), on="metric", how="left").write_csv(
        output/"official_implementation_metric_summary.csv")
    write_json(output/"official_implementation_manifest.json", dict(
        completed_at=timestamp(), config=asdict(cfg),
        predicted_sha256=digest(predicted), real_sha256=digest(real),
        dataset="RPE1 GEARS restricted 5000-gene panel",
        prediction_layer="GEARS or baseline plus heuristic count reconstruction",
        raw_reference_measured=True, challenge_leaderboard_result=False,
        calibrated_aggregate_available=False,
        reason_no_calibrated_aggregate="No compatible official RPE1 calibration bundle",
        de_precomputation="Official compute_de per perturbation, per_pert FDR; pilot parity required",
        bulk_orchestration="group-wise official functions; full-panel dispatch" if groupwise_bulk else "ordinary official compute_metrics",
        pilot=parity_pilot, seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        memory_available_after_bytes=psutil.virtual_memory().available))
    print("OFFICIAL_CPU_COUNT_METRICS_COMPLETE", str(output),
          f"{time.monotonic()-started:.1f}s", flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["predicted","real","output","cache"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--parity-pilot",action="store_true")
    parser.add_argument("--max-minutes",type=float)
    parser.add_argument("--progress")
    parser.add_argument("--groupwise-bulk",action="store_true")
    args=parser.parse_args()
    evaluate(args.predicted,args.real,args.output,args.cache,args.checkpoint,args.parity_pilot,
             max_minutes=args.max_minutes, progress=args.progress, groupwise_bulk=args.groupwise_bulk)
