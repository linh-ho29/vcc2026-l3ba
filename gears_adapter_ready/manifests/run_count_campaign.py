"""Sequential, time-bounded remaining count evaluation using existing CPU only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "4"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psutil
from count_generation.export_sampling_provenance import export as export_provenance
from evaluation.evaluate_counts_bounded_cpu import ChunkPaused, evaluate
from io_utils import digest, timestamp, write_json

CANONICAL = [
    "pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard"]
METHODS = ["gears_epoch_8", "statistical_mean_effect", "control_only",
           "gears_epoch_2", "gears_epoch_4", "gears_epoch_6"]


def verify_receipt(folder):
    receipt = json.loads((folder/"completion_receipt.json").read_text())
    for name, record in receipt["files"].items():
        assert (folder/name).stat().st_size == record["bytes"]
        assert digest(folder/name) == record["sha256"]
    return receipt


def publish(source, target):
    if target.exists():
        assert digest(source) == digest(target), str(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source,target)
    assert target.stat().st_size == source.stat().st_size
    assert digest(source) == digest(target)


def run(project, data_root, output_root, checkpoint, statistics, train_python, max_minutes):
    project, data_root = Path(project), Path(data_root)
    output_root, checkpoint = Path(output_root), Path(checkpoint)
    root = Path(__file__).resolve().parents[1]
    data_root.mkdir(parents=True,exist_ok=True)
    lock_path = data_root/"full_unseen/official_evaluation.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt = str(time.time_ns())
    started = time.monotonic()
    deadline = started + max_minutes*60
    progress = checkpoint/"campaign_progress"/attempt
    progress.parent.mkdir(parents=True,exist_ok=True)
    completed = []
    out = Path("/mnt/results/gears_recovery_20261104")
    out.mkdir(parents=True,exist_ok=True)

    def finish(status):
        receipt = dict(updated_at=timestamp(),status=status,seconds=time.monotonic()-started,
                       completed_or_reused_panels=completed,progress_prefix=str(progress),
                       method_order=METHODS,exposure_order=["unseen_test","seen_test"],
                       no_training=True,existing_cpu_only=True)
        write_json(out/("campaign_chunk_"+attempt+".json"),receipt)
        print("CAMPAIGN_CHUNK_END",json.dumps(receipt),flush=True)

    for exposure in ["unseen_test","seen_test"]:
        expected_targets = 384 if exposure == "unseen_test" else 1035
        base = data_root/("full_unseen" if exposure == "unseen_test" else "full_seen")
        base.mkdir(parents=True,exist_ok=True)
        for method in METHODS:
            destination = output_root/method/exposure
            if (destination/"completion_receipt.json").exists():
                verify_receipt(destination)
                completed.append(dict(method=method,exposure=exposure,reused=True))
                continue
            if time.monotonic() >= deadline-60:
                finish("paused_between_panels")
                return
            destination.mkdir(parents=True,exist_ok=True)
            pred_path = base/f"{method}_{exposure}.h5ad"
            pred_manifest = base/f"{method}_{exposure}_manifest.json"
            real_path = base/f"measured_{exposure}.h5ad"
            real_manifest = real_path.with_suffix(".json")
            epoch = int(method.rsplit("_",1)[1]) if method.startswith("gears_epoch_") else 8
            if pred_manifest.exists():
                record = json.loads(pred_manifest.read_text())
                assert digest(pred_path) == record["sha256"]
            else:
                if pred_path.exists():
                    raise ValueError("Uncommitted count file exists; preserve it and audit before export")
                if psutil.virtual_memory().available < 4*1024**3:
                    raise MemoryError("Insufficient free RAM for count export")
                subprocess.run([
                    train_python,str(root/"count_generation/export_count_panel.py"),
                    "--matched",str(data_root/"matched_measured_counts.h5ad"),
                    "--profiles",str(project/"predictions"/f"epoch_{epoch:02d}_profiles.npz"),
                    "--statistics",str(statistics),"--method",method,"--exposure",exposure,
                    "--output",str(base)],check=True)
                record = json.loads(pred_manifest.read_text())
            assert len(record["conditions"]) == expected_targets
            assert record["cells_per_target"] == 400 and record["background_fraction"] == 1e-4
            if not real_manifest.exists():
                subprocess.run([
                    sys.executable,str(root/"data/export_count_references.py"),
                    "--matched",str(data_root/"matched_measured_counts.h5ad"),
                    "--exposure",exposure,"--output",str(real_path)],check=True)
            real_record = json.loads(real_manifest.read_text())
            assert digest(real_path) == real_record["sha256"]
            assert real_record["unique_targets"] == expected_targets
            assert real_record["independent_reference_controls"] == 2297
            for path in [pred_path,pred_manifest,real_path,real_manifest]:
                publish(path,checkpoint/"count_files"/exposure/path.name)
            # Enrich provenance without changing any count file.
            sampling = base/(method+"_"+exposure+"_sampling.npz")
            sampling_receipt = export_provenance(
                data_root/"matched_measured_counts.h5ad",pred_path,pred_manifest,sampling)
            publish(sampling,destination/sampling.name)
            publish(sampling.with_suffix(".json"),destination/sampling.with_suffix(".json").name)
            available = psutil.virtual_memory().available
            resident = record["sparse_resident_bytes_estimate"] + real_record["sparse_resident_bytes_estimate"]
            budget = 3*resident + 2*1024**3
            groupwise = available < budget*1.15 or budget > 12*1024**3
            if groupwise:
                parity = json.loads((out/"official_groupwise_bulk_eight_target_parity.json").read_text())
                assert parity["passed"] and parity["equal_to_unmodified_official_api"]
                if available < 4*1024**3:
                    raise MemoryError("Insufficient headroom for official group-wise scoring")
            write_json(destination/("resource_preflight_"+attempt+".json"),
                       dict(available_bytes=available,conservative_resident_budget_bytes=budget,
                            groupwise_bulk=groupwise,threads=4,device="cpu",
                            input_count_sha256=record["sha256"]))
            remaining_minutes = (deadline-time.monotonic())/60
            if remaining_minutes < 1:
                finish("paused_after_count_export")
                return
            local_output = base/"evaluation"/method
            try:
                result = evaluate(
                    pred_path,real_path,local_output,base/"cache"/method,
                    checkpoint=checkpoint,max_minutes=remaining_minutes,
                    progress=progress,groupwise_bulk=groupwise)
            except ChunkPaused:
                finish("paused_between_de_targets")
                return
            import polars as pl
            summary = pl.read_csv(local_output/"official_implementation_metric_summary.csv")
            assert set(CANONICAL).issubset(set(summary["metric"].to_list()))
            assert result["perturbation"].n_unique() == expected_targets
            for path in local_output.iterdir():
                if path.is_file():
                    publish(path,destination/path.name)
            for path in [pred_manifest,real_manifest]:
                publish(path,destination/path.name)
            write_json(destination/"completion_receipt.json",dict(
                completed_at=timestamp(),method=method,exposure=exposure,
                targets=expected_targets,canonical_metrics=CANONICAL,
                calibrated_aggregate_available=False,challenge_leaderboard_result=False,
                files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
                       for p in destination.iterdir() if p.is_file()}))
            completed.append(dict(method=method,exposure=exposure,reused=False))
            print("CAMPAIGN_PANEL_COMPLETE",method,exposure,flush=True)
    finish("all_twelve_panels_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",default="/mnt/results/gears_adapter_ready")
    parser.add_argument("--data-root",default="/workspace/gears_raw_eval")
    parser.add_argument("--output-root",default="/mnt/results/gears_adapter_ready/count_evaluation_v2")
    parser.add_argument("--checkpoint",default="/mnt/shared-workspace/shared/gears_rpe1_official_de_cache")
    parser.add_argument("--statistics",default="/workspace/gears_run/prepared/training_statistics.npz")
    parser.add_argument("--train-python",default="/workspace/gears_env/bin/python")
    parser.add_argument("--max-minutes",type=float,default=45)
    args = parser.parse_args()
    run(args.project,args.data_root,args.output_root,args.checkpoint,args.statistics,
        args.train_python,args.max_minutes)
