"""Audit a completed panel, checkpoint readback, and target-transcript coverage."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anndata as ad
import h5py
import pandas as pd
import polars as pl
from evaluation.evaluate_counts_bounded_cpu import config
from io_utils import digest, timestamp, write_json

CANONICAL = [
    "pds_cosine","expr_mse_unbiased_capped_norm","de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw","de_wilcoxon_sig_jaccard"]


def audit(panel, predicted, real, checkpoint, output):
    panel, output, checkpoint = Path(panel), Path(output), Path(checkpoint)
    output.mkdir(parents=True,exist_ok=True)
    completion = json.loads((panel/"completion_receipt.json").read_text())
    for name, record in completion["files"].items():
        path = panel/name
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
    manifest = json.loads((panel/"official_implementation_manifest.json").read_text())
    assert manifest["config"] == asdict(config())
    assert not manifest["challenge_leaderboard_result"] and not manifest["calibrated_aggregate_available"]
    assert digest(predicted) == manifest["predicted_sha256"]
    assert digest(real) == manifest["real_sha256"]
    measured = ad.read_h5ad(real,backed="r")
    try:
        targets = set(measured.obs.target.astype(str))-{"non-targeting"}
        genes = set(measured.var_names)
        var = measured.var.copy()
        controls = measured.obs_names[measured.obs.target.astype(str)=="non-targeting"]
        control_hash = hashlib.sha256("\n".join(controls).encode()).hexdigest()
    finally:
        measured.file.close()
    summary = pl.read_csv(panel/"official_implementation_metric_summary.csv")
    per_target = pl.read_csv(panel/"official_implementation_per_target_metrics.csv")
    assert set(CANONICAL).issubset(set(summary["metric"].to_list()))
    assert set(per_target["perturbation"].to_list()) == targets
    checkpoint_records = []
    progress_path = output/"checkpoint_audit_progress.json"
    audit_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for side, source_hash in [("real",manifest["real_sha256"]),("pred",manifest["predicted_sha256"])]:
        directory = checkpoint/side/source_hash[:20]
        identity = json.loads((directory/"identity.json").read_text())
        assert identity == dict(source_sha256=source_hash,control_cell_ids_sha256=control_hash,
                                config=asdict(config()))
        found = set()
        names = {p.name for p in directory.iterdir()}
        for path in sorted(directory.glob("*.parquet")):
            if path.with_suffix(".json").name not in names:
                continue
            record = json.loads(path.with_suffix(".json").read_text())
            assert record["source_sha256"] == source_hash and record["target"] in targets
            key = source_hash+":"+path.name
            if key not in audit_progress:
                # One FUSE read; parsing from RAM avoids many remote range requests.
                payload = path.read_bytes()
                assert hashlib.sha256(payload).hexdigest() == record["sha256"]
                frame = pl.read_parquet(io.BytesIO(payload))
                assert frame.height == record["rows"]
                assert set(frame["target"].to_list()) == {record["target"]}
                assert set(frame["feature"].to_list()).issubset(genes)
                audit_progress[key] = record
            else:
                assert audit_progress[key] == record
            found.add(record["target"])
            if len(found)%20 == 0:
                write_json(progress_path,audit_progress)
                print("AUDIT_VERIFIED_TABLES",side,len(found),len(targets),flush=True)
        assert found == targets
        write_json(progress_path,audit_progress)
        checkpoint_records.append(dict(side=side,verified_tables=len(found),source_sha256=source_hash))
    with h5py.File("/workspace/gears_raw_counts/rpe1_raw_singlecell_01.h5ad","r") as handle:
        raw_var = ad.io.read_elem(handle["var"])
    # Independent ID-based check: processed symbols agree with the measured source.
    assert (raw_var.loc[var["gene_id"],"gene_name"].to_numpy() == var.index.to_numpy()).all()
    retained_ids = set(var["gene_id"])
    rows = []
    for target in sorted(targets):
        ids = raw_var.index[raw_var.gene_name.astype(str)==target].tolist()
        direct = target in genes
        present_by_id = bool(set(ids)&retained_ids)
        assert not (present_by_id and not direct), "Alias/ID conflict affects target exclusion"
        rows.append(dict(target=target,transcript_in_response_panel=direct,
                         raw_gene_ids=";".join(ids),raw_gene_id_in_response_panel=present_by_id,
                         symbol_present_in_raw_8749_gene_panel=bool(ids)))
    table = pd.DataFrame(rows)
    table.to_csv(output/"target_transcript_coverage.csv",index=False)
    result = dict(verified_at=timestamp(),passed=True,targets=len(targets),
                  canonical_metrics=summary.filter(pl.col("metric").is_in(CANONICAL)).to_dicts(),
                  checkpoint_readback=checkpoint_records,
                  target_transcripts_in_response_panel=int(table.transcript_in_response_panel.sum()),
                  target_transcripts_not_in_response_panel=int((~table.transcript_in_response_panel).sum()),
                  targets_absent_also_from_raw_8749_gene_symbol_panel=int((~table.symbol_present_in_raw_8749_gene_panel).sum()),
                  gene_id_symbol_conflicts=0,
                  target_warning_interpretation="Restricted response-panel coverage; no retained target transcript found under an alternate source gene ID",
                  official_implementation_seconds=manifest["seconds"],
                  peak_rss_gib=manifest["peak_rss_gib"],
                  calibrated_aggregate_available=False,challenge_leaderboard_result=False)
    write_json(output/"count_panel_integrity_audit.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ["panel","predicted","real","checkpoint","output"]:
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    audit(args.panel,args.predicted,args.real,args.checkpoint,args.output)
