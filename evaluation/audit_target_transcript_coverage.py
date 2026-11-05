"""Check target exclusions by both source Ensembl IDs and canonical symbols."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anndata as ad
import h5py
import pandas as pd
from io_utils import timestamp, write_json


def audit(reference, raw_source, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    measured = ad.read_h5ad(reference,backed="r")
    try:
        targets = set(measured.obs.target.astype(str))-{"non-targeting"}
        var = measured.var.copy()
    finally:
        measured.file.close()
    with h5py.File(raw_source,"r") as handle:
        raw_var = ad.io.read_elem(handle["var"])
    assert (raw_var.loc[var.gene_id,"gene_name"].to_numpy() == var.index.to_numpy()).all()
    retained_ids, genes = set(var.gene_id),set(var.index)
    rows = []
    for target in sorted(targets):
        ids = raw_var.index[raw_var.gene_name.astype(str)==target].tolist()
        direct = target in genes
        by_id = bool(set(ids)&retained_ids)
        rows.append(dict(target=target,transcript_in_response_panel=direct,
                         raw_gene_ids=";".join(ids),raw_gene_id_in_response_panel=by_id,
                         alias_id_conflict=by_id and not direct,
                         symbol_present_in_raw_8749_gene_panel=bool(ids)))
    table = pd.DataFrame(rows)
    table.to_csv(output/"target_transcript_coverage.csv",index=False)
    result = dict(verified_at=timestamp(),targets=len(targets),
                  target_transcripts_in_response_panel=int(table.transcript_in_response_panel.sum()),
                  target_transcripts_not_in_response_panel=int((~table.transcript_in_response_panel).sum()),
                  targets_absent_also_from_raw_8749_gene_symbol_panel=int((~table.symbol_present_in_raw_8749_gene_panel).sum()),
                  gene_id_symbol_conflicts=int(table.alias_id_conflict.sum()),
                  response_gene_id_symbol_alignment_with_raw_source=True)
    write_json(output/"target_coverage_audit.json",result)
    assert result["gene_id_symbol_conflicts"] == 0, "Target-gene mapping conflict requires review"
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ["reference","raw-source","output"]:
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    audit(args.reference,args.raw_source,args.output)
