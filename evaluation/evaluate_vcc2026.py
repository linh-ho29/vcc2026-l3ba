"""Official metric implementations on explicitly identified local raw references."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from io_utils import write_json


def evaluate(predicted,real,output,pert_col="target"):
    # Install pinned cell-eval2 in its own environment; do not alter GEARS training.
    from cell_eval2 import EvalConfig,compute_metrics
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cfg=EvalConfig.from_preset("vcc2026")
    cfg=replace(cfg,pert_col=pert_col,device="cpu",num_threads=4,
                de=replace(cfg.de,backend="scanpy"))
    result=compute_metrics(str(predicted),str(real),config=cfg)
    result.write_csv(output/"official_implementation_metrics.csv")
    write_json(output/"official_implementation_manifest.json",dict(
        config=asdict(cfg),dataset="RPE1 restricted GEARS response-gene panel",
        challenge_leaderboard_result=False,
        predicted_data="heuristic reconstructed single-cell integer counts",
        reference_data="matched measured RPE1 raw single-cell counts",
        calibrated_aggregate_available=False,
        calibrated_aggregate_reason="No matching official calibrated RPE1 reference bundle; challenge-context anchors are not transferable.",
    ))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--predicted",required=True)
    p.add_argument("--real",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--pert-col",default="target")
    a=p.parse_args()
    evaluate(a.predicted,a.real,a.output,a.pert_col)
