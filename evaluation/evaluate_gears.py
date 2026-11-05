"""Final sealed-reference evaluation; normalized diagnostics, not VCC scores."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats

from common import load_prepared,write_json,publish

OFFICIAL_METRICS=[
    "pds_cosine","expr_mse_unbiased_capped_norm","de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw","de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
]


def reference_statistics(x,rows):
    matrix=x[rows]
    mean=np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
    second=np.asarray(matrix.power(2).mean(axis=0)).ravel().astype(np.float64)
    variance=np.maximum(second-mean**2,0)*len(rows)/max(len(rows)-1,1)
    return mean,np.sqrt(variance),len(rows)


def safe_correlation(pred,true,rank=False):
    if np.std(pred)<1e-12 or np.std(true)<1e-12:
        return None
    result=stats.spearmanr(pred,true).statistic if rank else stats.pearsonr(pred,true).statistic
    return float(result) if np.isfinite(result) else None


def evaluate(prepared,prediction_dir,output,publish_dir=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    meta,obs,var,roles,graph,x=load_prepared(prepared)
    conditions=obs.condition.astype(str).to_numpy()
    ctrl_mean,ctrl_std,nctrl=reference_statistics(x,roles["reference_control"])
    profiles={epoch:np.load(Path(prediction_dir)/f"epoch_{epoch:02d}_profiles.npz",
                            allow_pickle=False) for epoch in [2,4,6,8]}
    assert all(np.array_equal(profiles[2]["conditions"],p["conditions"]) for p in profiles.values())
    common_conditions=profiles[2]["conditions"]
    statistical=np.load(Path(prepared)/"training_statistics.npz")["statistical_effect"]
    input_mean=profiles[2]["control_expression_mean"]
    target_records=[]
    for i,condition in enumerate(common_conditions):
        role="unseen_test" if np.any(conditions[roles["unseen_test"]]==condition) else "seen_test"
        rows=np.flatnonzero((conditions==condition)&(roles["roles"]==role))
        if not len(rows):
            continue
        true_mean,true_std,n=reference_statistics(x,rows)
        true_delta=true_mean-ctrl_mean
        # Supplemental rank diagnostic: Welch t statistic on model-normalized inputs.
        test=stats.ttest_ind_from_stats(
            true_mean,true_std,n,ctrl_mean,ctrl_std,nctrl,equal_var=False)
        scores=np.nan_to_num(np.abs(test.statistic),nan=0.,posinf=1e30,neginf=1e30)
        top=np.argsort(scores)[-20:]
        own=condition.replace("+ctrl","")
        eligible=np.array([j for j in top if var.gene_name.iloc[j] != own],dtype=int)
        predicted={
            "control_only":input_mean,
            "statistical_mean_effect":input_mean+statistical,
            **{f"gears_epoch_{epoch}":p["predicted_expression_mean"][i] for epoch,p in profiles.items()},
        }
        for name,mean in predicted.items():
            delta=mean.astype(np.float64)-ctrl_mean
            sign=lambda a:np.where(np.abs(a)<1e-8,0,np.sign(a))
            target_records.append(dict(
                method=name,exposure=role,condition=condition,n_reference_cells=n,
                model_scale_mse=float(np.mean((mean-true_mean)**2)),
                top20_model_scale_mse=float(np.mean((mean[top]-true_mean[top])**2)),
                pearson_delta=safe_correlation(delta,true_delta),
                spearman_delta=safe_correlation(delta,true_delta,rank=True),
                top20_direction_accuracy_excluding_target=float(
                    np.mean(sign(delta[eligible])==sign(true_delta[eligible]))) if len(eligible) else None,
                n_direction_genes=len(eligible),
            ))
    frame=pd.DataFrame(target_records)
    frame.to_csv(output/"per_target_normalized_metrics.csv",index=False)
    metrics=["model_scale_mse","top20_model_scale_mse","pearson_delta","spearman_delta",
             "top20_direction_accuracy_excluding_target"]
    summary=[]
    for (method,exposure),group in frame.groupby(["method","exposure"]):
        record=dict(method=method,exposure=exposure,n_targets=len(group),
                    n_reference_cells=int(group.n_reference_cells.sum()))
        for metric in metrics:
            good=group[metric].notna()
            record[metric+"_macro"]=float(group.loc[good,metric].mean()) if good.any() else None
            record[metric+"_cell_weighted"]=float(np.average(
                group.loc[good,metric],weights=group.loc[good,"n_reference_cells"])) if good.any() else None
            record[metric+"_valid_targets"]=int(good.sum())
        record.update({m:None for m in OFFICIAL_METRICS})
        record.update(
            official_aggregate=None,
            official_status="Not calculated here: normalized-scale diagnostics only; raw single-cell reference and count-layer evaluation are separate.",
        )
        summary.append(record)
    pd.DataFrame(summary).to_csv(output/"comparison_normalized_metrics.csv",index=False)
    write_json(output/"normalized_evaluation_manifest.json",dict(
        dataset=meta["dataset"],scale=meta["expression_scale"],
        final_evaluation=True,test_used_for_model_selection=False,
        raw_counts=False,challenge_leaderboard_result=False,
        reference_control_cells=nctrl,reference_control_role="reference_control",
        control_inputs_role="fit_control",baseline_training_role="fit",
        aggregation="Macro over perturbation targets; cell-weighted values separately labeled.",
        direction_diagnostic="Top 20 absolute Welch-t scores on model-normalized references; own target excluded; not official Wilcoxon metric.",
        near_zero_direction_tolerance=1e-8,
        undefined_correlations="Unavailable, excluded with explicit valid-target denominators.",
        inherited_preprocessing_caveat=meta["preprocessing_caveat"],
    ))
    if publish_dir:
        for p in output.iterdir():
            if p.is_file(): publish(p,Path(publish_dir)/p.name)
    print(pd.DataFrame(summary)[["method","exposure","model_scale_mse_macro","pearson_delta_macro"]].to_string(index=False))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepared",required=True)
    p.add_argument("--prediction-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--publish-dir")
    a=p.parse_args()
    evaluate(a.prepared,a.prediction_dir,a.output,a.publish_dir)
