"""Export normalized target profiles before reading final evaluation outcomes."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from common import load_prepared,publish,stable_seed,timestamp,write_json
from inference.predict_effects import GEARSPredictor


def export_predictions(checkpoint,prepared,output,publish_dir=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    meta,obs,var,roles,graph,x=load_prepared(prepared)
    predictor=GEARSPredictor(checkpoint)
    rng=np.random.default_rng(stable_seed("export_control_panel"))
    control_rows=rng.choice(roles["fit_control"],size=64,replace=False)
    control_state=x[control_rows].toarray()
    splits=predictor.metadata["frozen_split"]["conditions"]
    conditions=sorted(set(splits["test"])|set(splits["train"])-{"ctrl"})
    n,g=len(conditions),len(var)
    predicted=np.zeros((n,g),np.float32)
    effects=np.zeros((n,g),np.float32)
    slot0=np.zeros((n,g),np.float32)
    other=np.zeros((n,g),np.float32)
    start=time.time()
    for i,condition in enumerate(conditions):
        bundle=predictor.predict_effects([condition],control_state)
        delta=bundle["predicted_effects"][0]
        predicted[i]=bundle["predicted_expression"][0].mean(axis=0)
        effects[i]=delta.mean(axis=0)
        slot0[i]=delta[0]
        other[i]=delta[1:16].mean(axis=0)
        if i%100==0:
            print("PREDICTIONS",predictor.state["epoch"],i,n,
                  f"{time.time()-start:.1f}s",flush=True)
    epoch=predictor.state["epoch"]
    path=output/f"epoch_{epoch:02d}_profiles.npz"
    np.savez_compressed(
        path,conditions=np.asarray(conditions),predicted_expression_mean=predicted,
        predicted_effects_mean=effects,effect_slot0=slot0,effect_other_slots=other,
        control_state=control_state,control_cell_ids=obs.index[control_rows].to_numpy(dtype=str),
        control_expression_mean=control_state.mean(axis=0),
        gene_ids=np.asarray(predictor.gene_ids),gene_symbols=np.asarray(predictor.genes),
    )
    pd.DataFrame(dict(
        condition=conditions,
        exposure=["unseen_test" if c in splits["test"] else "seen_test" for c in conditions],
    )).to_csv(output/f"epoch_{epoch:02d}_condition_index.csv",index=False)
    record=dict(
        created=timestamp(),checkpoint_epoch=epoch,checkpoint_sha256=predictor.checkpoint_sha256,
        targets=n,response_genes=g,controls_per_target=64,control_role="fit_control",
        scale="natural log1p normalized expression",raw_counts=False,
        normalization=meta["normalization_target_sum"],batch_size=16,
        measured_test_outcomes_used_to_predict=False,seconds=time.time()-start,
        units="model-scale predicted profiles and additive model-scale deltas",
    )
    write_json(output/f"epoch_{epoch:02d}_prediction_manifest.json",record)
    if publish_dir:
        for p in [path,output/f"epoch_{epoch:02d}_condition_index.csv",
                  output/f"epoch_{epoch:02d}_prediction_manifest.json"]:
            publish(p,Path(publish_dir)/p.name)
    print("PREDICTIONS_EXPORTED",epoch,n,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--prepared",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--publish-dir")
    a=p.parse_args()
    export_predictions(a.checkpoint,a.prepared,a.output,a.publish_dir)
