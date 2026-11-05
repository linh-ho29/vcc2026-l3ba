"""Prediction/control features for an unseen target; no measured outcome inputs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from common import load_prepared,publish,stable_seed,write_json
from inference.predict_effects import GEARSPredictor
from adapter_interface.validation import validate_bundle


def export_bundle(checkpoint,prepared,output,publish_dir=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    meta,obs,var,roles,graph,x=load_prepared(prepared)
    predictor=GEARSPredictor(checkpoint)
    rng=np.random.default_rng(stable_seed("export_control_panel"))
    rows=rng.choice(roles["fit_control"],size=64,replace=False)
    controls=x[rows].toarray().astype(np.float32)
    np.savez_compressed(output/"control_states.npz",control_state=controls,
                        cell_ids=obs.index[rows].to_numpy(dtype=str),
                        gene_ids=np.asarray(predictor.gene_ids),
                        gene_symbols=np.asarray(predictor.genes))
    target=predictor.metadata["frozen_split"]["conditions"]["test"][0]
    assert all(g not in predictor.training_targets for g in target.split("+") if g!="ctrl")
    bundle=predictor.predict_effects([target],controls)
    validate_bundle(bundle)
    arrays={k:v for k,v in bundle.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(output/"unseen_target_adapter_bundle.npz",**arrays)
    manifest={k:v for k,v in bundle.items() if not isinstance(v,np.ndarray)}
    manifest.update(
        input_control_cell_ids=obs.index[rows].tolist(),
        input_control_role="fit_control",
        measured_test_outcomes_used=False,preprocessing_version=meta["gears_commit"],
        gene_vocabulary="gene_index.csv",perturbation_vocabulary="target_index.csv",
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
        adapter_trained=False,external_embeddings="optional separately aligned inputs; checkpoint unchanged",
    )
    write_json(output/"adapter_manifest.json",manifest)
    if publish_dir:
        for path in output.iterdir():
            if path.is_file():
                publish(path,Path(publish_dir)/path.name)
    print("ADAPTER_BUNDLE_EXPORTED",target,controls.shape,flush=True)
    return rows,controls


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--prepared",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--publish-dir")
    a=p.parse_args()
    export_bundle(a.checkpoint,a.prepared,a.output,a.publish_dir)
