"""Primary exports after eight completed epochs; no optional count work here."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from adapter_interface.export_features import export_bundle
from common import write_json
from inference.extract_embeddings import extract
from inference.export_predictions import export_predictions


def run(run_dir,prepared,output,publish_root):
    run_dir,output,publish_root=Path(run_dir),Path(output),Path(publish_root)
    last=torch.load(run_dir/"checkpoints/epoch_08.pt",map_location="cpu",weights_only=False)
    if last["epoch"]!=8:
        raise ValueError("Primary export requires eight completed epochs")
    expected=[run_dir/"checkpoints"/f"epoch_{i:02d}.pt" for i in range(1,9)]
    if not all(p.exists() for p in expected):
        raise ValueError("Missing epoch checkpoint")
    # Freeze selection/generation policy before opening final outcome references.
    write_json(output/"evaluation_settings_frozen.json",dict(
        epochs=[2,4,6,8],selected_by_validation_epoch=last["best_epoch"],
        controls_per_target=64,control_selection_seed_label="export_control_panel",
        inference_batch_size=16,baseline="equal-weight fitting-target mean effect",
        count_layer_policy="separate optional module, no test tuning",
        heldout_targets_unchanged=True,
    ))
    extract(expected[-1],output/"embeddings",publish_root/"embeddings")
    best=last["best_epoch"]
    if best!=8:
        extract(expected[best-1],output/f"embeddings_best_epoch_{best:02d}",
                publish_root/f"embeddings_best_epoch_{best:02d}")
    export_bundle(expected[-1],prepared,output/"adapter_interface",
                  publish_root/"adapter_interface")
    # Export epoch 8 first, then the prespecified comparison checkpoints.
    for epoch in [8,2,4,6]:
        export_predictions(expected[epoch-1],prepared,output/"predictions",
                           publish_root/"predictions")
    write_json(output/"primary_exports_complete.json",dict(
        completed_training_epochs=8,embeddings=True,adapter_bundle=True,
        predictions_epochs=[2,4,6,8],adapter_trained=False,
        github_uploaded=False,
    ))
    print("PRIMARY_EXPORTS_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True)
    p.add_argument("--prepared",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--publish-root",required=True)
    a=p.parse_args()
    run(a.run_dir,a.prepared,a.output,a.publish_root)
