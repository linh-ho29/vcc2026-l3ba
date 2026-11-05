#!/usr/bin/env python
"""Export gene/perturbation embeddings and held-out effect predictions."""
import argparse
import os

import numpy as np
import torch

from gears import GEARS
from gears.inference import evaluate
from gears.pertdata import PertData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/results/jurkat_gears/best_checkpoint.pt")
    ap.add_argument("--data_dir", default="/workspace/jurkat/gears_data")
    ap.add_argument("--dataset", default="nadig25jurkat")
    ap.add_argument("--out_dir", default="/workspace/jurkat/artifacts")
    args = ap.parse_args()

    pd_ = PertData(data_path=args.data_dir)
    pd_.load(data_path=os.path.join(args.data_dir, args.dataset))
    custom_split = os.path.join(args.data_dir, args.dataset, "splits", "custom_trackJ.pkl")
    pd_.prepare_split(split="custom", split_dict_path=custom_split)
    pd_.get_dataloader(batch_size=8, test_batch_size=16)

    model = GEARS(pd_, device="cpu")
    model.model_initialize(hidden_size=32)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.best_model.load_state_dict(ck["state_dict"])
    model.best_model.eval()

    # perturbation embeddings (GNN pert_emb table)
    pert_names = list(model.pert_list)
    pert_emb = model.best_model.pert_emb.weight.detach().cpu().numpy()
    np.savez(os.path.join(args.out_dir, "perturbation_embeddings.npz"),
             perturbations=np.array(pert_names), embeddings=pert_emb,
             checkpoint_epoch=ck.get("epoch"))
    print(f"[export] perturbation_embeddings: {pert_emb.shape} for {len(pert_names)} perts")

    # gene embeddings: gene2vec init table used by the model (frozen input emb)
    gene_names = list(pd_.adata.var["gene_name"])
    gene_emb = None
    for attr in ["emb_gene", "gene_emb"]:
        if hasattr(model.best_model, attr):
            t = getattr(model.best_model, attr)
            gene_emb = (t.weight if hasattr(t, "weight") else t).detach().cpu().numpy()
            break
    if gene_emb is None:  # fall back to the config-stored gene2vec table
        ge = model.config.get("gene_emb")
        gene_emb = np.asarray(ge) if ge is not None else None
    if gene_emb is not None:
        n = min(len(gene_names), gene_emb.shape[0])
        np.savez(os.path.join(args.out_dir, "gene_embeddings.npz"),
                 genes=np.array(gene_names[:n]), embeddings=gene_emb[:n],
                 note="gene2vec-initialized gene embedding table (frozen)")
        print(f"[export] gene_embeddings: {gene_emb[:n].shape}")
    else:
        print("[export] no gene embedding table found on model")

    # held-out test predictions
    test_res = evaluate(pd_.dataloader["test_loader"], model.best_model,
                        model.config["uncertainty"], "cpu")
    np.savez(os.path.join(args.out_dir, "heldout_effect_predictions.npz"),
             pert_cat=np.array(test_res["pert_cat"]),
             pred=test_res["pred"], truth=test_res["truth"],
             pred_de=test_res["pred_de"], truth_de=test_res["truth_de"],
             checkpoint_epoch=ck.get("epoch"),
             note="pred/truth are post-perturbation expression, CP10k+log1p space")
    print(f"[export] heldout predictions: {test_res['pred'].shape}")


if __name__ == "__main__":
    main()
