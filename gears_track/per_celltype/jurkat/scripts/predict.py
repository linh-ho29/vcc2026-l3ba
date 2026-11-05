#!/usr/bin/env python
"""Track J predict.py: adapter-ready inference interface for the Jurkat GEARS model.

Exposes predict_perturbation(perturbation_id, control_cells) returning:
  - effect vector (predicted delta vs control mean, GEARS log-normalized space)
  - predicted perturbed expression vector
  - perturbation embedding (from the GNN pert_emb table)
  - optional gene embeddings
  - gene vocabulary
  - effect scale (L2 norm of effect vector)
  - graph coverage (whether the target is a node in the GEARS pert graph)
  - model + preprocessing metadata

CLI:
  python predict.py --checkpoint best_checkpoint.pt --perturbation NELFE [--out json]
  python predict.py --export_manifest   # writes MANIFEST.json for the adapter
"""
import argparse
import json
import os
import pickle

import anndata as ad
import numpy as np
import torch

from gears import GEARS
from gears.pertdata import PertData


class JurkatGearsAdapter:
    def __init__(self, checkpoint, data_dir="/workspace/jurkat/gears_data",
                 dataset="nadig25jurkat", device="cpu"):
        self.device = device
        self.pd_ = PertData(data_path=data_dir)
        self.pd_.load(data_path=os.path.join(data_dir, dataset))
        custom_split = os.path.join(data_dir, dataset, "splits", "custom_trackJ.pkl")
        self.pd_.prepare_split(split="custom", split_dict_path=custom_split)
        self.pd_.get_dataloader(batch_size=8, test_batch_size=16)
        self.model = GEARS(self.pd_, device=device)
        self.model.model_initialize(hidden_size=32)
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.best_model.load_state_dict(ck["state_dict"])
        self.model.best_model.eval()
        self.adata = self.pd_.adata
        self.ctrl_mean = np.asarray(self.adata[self.adata.obs["condition"] == "ctrl"]
                                    .X.mean(axis=0)).ravel()
        self.gene_vocab = list(self.adata.var["gene_name"])
        self.pert2idx = dict(self.model.node_map_pert.items()) if hasattr(
            self.model, "node_map_pert") else {}
        self.pert_list = list(self.model.pert_list)
        self.meta = {"checkpoint": checkpoint, "checkpoint_epoch": ck.get("epoch"),
                     "hidden_size": 32, "seed": 42, "cell_line": "Jurkat",
                     "normalization": "CP10k + log1p",
                     "n_genes": len(self.gene_vocab),
                     "n_pert_graph_nodes": len(self.pert_list)}

    def perturbation_embedding(self, perturbation_id):
        """Row of pert_emb for the perturbation graph node ('GENE+ctrl')."""
        key = perturbation_id if not perturbation_id.endswith("+ctrl") else perturbation_id
        node = key if key in self.pert2idx else f"{key}+ctrl"
        if node not in self.pert2idx:
            return None, False
        idx = self.pert2idx[node]
        emb = self.model.best_model.pert_emb.weight[idx].detach().cpu().numpy()
        return emb, True

    def predict_perturbation(self, perturbation_id, control_cells=None):
        """control_cells: optional AnnData/matrix of control cells (unused by GEARS,
        which uses its internal ctrl profile; accepted for interface compatibility
        and returned in metadata)."""
        gene = perturbation_id.replace("+ctrl", "")
        emb, covered = self.perturbation_embedding(gene)
        if not covered:
            return {"perturbation_id": perturbation_id, "graph_coverage": False,
                    "error": "target not in GEARS perturbation graph; "
                             "cross-graph prediction requires a separate adaptation stage"}
        pred = self.model.predict([[gene]])[gene]  # log-normalized space
        effect = pred - self.ctrl_mean
        return {"perturbation_id": perturbation_id,
                "graph_coverage": True,
                "effect_vector": effect.tolist(),
                "predicted_expression": pred.tolist(),
                "perturbation_embedding": emb.tolist(),
                "gene_embeddings_optional": None,
                "gene_vocabulary": self.gene_vocab,
                "effect_scale_l2": float(np.linalg.norm(effect)),
                "control_cells_used": 0 if control_cells is None else 1,
                "metadata": self.meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/results/jurkat_gears/best_checkpoint.pt")
    ap.add_argument("--perturbation", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export_manifest", action="store_true")
    args = ap.parse_args()
    adapter = JurkatGearsAdapter(args.checkpoint)
    if args.export_manifest:
        manifest = {
            "adapter": "JurkatGearsAdapter (predict.py)",
            "interface": "predict_perturbation(perturbation_id, control_cells) -> dict",
            "returns": ["effect_vector", "predicted_expression",
                        "perturbation_embedding", "gene_embeddings_optional",
                        "gene_vocabulary", "effect_scale_l2", "graph_coverage",
                        "metadata"],
            "effect_definition": "predicted_expression - control_mean, "
                                 "CP10k+log1p space (same as training)",
            "graph_coverage_note": "False for targets absent from the GEARS "
                                   "perturbation graph; no transfer without adaptation",
            "cross_cell_line_note": "predictions are Jurkat-specific; do not transfer "
                                    "to another cell line without context adaptation",
            "model_metadata": adapter.meta,
        }
        out = args.out or "/mnt/results/jurkat_gears/MANIFEST.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[manifest] {out}")
        return
    if not args.perturbation:
        raise SystemExit("--perturbation required (or --export_manifest)")
    res = adapter.predict_perturbation(args.perturbation)
    out = args.out or f"/mnt/results/jurkat_gears/prediction_{args.perturbation}.json"
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"[predict] {args.perturbation}: coverage={res['graph_coverage']} "
          f"scale={res.get('effect_scale_l2')}")


if __name__ == "__main__":
    main()
