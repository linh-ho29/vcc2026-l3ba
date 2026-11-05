"""Export precisely named upstream GEARS representations without retraining."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from common import publish, write_json
from inference.predict_effects import GEARSPredictor


def statistics(values):
    norm = np.linalg.norm(values, axis=1)
    singular = np.linalg.svd(values.astype(np.float64), compute_uv=False)
    probabilities = singular[singular > 0] / singular.sum() if singular.sum() else np.array([])
    return dict(
        shape=list(values.shape), norm_min=float(norm.min()), norm_max=float(norm.max()),
        norm_mean=float(norm.mean()), norm_median=float(np.median(norm)),
        norm_p05=float(np.quantile(norm,.05)), norm_p95=float(np.quantile(norm,.95)),
        zero_norm_rows=int((norm == 0).sum()),
        effective_rank=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if len(probabilities) else 0.,
        effective_rank_definition="exp(Shannon entropy of normalized singular values); uncentered",
    )


def extract(checkpoint, output, publish_dir=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    predictor=GEARSPredictor(checkpoint)
    model=predictor.model
    with torch.no_grad():
        ids=torch.arange(len(predictor.genes))
        lookup=model.gene_emb(ids)
        base=model.emb_trans(model.bn_emb(lookup))
        positions=model.emb_pos(ids.repeat(16))
        positional_pre=positions[:len(ids)].clone()
        for index,layer in enumerate(model.layers_emb_pos):
            positions=layer(positions,model.G_coexpress,model.G_coexpress_weight)
            if index < len(model.layers_emb_pos)-1:
                positions=positions.relu()
        positions=positions.reshape(16,len(ids),32)
        combined_0=model.emb_trans_v2(base + .2*positions[0])
        combined_other=model.emb_trans_v2(base + .2*positions[1])
        arrays=dict(
            gene_lookup=lookup.numpy().copy(),
            gene_position_pre=positional_pre.numpy().copy(),
            gene_position_post_slot0=positions[0].numpy().copy(),
            gene_position_post_other_slots=positions[1].numpy().copy(),
            gene_combined_slot0=combined_0.numpy().copy(),
            gene_combined_other_slots=combined_other.numpy().copy(),
            **predictor.target_representations(),
        )
    # Degree here excludes self-loops so graph-only support is explicit.
    go=model.G_sim.numpy()
    nonself=go[0] != go[1]
    in_degree=np.bincount(go[1,nonself],minlength=len(predictor.targets))
    out_degree=np.bincount(go[0,nonself],minlength=len(predictor.targets))
    co=model.G_coexpress.numpy()
    co_nonself=co[0] != co[1]
    gene_degree=np.bincount(co[1,co_nonself],minlength=len(predictor.genes))
    pd.DataFrame(dict(
        target_index=np.arange(len(predictor.targets)),gene_symbol=predictor.targets,
        direct_training_exposure=[g in predictor.training_targets for g in predictor.targets],
        in_response_panel=[g in predictor.genes for g in predictor.targets],
        go_in_degree_nonself=in_degree,go_out_degree_nonself=out_degree,
        graph_only=[g not in predictor.training_targets and in_degree[i]>0
                    for i,g in enumerate(predictor.targets)],
    )).to_csv(output/"target_index.csv",index=False)
    pd.DataFrame(dict(
        gene_index=np.arange(len(predictor.genes)),gene_id=predictor.gene_ids,
        gene_symbol=predictor.genes,coexpression_in_degree_nonself=gene_degree,
    )).to_csv(output/"gene_index.csv",index=False)
    np.savez_compressed(output/"representations.npz",**arrays)
    heldout=[p.replace("+ctrl","") for p in predictor.metadata["frozen_split"]["conditions"]["test"]]
    heldout_indices=[predictor.target_index[g] for g in heldout]
    summary=dict(
        checkpoint_epoch=predictor.state["epoch"],checkpoint_sha256=predictor.checkpoint_sha256,
        embedding_dimension=32,representations={k:statistics(v) for k,v in arrays.items()},
        target_vocabulary_size=len(predictor.targets),response_genes=len(predictor.genes),
        unseen_test_targets=len(heldout),
        unseen_test_targets_with_nonself_go_incoming_edges=int((in_degree[heldout_indices]>0).sum()),
        unsupported_dataset_conditions=predictor.metadata["unsupported_conditions"],
        definitions={
            "gene_lookup":"Learned gene_emb lookup; this tensor is not directly propagated by the coexpression GNN.",
            "gene_position_pre":"Separate learned emb_pos lookup before coexpression propagation.",
            "gene_position_post_slot0":"layers_emb_pos(emb_pos), graph-connected first batch slot.",
            "gene_position_post_other_slots":"Exact upstream propagation for subsequent batch slots.",
            "gene_combined":"emb_trans_v2(ReLU(bn_emb(gene_lookup)) + 0.2*gene_position_post).",
            "target_pre_go":"pert_emb lookup.",
            "target_post_go":"sim_layers output after GO propagation.",
            "target_fused":"pert_fuse(target_post_go), for a single-target perturbation.",
        },
        caveats=[
            "Forward preserves upstream single-copy coexpression edge indexing for repeated batch embeddings; exported slot semantics are explicit.",
            "Graph-only is absence of direct target examples, not proof of absent indirect training gradients.",
            "Nonzero predicted effects do not establish biological accuracy; evaluate against sealed measured references.",
        ],
    )
    write_json(output/"embedding_manifest.json",summary)
    if publish_dir:
        for path in output.iterdir():
            if path.is_file():
                publish(path,Path(publish_dir)/path.name)
    print("EMBEDDINGS_EXPORTED",predictor.state["epoch"],len(predictor.targets),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--publish-dir")
    a=p.parse_args()
    extract(a.checkpoint,a.output,a.publish_dir)
