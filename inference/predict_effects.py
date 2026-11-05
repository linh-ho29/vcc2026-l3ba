"""Checkpoint-only adapter-ready inference; normalized expression, not raw counts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from common import GEARS_COMMIT, build_model, digest


class GEARSPredictor:
    def __init__(self, checkpoint, threads=4):
        self.checkpoint = str(checkpoint)
        self.state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.metadata = self.state["dataset"]
        self.model = build_model(self.state["graph"])
        self.model.load_state_dict(self.state["model"])
        self.model.eval()
        torch.set_num_threads(threads)
        self.genes = self.metadata["gene_symbols"]
        self.gene_ids = self.metadata["gene_ids"]
        self.targets = self.metadata["target_symbols"]
        self.target_index = {g:i for i,g in enumerate(self.targets)}
        self.training_targets = {
            g for p in self.metadata["frozen_split"]["conditions"]["train"]
            for g in p.split("+") if g != "ctrl"
        }
        self.checkpoint_sha256 = digest(checkpoint)

    def _ids(self, condition):
        return [g for g in condition.split("+") if g != "ctrl"]

    def predict_effects(self, perturbation_ids, control_cells, context_id=None):
        """One target condition per call; effects shape [targets, controls, genes].

        control_cells must be a finite float32 model-scale matrix in exact gene
        order, or AnnData with exact var_names matching stored symbols/Ensembl IDs.
        Unknown contexts are rejected: this checkpoint was fitted in RPE1 only.
        """
        if context_id not in (None, "rpe1", "RPE1"):
            raise ValueError("This is an RPE1 model, not a cross-context model")
        if hasattr(control_cells, "var_names"):
            if list(control_cells.var_names) not in [self.genes, self.gene_ids]:
                raise ValueError("Control-cell gene ordering does not match checkpoint")
            values = control_cells.X
            values = values.toarray() if hasattr(values, "toarray") else values
        else:
            values = control_cells
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.genes) or not len(values):
            raise ValueError("Controls must have shape [n_controls, n_response_genes]")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite controls")
        if isinstance(perturbation_ids, str):
            perturbation_ids = [perturbation_ids]
        embeddings = self.target_representations()
        outputs, target_emb, masks = [], [], []
        with torch.no_grad():
            for condition in perturbation_ids:
                target_ids = self._ids(condition)
                if not target_ids or any(g not in self.target_index for g in target_ids):
                    raise ValueError(f"Unsupported perturbation target: {condition}")
                if len(target_ids) != 1:
                    raise ValueError("This RPE1 single-target baseline does not validate combinations")
                ids = [self.target_index[g] for g in target_ids]
                predictions = []
                for start in range(0, len(values), 16):
                    chunk = values[start:start+16]
                    n = len(chunk)
                    # Stable batch slots, including a final partial chunk.
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 16-n, axis=0)])
                    batch = Batch.from_data_list([
                        Data(x=torch.from_numpy(v[:,None]), pert_idx=ids, pert=condition)
                        for v in chunk
                    ])
                    predictions.append(self.model(batch).numpy()[:n])
                prediction = np.concatenate(predictions)
                outputs.append(prediction)
                vector = embeddings["target_post_go"][ids].sum(axis=0)
                fused = self.model.pert_fuse(torch.tensor(np.stack([vector, vector])))[0].numpy()
                target_emb.append(fused)
                masks.append([g in self.training_targets for g in target_ids])
        expression = np.stack(outputs)
        return dict(
            predicted_expression=expression, predicted_effects=expression-values[None,:,:],
            control_state=values.copy(), gene_ids=self.gene_ids, gene_symbols=self.genes,
            perturbation_ids=list(perturbation_ids), perturbation_embeddings=np.stack(target_emb),
            response_gene_coverage=np.ones(len(self.genes), dtype=bool),
            target_coverage=np.ones(len(perturbation_ids), dtype=bool),
            training_exposure=masks, context_id="rpe1",
            effect_units="additive change in natural-log1p normalized expression",
            expression_scale=self.metadata["expression_scale"],
            checkpoint_epoch=self.state["epoch"], checkpoint_sha256=self.checkpoint_sha256,
            model_version=f"GEARS@{GEARS_COMMIT}",
            inference_batch_size=16,
            caveat="Exact upstream graph propagation is batch-slot-dependent; fixed batch-16 ordering is part of this interface.",
        )

    @torch.no_grad()
    def target_representations(self):
        ids = torch.arange(len(self.targets))
        before = self.model.pert_emb(ids)
        after = before
        for index, layer in enumerate(self.model.sim_layers):
            after = layer(after, self.model.G_sim, self.model.G_sim_weight)
            if index < self.model.num_layers-1:
                after = after.relu()
        fused = self.model.pert_fuse(after)
        return dict(target_pre_go=before.numpy().copy(),
                    target_post_go=after.numpy().copy(),
                    target_fused=fused.numpy().copy())


def predict_effects(checkpoint, perturbation_ids, control_cells, context_id=None):
    return GEARSPredictor(checkpoint).predict_effects(
        perturbation_ids, control_cells, context_id=context_id)
