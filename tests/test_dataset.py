"""Tests for PerturbSeqDataset (train.py): condition grouping, per-cell-line
NTC sampling, and filtering of conditions missing a dependency. Synthetic
data only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import CELL_LINES, PerturbSeqDataset

NUM_GENES, D_LATENT = 20, 16


def _make_dataset(genes=("GENE_A", "GENE_B"), lines=("K562", "HepG2"), n_cells_per_condition=4):
    ntc_latents = {line: torch.randn(10, D_LATENT) for line in lines}
    ntc_expr = {line: torch.rand(10, NUM_GENES) for line in lines}
    gears_pert_bank = {g: torch.randn(4, 64) for g in genes}

    rows, latents, exprs = [], [], []
    for g in genes:
        for line in lines:
            for _ in range(n_cells_per_condition):
                rows.append({"gene": g, "cell_line": line})
                latents.append(torch.randn(D_LATENT))
                exprs.append(torch.rand(NUM_GENES))
    obs_df = pd.DataFrame(rows)
    pert_latents = torch.stack(latents)
    pert_expr = torch.stack(exprs)
    stat_delta_bank = {(g, line): torch.randn(NUM_GENES) * 0.2 for g in genes for line in lines}

    return PerturbSeqDataset(
        ntc_latents, ntc_expr, pert_latents, pert_expr, obs_df, gears_pert_bank, stat_delta_bank,
    )


def test_condition_count_matches_gene_times_line():
    ds = _make_dataset(genes=("A", "B", "C"), lines=("K562", "HepG2"))
    assert len(ds) == 3 * 2


def test_sample_batch_shapes():
    ds = _make_dataset()
    batch = ds.sample_batch("GENE_A", "K562", batch_size=12)
    assert batch["z_0"].shape == (12, D_LATENT)
    assert batch["x_ctrl"].shape == (12, NUM_GENES)
    assert batch["z_true"].shape == (12, D_LATENT)
    assert batch["y_true"].shape == (12, NUM_GENES)
    assert batch["e_p_stack"].shape == (12, 4, 64)
    assert batch["delta_stat"].shape == (12, NUM_GENES)
    assert batch["cell_line_idx"].shape == (12,)


def test_sample_batch_ntc_comes_from_same_cell_line_pool():
    ds = _make_dataset(lines=("K562", "HepG2"))
    batch = ds.sample_batch("GENE_A", "HepG2", batch_size=50)
    hepg2_pool = ds.ntc_latents["HepG2"]
    for row in batch["z_0"]:
        assert any(torch.allclose(row, pool_row) for pool_row in hepg2_pool)


def test_sample_batch_condition_level_inputs_constant_within_batch():
    ds = _make_dataset()
    batch = ds.sample_batch("GENE_A", "K562", batch_size=10)
    first = batch["e_p_stack"][0]
    assert all(torch.allclose(first, batch["e_p_stack"][i]) for i in range(10))
    assert (batch["cell_line_idx"] == CELL_LINES.index("K562")).all()


def test_drops_conditions_missing_gears_vector():
    genes = ("GENE_A", "GENE_B")
    lines = ("K562",)
    ntc_latents = {line: torch.randn(10, D_LATENT) for line in lines}
    ntc_expr = {line: torch.rand(10, NUM_GENES) for line in lines}
    gears_pert_bank = {"GENE_A": torch.randn(4, 64)}  # GENE_B missing on purpose
    rows, latents, exprs = [], [], []
    for g in genes:
        for _ in range(3):
            rows.append({"gene": g, "cell_line": "K562"})
            latents.append(torch.randn(D_LATENT))
            exprs.append(torch.rand(NUM_GENES))
    obs_df = pd.DataFrame(rows)
    stat_delta_bank = {(g, "K562"): torch.randn(NUM_GENES) for g in genes}

    ds = PerturbSeqDataset(
        ntc_latents, ntc_expr, torch.stack(latents), torch.stack(exprs),
        obs_df, gears_pert_bank, stat_delta_bank,
    )
    assert ("GENE_B", "K562") not in ds.condition_keys
    assert ("GENE_A", "K562") in ds.condition_keys


def test_raises_when_no_conditions_survive_filtering():
    ntc_latents = {"K562": torch.randn(5, D_LATENT)}
    ntc_expr = {"K562": torch.rand(5, NUM_GENES)}
    obs_df = pd.DataFrame([{"gene": "GENE_A", "cell_line": "K562"}])
    with pytest.raises(ValueError, match="No usable"):
        PerturbSeqDataset(
            ntc_latents, ntc_expr, torch.randn(1, D_LATENT), torch.rand(1, NUM_GENES),
            obs_df, gears_pert_bank={}, stat_delta_bank={},
        )
