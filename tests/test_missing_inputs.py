"""Missing/invalid input handling: absent GEARS, bad gene ordering, unit
mismatch, availability-mask semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_adapter import AdapterConfig, HybridAdapter, BatchValidationError
from hybrid_adapter.synthetic import SyntheticConfig, generate_whole_line_batch

B, G = 3, 24


def _batch(**kwargs):
    cfg = SyntheticConfig(batch_size=B, n_genes=G)
    return generate_whole_line_batch(cfg, **kwargs)


def test_missing_gears_effects_produce_no_invalid_values():
    model = HybridAdapter(AdapterConfig())
    batch = _batch(include_gears=False)
    out = model(batch)
    assert torch.isfinite(out["corrected_effects"]).all()
    assert torch.isfinite(out["residual_effects"]).all()


def test_missing_gears_with_gears_only_mode_raises():
    model = HybridAdapter(AdapterConfig(perturbation_mode="gears_only"))
    batch = _batch(include_gears=False)
    with pytest.raises(BatchValidationError, match="gears_only"):
        model(batch)


def test_gears_only_mode_with_gears_present():
    model = HybridAdapter(AdapterConfig(perturbation_mode="gears_only"))
    out = model(_batch(include_gears=True))
    assert torch.isfinite(out["corrected_effects"]).all()


def test_scgpt_only_mode():
    model = HybridAdapter(AdapterConfig(perturbation_mode="scgpt_only"))
    out = model(_batch(include_gears=True))
    assert torch.isfinite(out["corrected_effects"]).all()


def test_invalid_gene_ordering_raises():
    """Duplicated / inconsistent gene axes must be rejected."""
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    bad = dict(batch)
    gene_ids = list(batch["response_gene_ids"])
    gene_ids[0] = gene_ids[1]  # duplicate -> ambiguous ordering
    bad["response_gene_ids"] = gene_ids
    with pytest.raises(BatchValidationError, match="duplicates"):
        model(bad)


def test_incompatible_effect_units_raise():
    model = HybridAdapter(AdapterConfig(effect_units="log2_fold_change"))
    batch = _batch()
    batch["effect_units"] = "raw_counts"  # incompatible scale
    with pytest.raises(BatchValidationError, match="units"):
        model(batch)


def test_incompatible_expression_units_raise():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    batch["expression_units"] = "z_score"
    with pytest.raises(BatchValidationError, match="units"):
        model(batch)


def test_unknown_batch_key_raises():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    batch["mystery_field"] = 1
    with pytest.raises(BatchValidationError, match="Unknown batch keys"):
        model(batch)


def test_unknown_availability_mask_key_raises():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    batch["availability_masks"] = dict(batch["availability_masks"], mystery_source=True)
    with pytest.raises(BatchValidationError, match="Unknown availability mask"):
        model(batch)


def test_explicit_mask_overrides_presence():
    """A tensor present but masked unavailable must be treated as absent."""
    model = HybridAdapter(AdapterConfig())
    batch = _batch(include_gears=True)
    batch["availability_masks"] = dict(batch["availability_masks"], gears_effects=False)
    out = model(batch)
    assert torch.isfinite(out["corrected_effects"]).all()


def test_missing_required_key_raises():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    del batch["control_features"]
    with pytest.raises(BatchValidationError, match="Missing required batch keys"):
        model(batch)


def test_shape_mismatch_raises():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    batch["stat_effects"] = batch["stat_effects"][:, :-1]  # wrong gene count
    with pytest.raises(BatchValidationError, match="shape"):
        model(batch)


def test_stat_coverage_must_be_binary():
    model = HybridAdapter(AdapterConfig())
    batch = _batch()
    batch["stat_coverage"] = torch.full((B, G), 0.5)
    with pytest.raises(BatchValidationError, match="binary"):
        model(batch)
