"""Validate that adapter inputs remain independent of measured outcomes."""
import numpy as np


def validate_bundle(bundle):
    effect = bundle["predicted_effects"]
    controls = bundle["control_state"]
    assert effect.ndim == 3
    assert controls.shape == effect.shape[1:]
    assert effect.shape[0] == len(bundle["perturbation_ids"])
    assert effect.shape[2] == len(bundle["gene_ids"])
    assert len(set(bundle["gene_ids"])) == effect.shape[2]
    assert np.isfinite(effect).all() and np.isfinite(controls).all()
    np.testing.assert_allclose(
        bundle["predicted_expression"], effect+controls[None,:,:],
        rtol=1e-5, atol=1e-6,
    )
    assert bundle["perturbation_embeddings"].shape == (effect.shape[0],32)
    assert "truth" not in bundle and "measured_test_expression" not in bundle
    return True
