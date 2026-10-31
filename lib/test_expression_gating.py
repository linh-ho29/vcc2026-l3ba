"""Unit tests for expression_gating.py - synthetic data, no real files needed."""
import numpy as np
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights, apply_gate_to_effect


def test_undetected_gene_gates_to_zero_except_in_none_mode():
    """Biological safeguard: zero detection across every control cell means
    zero confidence in a transferred effect - for every gating mode EXCEPT
    'none', which is a deliberate escape hatch representing the genuinely
    unmodified baseline (matching what's actually submitted today) for fair
    ablation comparison. Gating-related safeguards only apply once gating
    is actually turned on."""
    stats = {
        "detection_fraction": np.array([0.0, 0.5, 1.0]),
        "percentile": np.array([50.0, 50.0, 99.0]),
        "baseline_mean": np.array([0.0, 2.0, 10.0]),
    }
    for mode in ["binary_detection", "percentile", "combined"]:
        w = compute_gate_weights(stats, GateConfig(mode=mode))
        assert w[0] == 0.0, f"mode={mode}: undetected gene did not gate to zero"

    w_none = compute_gate_weights(stats, GateConfig(mode="none"))
    assert np.allclose(w_none, 1.0), "mode=none must be a true no-op, matching the unmodified baseline"


def test_none_mode_is_fully_permissive():
    stats = {"detection_fraction": np.array([0.3, 0.9]), "percentile": np.array([20.0, 80.0]),
             "baseline_mean": np.array([1.0, 5.0])}
    w = compute_gate_weights(stats, GateConfig(mode="none"))
    assert np.allclose(w, 1.0)


def test_gate_weights_bounded():
    rng = np.random.default_rng(0)
    stats = {
        "detection_fraction": rng.uniform(0, 1, 100),
        "percentile": rng.uniform(0, 100, 100),
        "baseline_mean": rng.uniform(0, 20, 100),
    }
    for mode in ["binary_detection", "percentile", "combined"]:
        w = compute_gate_weights(stats, GateConfig(mode=mode))
        assert np.all(w >= 0.0) and np.all(w <= 1.0), f"mode={mode} produced out-of-bounds weight"


def test_monotonic_in_detection_fraction():
    """Higher detection fraction should never produce a lower gate weight,
    all else equal (smooth monotonic gate, not an arbitrary function)."""
    d = np.linspace(0.001, 1.0, 50)  # avoid exact 0 to not hit the hard-zero floor
    stats = {"detection_fraction": d, "percentile": np.full(50, 50.0), "baseline_mean": d * 10}
    w = compute_gate_weights(stats, GateConfig(mode="combined"))
    assert np.all(np.diff(w) >= -1e-9), "gate weight not monotonic in detection fraction"


def test_apply_gate_zero_weight_zeroes_effect():
    raw_effect = np.array([1.0, -0.8, 0.5])
    gate = np.array([0.0, 1.0, 0.0])
    gated = apply_gate_to_effect(raw_effect, gate)
    assert gated[0] == 0.0 and gated[2] == 0.0
    assert gated[1] == raw_effect[1]  # full-confidence gene passes through unchanged


def test_apply_gate_preserves_sign():
    """Gating scales confidence, it must never flip the direction of an effect."""
    raw_effect = np.array([1.5, -1.5])
    gate = np.array([0.3, 0.3])
    gated = apply_gate_to_effect(raw_effect, gate)
    assert np.sign(gated[0]) == np.sign(raw_effect[0])
    assert np.sign(gated[1]) == np.sign(raw_effect[1])


def test_compute_context_gate_stats_dense():
    # 4 cells, 3 genes: gene 0 always zero, gene 1 always expressed, gene 2 half expressed
    X = np.array([
        [0, 5, 3],
        [0, 4, 0],
        [0, 6, 2],
        [0, 5, 0],
    ], dtype=np.float32)
    stats = compute_context_gate_stats(X)
    assert stats["detection_fraction"][0] == 0.0
    assert stats["detection_fraction"][1] == 1.0
    assert stats["detection_fraction"][2] == 0.5
    assert stats["baseline_mean"][0] == 0.0
    # gene 1 has the highest mean -> highest percentile
    assert stats["percentile"][1] == 100.0
    # gene 0 has the lowest mean -> lowest percentile
    assert stats["percentile"][0] == 0.0


def test_direction_aware_downregulation_matches_combined_gate():
    """For negative (downregulation) effects, direction-aware gating should
    be identical to the plain combined gate - no special treatment."""
    from expression_gating import compute_gate_weights_direction_aware
    stats = {"detection_fraction": np.array([0.0, 0.5]), "percentile": np.array([50.0, 80.0]),
              "baseline_mean": np.array([0.0, 3.0])}
    raw_effect = np.array([-1.0, -0.5])  # both downregulation
    config = GateConfig(mode="combined")
    direction_aware = compute_gate_weights_direction_aware(stats, raw_effect, config)
    plain_combined = compute_gate_weights(stats, config)
    assert np.allclose(direction_aware, plain_combined)


def test_direction_aware_induction_is_not_hard_zeroed():
    """An undetected gene with a POSITIVE (induction) K562 effect should not
    be forced to a hard zero the way downregulation would be - absence of
    baseline detection doesn't rule out inducibility."""
    from expression_gating import compute_gate_weights_direction_aware
    stats = {"detection_fraction": np.array([0.0]), "percentile": np.array([50.0]),
              "baseline_mean": np.array([0.0])}
    raw_effect_up = np.array([1.0])
    raw_effect_down = np.array([-1.0])
    config = GateConfig(mode="combined")

    gate_up = compute_gate_weights_direction_aware(stats, raw_effect_up, config)
    gate_down = compute_gate_weights_direction_aware(stats, raw_effect_down, config)

    assert gate_down[0] == 0.0, "downregulation on an undetected gene should still hard-zero"
    assert gate_up[0] > 0.0, "induction on an undetected gene should NOT hard-zero"


def test_direction_aware_induction_still_discounted():
    """Induction on an undetected gene gets SOME weight, but not full
    confidence - we have no direct evidence it's actually inducible."""
    from expression_gating import compute_gate_weights_direction_aware
    stats = {"detection_fraction": np.array([0.0]), "percentile": np.array([95.0]),
              "baseline_mean": np.array([0.0])}
    raw_effect_up = np.array([1.0])
    gate = compute_gate_weights_direction_aware(stats, raw_effect_up, GateConfig(mode="combined"))
    assert 0.0 < gate[0] < 1.0, "induction weight should be discounted, not full confidence"


if __name__ == "__main__":
    import sys
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)