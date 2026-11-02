"""H7 validation package tests on synthetic data (Phase 1 + Phase 2 functions)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from h7_validation import (
    compsim,
    compare_shared_vs_per_line_alpha,
    cross_line_transfer_accuracy,
    fit_alpha_for_phase,
    fit_alpha_per_line,
    head_to_head_transfer,
    homogeneity_passes,
    homogeneity_variance_ratios,
    interpret_ordering,
    phase_proportions,
    phase_silhouette,
    score_cell_cycle_phases,
    total_variation_distance,
    universality_summary,
)
from h7_validation.phase_deltas import (
    between_within_phase_variance_ratio,
    masked_response_check,
    per_phase_delta,
)

N_CELLS, N_GENES = 300, 80
PHASES = ("G1", "S", "G2M")
MARKERS = ["PCNA", "MCM2", "TYMS", "FEN1", "MCM4", "RRM1", "UNG", "GINS2",
           "MCM5", "MCM6", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2",
           "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B"]


# Shared cell-cycle programs across lines (H7 claim 6: same states everywhere).
PHASE_CENTERS = None


def _synthetic_line(rng, composition, phase_shift=1.0):
    global PHASE_CENTERS
    gene_names = [f"G{i:04d}" for i in range(N_GENES)]
    for i, m in enumerate(MARKERS):
        gene_names[i] = m
    if PHASE_CENTERS is None:
        PHASE_CENTERS = rng.normal(0, phase_shift, size=(len(PHASES), N_GENES))
    assign = rng.choice(len(PHASES), size=N_CELLS, p=composition)
    expr = PHASE_CENTERS[assign] + rng.normal(0, 0.15, size=(N_CELLS, N_GENES))
    labels = np.array([PHASES[i] for i in assign])
    return expr, labels, gene_names


def test_phase_scoring_recovers_structure():
    rng = np.random.default_rng(0)
    expr, labels, gene_names = _synthetic_line(rng, [1 / 3, 1 / 3, 1 / 3])
    scored, scores = score_cell_cycle_phases(expr, gene_names)
    # Scoring runs and produces the three labels.
    assert set(np.unique(scored)) <= set(PHASES)
    assert set(np.unique(scored)) == set(PHASES)
    assert "S.score" in scores and "G2M.score" in scores


def test_phase_silhouette_positive_for_discrete_phases():
    rng = np.random.default_rng(1)
    expr, labels, gene_names = _synthetic_line(rng, [1 / 3, 1 / 3, 1 / 3], phase_shift=2.0)
    sil = phase_silhouette(expr, labels)
    assert sil["overall"] > 0.2  # discrete clusters


def test_homogeneity_variance_ratios_below_one():
    rng = np.random.default_rng(2)
    expr, labels, gene_names = _synthetic_line(rng, [0.4, 0.3, 0.3], phase_shift=1.5)
    ratios = homogeneity_variance_ratios(expr, labels)
    assert homogeneity_passes(ratios)
    for stats in ratios.values():
        assert stats["ratio"] < 1.0


def test_cross_line_transfer_accuracy_above_chance():
    rng = np.random.default_rng(3)
    expr_a, lab_a, genes = _synthetic_line(rng, [1 / 3, 1 / 3, 1 / 3])
    expr_b, lab_b, _ = _synthetic_line(rng, [0.3, 0.4, 0.3])
    res = cross_line_transfer_accuracy(expr_a, lab_a, expr_b, lab_b)
    assert res["accuracy"] > res["chance"]
    assert res["missing_phases"] == []


def test_universality_summary_runs():
    rng = np.random.default_rng(4)
    lines = {}
    for i, name in enumerate(("A", "B", "C")):
        expr, lab, _ = _synthetic_line(rng, [0.4, 0.3, 0.3])
        lines[name] = (expr, lab)
    summary = universality_summary(lines)
    assert set(summary.keys()) == {"A", "B", "C"}
    for res in summary.values():
        assert res["accuracy"] > res["chance"]


def test_per_phase_delta_and_variance_ratio():
    rng = np.random.default_rng(5)
    control = rng.normal(0, 0.2, size=(150, N_GENES))
    control_ph = np.array([PHASES[i] for i in rng.choice(3, 150)])
    perturbed = control + 0.8  # uniform shift plus phase-specific structure
    pert_ph = control_ph
    deltas = {}
    for k, phase in enumerate(PHASES):
        d, n = per_phase_delta(perturbed + k * 0.5, pert_ph, control, control_ph, phase)
        deltas[phase] = d
        assert n > 0
    ratio = between_within_phase_variance_ratio(deltas)
    assert np.isfinite(ratio) and ratio > 0


def test_per_phase_delta_min_cells_guard():
    rng = np.random.default_rng(6)
    control = rng.normal(size=(10, N_GENES))
    control_ph = np.array(["G1"] * 10)
    perturbed = rng.normal(size=(10, N_GENES))
    pert_ph = np.array(["G1"] * 10)
    with pytest.raises(ValueError, match="too few cells"):
        per_phase_delta(perturbed, pert_ph, control, control_ph, "S")


def test_alpha_decomposition_recovers_planted_alpha():
    """Generate deltas with a known alpha and check the fit recovers it."""
    rng = np.random.default_rng(7)
    G = 200
    true_alpha = 0.7
    invariant = rng.normal(0, 1.0, size=G)
    deltas = {}
    for line in range(4):
        dependent = rng.normal(0, 1.0, size=G)
        deltas[f"line{line}"] = true_alpha * invariant + (1 - true_alpha) * dependent
    fit = fit_alpha_for_phase(deltas)
    assert abs(fit["alpha"] - true_alpha) < 0.15
    assert 0.0 <= fit["alpha"] <= 1.0


def test_shared_vs_per_line_alpha_comparison():
    rng = np.random.default_rng(8)
    G = 150
    invariant = rng.normal(0, 1.0, size=G)
    deltas = {}
    for line in range(3):
        dependent = rng.normal(0, 1.0, size=G)
        deltas[f"line{line}"] = 0.6 * invariant + 0.4 * dependent
    comparison = compare_shared_vs_per_line_alpha(deltas)
    assert 0.0 <= comparison["shared_alpha"] <= 1.0
    assert np.isfinite(comparison["p_value"])
    # With a truly shared alpha, claim 5 should be supported.
    assert comparison["claim5_supported"]


def test_sign_flip_counts():
    rng = np.random.default_rng(9)
    G = 100
    invariant = rng.normal(0, 1.0, size=G)
    deltas = {f"line{i}": invariant + rng.normal(0, 1.0, size=G) for i in range(3)}
    counts = __import__("h7_validation.alpha_decomposition", fromlist=["sign_flip_counts"]).sign_flip_counts(deltas)
    assert counts["n_sign_flip_genes"] >= 0


def test_compsim_and_tv():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert abs(total_variation_distance(p, q) - 0.3) < 1e-9
    assert abs(compsim(p, p) - 1.0) < 1e-9
    assert compsim(p, q) == 1.0 - 0.3
    with pytest.raises(ValueError):
        total_variation_distance(p, np.array([0.5, 0.3, 0.4]))  # doesn't sum to 1


def test_head_to_head_transfer_runs():
    rng = np.random.default_rng(10)
    G = 60
    lines = ["A", "B", "C"]
    proportions = {"A": np.array([0.5, 0.3, 0.2]),
                   "B": np.array([0.45, 0.35, 0.2]),
                   "C": np.array([0.2, 0.3, 0.5])}
    line_deltas, phase_deltas = {}, {}
    for pert in ("pert1", "pert2"):
        line_deltas[pert] = {l: rng.normal(0, 0.5, size=G) for l in lines}
        phase_deltas[pert] = {
            l: {p: rng.normal(0, 0.5, size=G) for p in PHASES} for l in lines
        }
    result = head_to_head_transfer(
        line_deltas, phase_deltas, proportions, held_out_perturbations=["pert1", "pert2"]
    )
    assert "h2_mean_error" in result and "h7_mean_error" in result
    assert len(result["pairs"]) > 0


def test_phase_proportions():
    labels = np.array(["G1", "G1", "S", "G2M", "G1"])
    p = phase_proportions(labels)
    assert abs(p.sum() - 1.0) < 1e-9
    assert abs(p[0] - 0.6) < 1e-9


def test_ablation_config_mapping():
    from h7_validation import ABLATION_MODELS, config_for_ablation

    m0 = config_for_ablation("M0")
    assert m0.response_mode == "whole_line"
    for variant, alpha_mode, comp in [
        ("M1", "global", False),
        ("M2", "phase_shared", False),
        ("M3", "phase_per_line", False),
        ("M4", "phase_shared", True),
    ]:
        cfg = config_for_ablation(variant, n_phases=4)
        assert cfg.response_mode == "subpopulation"
        assert cfg.alpha_mode == alpha_mode
        assert cfg.composition_weighting == comp
    with pytest.raises(ValueError):
        config_for_ablation("M9")


def test_interpret_ordering():
    res = interpret_ordering("M4 > M2 > M1 > M0")
    assert res["h7_predicted_ordering_holds"]
    assert res["subpopulation_resolution_helps"]
    res_fail = interpret_ordering("M0 > M1 > M2 > M4")
    assert not res_fail["h7_predicted_ordering_holds"]
    assert not res_fail["subpopulation_resolution_helps"]
