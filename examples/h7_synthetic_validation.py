"""H7 synthetic validation demo: run the Phase 1 -> Phase 2 pipeline end-to-end
on artificial data whose generative process follows the H7 hypothesis, then
run the M0-M4 ablation harness.

*** ALL DATA HERE IS ARTIFICIAL. ***
Results demonstrate that the *code* works, not that H7 holds biologically.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from h7_validation import (
    compsim,
    compare_shared_vs_per_line_alpha,
    head_to_head_transfer,
    homogeneity_passes,
    homogeneity_variance_ratios,
    interpret_ordering,
    phase_proportions,
    phase_silhouette,
    run_ablation,
    score_cell_cycle_phases,
    total_variation_distance,
    universality_summary,
)
from h7_validation.phase_deltas import between_within_phase_variance_ratio, per_phase_delta
from hybrid_adapter.synthetic import SyntheticConfig, generate_h7_batch

N_CELLS = 600
N_GENES = 120
PHASES = ("G1", "S", "G2M")


def make_line(rng: np.random.Generator, name: str, composition: np.ndarray, phase_centers: np.ndarray):
    """Artificial control cells with phase-structured expression.

    Phase programs (centers) are SHARED across lines — that is exactly H7
    claim 6 (the same subpopulation states exist in every line); only the
    compositions differ.
    """
    phase_assign = rng.choice(len(PHASES), size=N_CELLS, p=composition)
    expr = phase_centers[phase_assign] + rng.normal(0, 0.2, size=(N_CELLS, N_GENES))
    phases = np.array([PHASES[i] for i in phase_assign])
    return expr, phases


def main() -> None:
    rng = np.random.default_rng(0)
    gene_names = [f"Gene{i:04d}" for i in range(N_GENES)]
    # Ensure marker genes exist so the bundled Seurat sets score something.
    for i, g in enumerate(gene_names[:20]):
        gene_names[i] = ["PCNA", "MCM2", "TYMS", "FEN1", "MCM4", "RRM1", "UNG",
                         "GINS2", "MCM5", "MCM6", "CDK1", "NUSAP1", "UBE2C",
                         "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2",
                         "CKS1B"][i]

    compositions = {
        "lineA": np.array([0.5, 0.3, 0.2]),
        "lineB": np.array([0.4, 0.35, 0.25]),
        "lineC": np.array([0.2, 0.3, 0.5]),
    }
    shared_centers = rng.normal(0.0, 1.0, size=(len(PHASES), N_GENES))
    lines = {name: make_line(rng, name, comp, shared_centers)
             for name, comp in compositions.items()}

    # ---------------- Phase 1 ----------------
    print("=== Phase 1: subpopulation existence, homogeneity, universality ===")
    for name, (expr, phases) in lines.items():
        scored, scores = score_cell_cycle_phases(expr, gene_names)
        sil = phase_silhouette(expr, scored)
        ratios = homogeneity_variance_ratios(expr, scored)
        ok = homogeneity_passes(ratios)
        ratios_str = ", ".join(f"{k}: {v['ratio']:.2f}" for k, v in ratios.items())
        print(f"  {name}: silhouette={sil['overall']:.3f}, homogeneity_pass={ok}, "
              f"ratios={{{ratios_str}}}")

    scored_lines = {name: score_cell_cycle_phases(expr, gene_names)[0]
                    for name, (expr, phases) in lines.items()}
    universality = universality_summary(
        {name: (lines[name][0], scored_lines[name]) for name in lines}
    )
    for name, res in universality.items():
        print(f"  transfer -> {name}: accuracy={res['accuracy']:.3f} vs chance={res['chance']:.3f}, "
              f"missing_phases={res.get('missing_phases', [])}")

    # ---------------- Phase 2 ----------------
    print("\n=== Phase 2: response magnitude, decomposition, identity gate ===")
    h7 = generate_h7_batch(SyntheticConfig(n_phases=3, planted_alpha=(0.8, 0.5, 0.2),
                                           n_genes=512))
    alpha_true = h7["ground_truth_alpha"].numpy()
    print(f"  planted alpha per phase: {np.round(alpha_true, 2)}")

    # Per-phase deltas from the synthetic per-phase truth.
    phase_deltas = {f"phase{k}": h7["phase_stat_effects"][0, k].numpy() - h7["phase_stat_effects"][0].numpy().mean(axis=0)
                    for k in range(3)}
    ratio = between_within_phase_variance_ratio(phase_deltas)
    print(f"  between/within-phase variance ratio of deltas: {ratio:.2f} (large supports claim 3)")

    # Shared vs per-line alpha (two artificial 'lines' = two batch rows).
    deltas = {f"line{i}": h7["phase_stat_effects"][i, 0].numpy() for i in range(2)}
    comparison = compare_shared_vs_per_line_alpha(deltas)
    print(f"  shared alpha={comparison['shared_alpha']:.3f}, "
          f"per-line={[round(v, 3) for v in comparison['per_line_alpha'].values()]}, "
          f"homogeneity p (diagnostic)={comparison['p_value']:.3f}, "
          f"claim5_supported={comparison['claim5_supported']}")

    # CompSim head-to-head on synthetic compositions.
    p_a, p_c = compositions["lineA"], compositions["lineC"]
    print(f"  CompSim(lineA, lineC) = {compsim(p_a, p_c):.3f} "
          f"(TV = {total_variation_distance(p_a, p_c):.3f})")

    # ---------------- Phase 3: ablation ----------------
    print("\n=== Phase 3: M0-M4 ablation on synthetic H7 data ===")
    from hybrid_adapter import TrainConfig, make_h7_loaders
    train_loader, val_loader = make_h7_loaders(
        SyntheticConfig(n_phases=3, planted_alpha=(0.8, 0.5, 0.2), n_genes=64),
        n_train_batches=4, n_val_batches=2,
    )
    results = run_ablation(train_loader, val_loader, n_phases=3,
                           train_config=TrainConfig(max_epochs=5, patience=2))
    for variant in ("M0", "M1", "M2", "M3", "M4"):
        if variant in results:
            print(f"  {variant}: best_val_loss={results[variant]['best_val_loss']:.4f}")
    ordering = results["_ordering"]
    print(f"  observed ordering (best -> worst): {ordering}")
    print(f"  interpretation: {interpret_ordering(ordering)}")

    print("\nH7 synthetic validation pipeline completed. (Artificial data only — "
          "this demonstrates code correctness, not biological validity.)")


if __name__ == "__main__":
    main()
