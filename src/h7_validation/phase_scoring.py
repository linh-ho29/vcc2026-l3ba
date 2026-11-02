"""Phase 1 — cell-cycle phase scoring and cluster discreteness.

Scores control cells for cell-cycle phase using Seurat-style marker-gene
scoring (bundled Tirosh/Seurat gene sets, user-overridable), and quantifies
whether phase assignments form discrete clusters in expression space
(silhouette score per phase, per line).

Claims tested: 1 (subpopulation existence) and 6 (universality, together with
:mod:`h7_validation.transfer_classifier`).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Bundled canonical cell-cycle marker sets (Seurat CellCycleScoring gene lists,
# Tirosh et al. 2016). Users may override via config; these are the defaults.
S_GENE_SET: Tuple[str, ...] = (
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "POLE1", "MCM7", "HELLS", "RRM2", "GMNN", "CASP8AP2",
    "WDR76", "SLBP", "UBR7", "ATAD2", "POLA2", "CDC45", "CLSPN", "E2F8", "NASP",
)
G2M_GENE_SET: Tuple[str, ...] = (
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "FAM64A", "SMC4", "CCNB2",
    "CKAP2", "MSH2", "CKAP2L", "FAM72D", "CDCA3", "CDC20", "ANLN", "LBR", "CENPE",
    "TACC3", "GTSE1", "HJURP", "CDCA8", "TTK", "RFC4", "NUF2", "CDC6", "CCNB1",
    "CDC25C", "ANP32E", "TUBB4B", "CKAP5", "CENPW", "UBE2S", "CTCF", "AURKB",
)

PHASE_NAMES: Tuple[str, ...] = ("G1", "S", "G2M")


def score_cell_cycle_phases(
    expression: np.ndarray,
    gene_names: Iterable[str],
    s_genes: Optional[Iterable[str]] = None,
    g2m_genes: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Seurat-style phase scoring on log-normalised expression.

    Parameters
    ----------
    expression:
        ``[n_cells, n_genes]`` log-normalised (e.g. log1p-CPM) control-cell matrix.
    gene_names:
        Gene symbols matching the columns of ``expression``.
    s_genes, g2m_genes:
        Optional marker overrides; defaults are the bundled Seurat sets.

    Returns
    -------
    phases:
        ``[n_cells]`` array of phase labels ("G1", "S", "G2M").
    scores:
        Dict with per-cell S.score and G2M.score arrays.
    """
    genes = list(gene_names)
    gene_index = {g: i for i, g in enumerate(genes)}
    s_set = list(s_genes) if s_genes is not None else list(S_GENE_SET)
    g2m_set = list(g2m_genes) if g2m_genes is not None else list(G2M_GENE_SET)

    def _mean_z(marker_set: List[str]) -> np.ndarray:
        idx = [gene_index[g] for g in marker_set if g in gene_index]
        if not idx:
            raise ValueError(
                "None of the provided marker genes were found in the expression "
                "matrix; check gene-name capitalisation and the gene axis."
            )
        block = expression[:, idx]
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return ((block - mu) / sd).mean(axis=1)

    s_score = _mean_z(s_set)
    g2m_score = _mean_z(g2m_set)

    phases = np.empty(expression.shape[0], dtype=object)
    phases[:] = "G1"
    phases[(s_score > g2m_score) & (s_score > 0)] = "S"
    phases[(g2m_score > s_score) & (g2m_score > 0)] = "G2M"
    return phases, {"S.score": s_score, "G2M.score": g2m_score}


def phase_silhouette(
    expression: np.ndarray,
    phases: np.ndarray,
    metric: str = "euclidean",
    max_cells: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Mean silhouette score of phase labels, overall and per phase.

    Subsamples to ``max_cells`` for tractability. A positive, clearly nonzero
    silhouette supports claim 1 (phases form discrete clusters).
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    rng = np.random.default_rng(seed)
    n = expression.shape[0]
    if n > max_cells:
        idx = rng.choice(n, size=max_cells, replace=False)
        expression = expression[idx]
        phases = phases[idx]

    labels = np.asarray(phases)
    unique = sorted(set(labels))
    if len(unique) < 2:
        return {"overall": float("nan"), **{u: float("nan") for u in unique}}

    overall = float(silhouette_score(expression, labels, metric=metric))
    samples = silhouette_samples(expression, labels, metric=metric)
    per_phase = {u: float(samples[labels == u].mean()) for u in unique}
    return {"overall": overall, **per_phase}


def phase_proportions(
    phases: np.ndarray, phase_order: Optional[List[str]] = None
) -> np.ndarray:
    """Phase-proportion vector p_c(phase) for one line."""
    order = list(phase_order) if phase_order is not None else list(PHASE_NAMES)
    counts = np.array([(phases == p).sum() for p in order], dtype=np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("Cannot compute phase proportions for an empty set of cells.")
    return counts / total
