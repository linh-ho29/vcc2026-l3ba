"""Phase 1 — cross-line transfer of phase labels.

Claim 6 test: train a phase classifier on line A's control cells, test on
lines B, C, ... High cross-line accuracy supports the "universal subpopulation
repertoire" claim. If a phase is missing in some line, claim 6 is falsified
and the hypothesis needs revision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def train_phase_classifier(
    expression: np.ndarray,
    phases: np.ndarray,
    seed: int = 0,
    max_iter: int = 500,
):
    """Train a multinomial logistic-regression phase classifier on one line.

    Standardises features and fits logistic regression on the phase labels.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    model.fit(expression, np.asarray(phases))
    return model


def cross_line_transfer_accuracy(
    train_expression: np.ndarray,
    train_phases: np.ndarray,
    test_expression: np.ndarray,
    test_phases: np.ndarray,
    seed: int = 0,
) -> Dict[str, object]:
    """Train on one line's controls, evaluate phase-label accuracy on another.

    Returns accuracy, chance level (largest-class frequency), the set of
    phases seen in training, and the set of phases present in the test line.
    A test line missing a training phase is a claim-6 falsification signal.
    """
    model = train_phase_classifier(train_expression, train_phases, seed=seed)
    pred = model.predict(test_expression)
    accuracy = float((pred == np.asarray(test_phases)).mean())
    classes, counts = np.unique(np.asarray(test_phases), return_counts=True)
    chance = float(counts.max() / counts.sum())
    return {
        "accuracy": accuracy,
        "chance": chance,
        "train_phases": sorted(set(np.asarray(train_phases))),
        "test_phases": sorted(classes.tolist()),
        "missing_phases": sorted(set(np.asarray(train_phases)) - set(classes.tolist())),
    }


def universality_summary(
    lines: Dict[str, Tuple[np.ndarray, np.ndarray]],
    seed: int = 0,
) -> Dict[str, object]:
    """Leave-one-line-out cross-line transfer across all provided lines.

    ``lines`` maps line name -> (control_expression, phase_labels). For each
    held-out line, the classifier is trained on the union of the remaining
    lines. Claim 6 is supported when transfer accuracy is well above chance
    everywhere and no phase disappears in any line.
    """
    results: Dict[str, object] = {}
    for held_out in lines:
        train_expr_parts, train_phase_parts = [], []
        for name, (expr, ph) in lines.items():
            if name == held_out:
                continue
            train_expr_parts.append(expr)
            train_phase_parts.append(ph)
        if not train_expr_parts:
            results[held_out] = {"error": "no training lines provided"}
            continue
        results[held_out] = cross_line_transfer_accuracy(
            np.vstack(train_expr_parts),
            np.concatenate(train_phase_parts),
            lines[held_out][0],
            lines[held_out][1],
            seed=seed,
        )
    return results
