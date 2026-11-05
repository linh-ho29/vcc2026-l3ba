"""Tests for extract_gears_embeddings.py's standardization and merge logic.
Synthetic data only - no real checkpoints needed."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from extract_gears_embeddings import (
    from_representations_npz,
    make_fixed_projection,
    merge_cell_lines,
    standardize_dim,
)

CELL_LINES = ["K562", "HepG2", "RPE1", "Jurkat"]


def test_standardize_dim_identity_when_already_target():
    x = np.random.randn(5, 64).astype(np.float32)
    out = standardize_dim(x, target_dim=64)
    assert np.array_equal(out, x)


def test_standardize_dim_projects_up():
    x = np.random.randn(5, 32).astype(np.float32)
    out = standardize_dim(x, target_dim=64)
    assert out.shape == (5, 64)
    assert np.isfinite(out).all()


def test_standardize_dim_truncates_down():
    x = np.random.randn(5, 128).astype(np.float32)
    out = standardize_dim(x, target_dim=64)
    assert out.shape == (5, 64)
    assert np.array_equal(out, x[:, :64])


def test_projection_is_fixed_and_reproducible():
    p1 = make_fixed_projection(32, 64, seed=42)
    p2 = make_fixed_projection(32, 64, seed=42)
    assert np.array_equal(p1, p2)
    p3 = make_fixed_projection(32, 64, seed=43)
    assert not np.array_equal(p1, p3)


def test_merge_full_coverage_gene():
    per_line = {line: {"GENE_A": np.full(4, i, dtype=np.float32)} for i, line in enumerate(CELL_LINES)}
    emb, mask = merge_cell_lines(per_line, CELL_LINES, d_pert=4)
    assert torch.equal(mask["GENE_A"], torch.ones(4))
    for i in range(4):
        assert torch.allclose(emb["GENE_A"][i], torch.full((4,), float(i)))


def test_merge_imputes_missing_with_cross_line_mean():
    per_line = {
        "K562": {"GENE_B": np.array([1.0, 1.0], dtype=np.float32)},
        "HepG2": {},
        "RPE1": {},
        "Jurkat": {"GENE_B": np.array([3.0, 3.0], dtype=np.float32)},
    }
    emb, mask = merge_cell_lines(per_line, CELL_LINES, d_pert=2)
    assert torch.equal(mask["GENE_B"], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    # HepG2 and RPE1 slots should both be imputed to mean([1,1],[3,3]) = [2,2]
    assert torch.allclose(emb["GENE_B"][1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(emb["GENE_B"][2], torch.tensor([2.0, 2.0]))
    # The real sources must be untouched.
    assert torch.allclose(emb["GENE_B"][0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(emb["GENE_B"][3], torch.tensor([3.0, 3.0]))


def test_merge_single_source_imputes_to_itself():
    per_line = {
        "K562": {}, "HepG2": {"GENE_C": np.array([5.0], dtype=np.float32)}, "RPE1": {}, "Jurkat": {},
    }
    emb, mask = merge_cell_lines(per_line, CELL_LINES, d_pert=1)
    assert torch.equal(mask["GENE_C"], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(emb["GENE_C"], torch.tensor([[5.0], [5.0], [5.0], [5.0]]))


def test_merge_gene_in_no_line_is_excluded_not_zero_filled():
    per_line = {line: {} for line in CELL_LINES}
    per_line["K562"]["GENE_D"] = np.array([1.0], dtype=np.float32)
    emb, mask = merge_cell_lines(per_line, CELL_LINES, d_pert=1)
    assert "GENE_D" in emb
    # A gene present in literally zero lines should never appear in the output at all.
    assert all("GENE_NOWHERE" not in d for d in [emb])


def test_merge_masks_when_source_absent_for_gene():
    per_line = {
        "K562": {"GENE_E": np.array([1.0], dtype=np.float32)},
        "HepG2": {"GENE_E": np.array([1.0], dtype=np.float32)},
        "RPE1": {},
        "Jurkat": {"GENE_E": np.array([1.0], dtype=np.float32)},
    }
    emb, mask = merge_cell_lines(per_line, CELL_LINES, d_pert=1)
    assert mask["GENE_E"][2].item() == 0.0


def test_from_representations_npz_aligns_by_row_order():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        npz_path = tmp / "representations.npz"
        np.savez(npz_path, target_fused=embeddings, other_array=embeddings * 0)
        index_csv = tmp / "target_index.csv"
        with open(index_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_index", "gene_symbol"])
            writer.writerow([0, "GENE_X"])
            writer.writerow([1, "GENE_Y"])
            writer.writerow([2, "GENE_Z"])

        result = from_representations_npz(npz_path, index_csv, array_key="target_fused")

        assert set(result.keys()) == {"GENE_X", "GENE_Y", "GENE_Z"}
        assert np.array_equal(result["GENE_Y"], np.array([3.0, 4.0], dtype=np.float32))


def test_from_representations_npz_raises_on_row_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz_path = tmp / "representations.npz"
        np.savez(npz_path, target_fused=np.zeros((2, 2), dtype=np.float32))
        index_csv = tmp / "target_index.csv"
        with open(index_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_index", "gene_symbol"])
            writer.writerow([0, "GENE_X"])

        try:
            from_representations_npz(npz_path, index_csv, array_key="target_fused")
            assert False, "expected ValueError on row-count mismatch"
        except ValueError:
            pass


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
