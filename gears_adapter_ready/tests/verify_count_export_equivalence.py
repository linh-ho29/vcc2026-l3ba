"""Verify two exports of actual count data agree exactly; no simulated inputs."""
import argparse
import json

import anndata as ad


def verify(first, second):
    a, b = ad.read_h5ad(first), ad.read_h5ad(second)
    assert a.shape == b.shape
    assert a.obs.equals(b.obs)
    assert a.var.equals(b.var)
    assert (a.X != b.X).nnz == 0
    record = dict(passed=True, comparison="exact counts and axes",
                  cells=a.n_obs, genes=a.n_vars, nonzero=int(a.X.nnz))
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    verify(args.first, args.second)
