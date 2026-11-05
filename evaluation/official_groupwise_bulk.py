"""Memory-bounded official count metrics without changing the full-panel universe.

All numerical definitions come from the pinned evaluator. Only orchestration and
concatenation of independent group summaries are implemented here. This seam
must pass actual-data parity before it is used on any larger panel.
"""
import gc
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl

from cell_eval2.moments import GroupMoments
from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments
from cell_eval2.run import (
    _check_scale_limit_once, _moment_normalizations, _needed_normalizations,
    _prepare_de_cached, _validate_input_once, dispatch_anndata_metrics,
    dispatch_de_metrics, resolve_metrics, validate_pair,
)
from io_utils import digest, write_json
from data.csr_rows import CSRRowReader


def summarize_groups(source, cfg, cache):
    cache = Path(cache)/digest(source)[:20]
    cache.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha256=digest(source), config=asdict(cfg),
                    evaluator_revision="d86be3f4cd47015cd41046e9cc6a834777453594",
                    orchestration="independent official group summaries v1")
    identity_path = cache/"identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        write_json(identity_path, identity)
    data = ad.read_h5ad(source, backed="r")
    row_reader = CSRRowReader(source)
    try:
        labels = data.obs[cfg.pert_col].astype(str).to_numpy()
        targets = np.unique(labels)
        means, counts, sumsq, jackknife = [], [], [], []
        for target in targets:
            key = hashlib.sha256(str(target).encode()).hexdigest()[:20]
            path = cache/(key+".npz")
            sidecar = cache/(key+".json")
            if path.exists() and sidecar.exists():
                record = json.loads(sidecar.read_text())
                assert record["target"] == target and digest(path) == record["sha256"]
                assert record["bulk_target_sum"] == cfg.bulk_target_sum
                assert record["max_counts_per_cell"] == cfg.max_counts_per_cell
            else:
                group = row_reader.read(np.flatnonzero(labels == target),
                                        data.obs, data.var)
                _validate_input_once(group, "counts", allow_fractional=False)
                _check_scale_limit_once(group, "counts", cfg.max_counts_per_cell)
                perts, mean, moments = pseudobulk_bulk_lognorm_with_moments(
                    group, cfg.pert_col, bulk_target_sum=cfg.bulk_target_sum)
                assert perts.tolist() == [target]
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, mean=mean, count=moments.counts,
                                    sumsq=moments.sumsq, jk=moments.jk)
                temporary.replace(path)
                write_json(sidecar, dict(target=str(target), sha256=digest(path),
                                         bulk_target_sum=cfg.bulk_target_sum,
                                         input_type="counts", integer_validation=True,
                                         max_counts_per_cell=cfg.max_counts_per_cell))
                del group, moments, mean
            with np.load(path, allow_pickle=False) as values:
                means.append(values["mean"][0])
                counts.append(values["count"][0])
                sumsq.append(values["sumsq"][0])
                jackknife.append(values["jk"][0])
            gc.collect()
        moments = GroupMoments(targets, np.asarray(counts), np.asarray(sumsq),
                               jk=np.asarray(jackknife))
        return {"bulk_lognorm": (targets, np.vstack(means))}, {"bulk_lognorm": moments}
    finally:
        row_reader.close()
        data.file.close()


def compute_groupwise_metrics(predicted, real, cfg, de_pred, de_real, cache):
    assert cfg.input_type == "counts" and cfg.version == "v2"
    assert cfg.control_source == "real" and cfg.validate_input
    assert not cfg.allow_fractional_counts and not cfg.autodetect_input_type
    names, missing = resolve_metrics(cfg.metrics, version=cfg.version)
    assert not missing
    assert _needed_normalizations(names, comparator="bulk_lognorm") == ["bulk_lognorm"]
    assert _moment_normalizations(names, comparator="bulk_lognorm") == {"bulk_lognorm"}
    a, b = ad.read_h5ad(predicted, backed="r"), ad.read_h5ad(real, backed="r")
    try:
        validate_pair(a, b, pert_col=cfg.pert_col, control=cfg.control)
        assert a.var_names.equals(b.var_names)
        genes = a.var_names.to_numpy().astype(str)
        perturbations = set(a.obs[cfg.pert_col].astype(str)) - {cfg.control}
    finally:
        a.file.close()
        b.file.close()
    pred_bulks, pred_moments = summarize_groups(predicted, cfg, Path(cache)/"pred")
    real_bulks, real_moments = summarize_groups(real, cfg, Path(cache)/"real")
    prepared_de = _prepare_de_cached(
        de_pred, de_real, cfg=cfg, real_store=None, pred_store=None,
        de_real_supplied=True, de_pred_supplied=True)
    assert perturbations == set(prepared_de.perturbations)
    rows = dispatch_de_metrics(names, prepared_de, cfg)
    rows.extend(dispatch_anndata_metrics(
        names, pred_bulks, real_bulks, genes, cfg, comparator="bulk_lognorm",
        pred_moments=pred_moments, real_moments=real_moments,
        pred_bulks_full=pred_bulks, driver="official group-wise CPU orchestration"))
    return pl.DataFrame(rows, schema={
        "perturbation":pl.String, "metric":pl.String, "value":pl.Float64})
