"""Stream explicitly heuristic single-cell count predictions to a local H5AD."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import digest, stable_seed, timestamp, write_json
from count_generation.reconstruct_counts import reconstruct_counts


def append_csr(group, matrix):
    """Append without AnnData's growing full-array cached-property probes."""
    matrix = sp.csr_matrix(matrix)
    rows, genes = group.attrs["shape"]
    if matrix.shape[1] != genes:
        raise ValueError("Count matrix gene axis changed")
    offset = len(group["data"])
    if offset + matrix.nnz >= np.iinfo(group["indptr"].dtype).max:
        raise OverflowError("CSR pointer integer capacity exceeded")
    for key, values in (("data", matrix.data), ("indices", matrix.indices)):
        dataset = group[key]
        start = len(dataset)
        dataset.resize((start + len(values),))
        dataset[start:] = values
    indptr = group["indptr"]
    start = len(indptr)
    indptr.resize((start + matrix.shape[0],))
    indptr[start:] = matrix.indptr[1:].astype(np.int64) + offset
    group.attrs["shape"] = (rows + matrix.shape[0], genes)


def export(matched, profiles, statistics, method, exposure, output, pilot_targets=0):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    measured = ad.read_h5ad(matched, backed="r")
    try:
        fit = measured.obs.role.to_numpy() == "fit_control"
        fitting_controls = measured[fit].X.toarray()
        assert fitting_controls.shape == (9188,5000)
        panel = np.load(profiles, allow_pickle=False)
        effects = panel["predicted_effects_mean"]
        assert measured.var["gene_id"].tolist() == panel["gene_ids"].tolist()
        assert measured.var_names.tolist() == panel["gene_symbols"].tolist()
        eligible = set(measured.obs.loc[measured.obs.role == exposure, "condition"].astype(str))
        selected = [(i,str(c)) for i,c in enumerate(panel["conditions"]) if str(c) in eligible]
        if pilot_targets:
            selected = selected[:pilot_targets]
        if not selected:
            raise ValueError("No target condition selected")
        baseline = np.load(statistics, allow_pickle=False)["statistical_effect"]
        record = dict(method=method, exposure=exposure, conditions=[c for _,c in selected],
                      cells_per_target=400, background_fraction=1e-4,
                      seed_label="heuristic_counts:{condition}", adapter_trained=False,
                      predicted_counts=True, measured_perturbation_outcomes_used_to_generate=False,
                      reference_control_source="reference_control",
                      input_control_source="fit_control",
                      gene_panel_size=5000, input_profiles_sha256=digest(profiles),
                      training_statistics_sha256=digest(statistics),
                      pilot=bool(pilot_targets))
        stem = f"{method}_{exposure}" + ("_pilot" if pilot_targets else "")
        destination = output / (stem + ".h5ad")
        temporary = output / (stem + ".partial.h5ad")
        obs_parts = []
        empty = ad.AnnData(X=sp.csr_matrix((0,5000),dtype=np.int32),
                          obs=pd.DataFrame(index=pd.Index([],dtype=str)),
                          var=measured.var.copy())
        empty.write_h5ad(temporary, compression="gzip", compression_opts=1)
        with h5py.File(temporary, "a") as handle:
            for target_number, (i, condition) in enumerate(selected):
                if method == "control_only":
                    effect = np.zeros(5000, np.float32)
                elif method == "statistical_mean_effect":
                    effect = baseline
                elif method.startswith("gears_epoch_"):
                    effect = effects[i]
                else:
                    raise ValueError(f"Unknown method: {method}")
                sample = reconstruct_counts(effect, fitting_controls, n_cells=400,
                                            seed=stable_seed("heuristic_counts:"+condition),
                                            background_fraction=1e-4)
                append_csr(handle["X"], sample["counts"])
                obs_parts.append(pd.DataFrame(
                    dict(target=[condition.replace("+ctrl","")]*400,
                         condition=[condition]*400, role=["predicted"]*400),
                    index=[f"{stem}_{condition}_{j:04d}" for j in range(400)]))
                if target_number % 25 == 0:
                    print("COUNT_TARGET", stem, target_number, len(selected),
                          f"{time.monotonic()-started:.1f}s", flush=True)
            # Prediction controls are disjoint fitting cells. The preset uses the real
            # reference control pool, not these cells, as the origin on both sides.
            controls = measured[np.flatnonzero(fit)[:64]].to_memory()
            append_csr(handle["X"], controls.X.astype(np.int32))
            control_obs = pd.DataFrame(
                dict(target=["non-targeting"]*64, condition=["ctrl"]*64, role=["fit_control"]*64),
                index=["pred_control_"+str(c) for c in controls.obs_names])
            obs_parts.append(control_obs)
            ad.io.write_elem(handle, "obs", pd.concat(obs_parts))
            ad.io.write_elem(handle, "uns", {"heuristic_count_layer": json.dumps(record)})
        temporary.replace(destination)
        with h5py.File(destination, "r") as handle:
            nnz = len(handle["X/data"])
        record.update(filename=destination.name, sha256=digest(destination),
                      cells=len(selected)*400+64, nonzero=nnz,
                      sparse_resident_bytes_estimate=nnz*8+(len(selected)*400+65)*4,
                      bytes=destination.stat().st_size, seconds=time.monotonic()-started,
                      completed_at=timestamp())
        write_json(output/(stem+"_manifest.json"), record)
        print(json.dumps(record, indent=2), flush=True)
        return record
    finally:
        measured.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["matched", "profiles", "statistics", "method", "exposure", "output"]:
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--pilot-targets",type=int,default=0)
    args=parser.parse_args()
    export(args.matched,args.profiles,args.statistics,args.method,args.exposure,args.output,args.pilot_targets)
