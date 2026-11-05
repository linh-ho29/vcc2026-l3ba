"""Shared I/O, provenance, and exact upstream model construction."""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch

GEARS_COMMIT = "f374e43e197b295016d80395d7a54ddb81cc6769"
ROOT = Path(__file__).resolve().parent


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False))


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_seed(label, seed=42):
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "little")


def seed_all(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def metadata(h5ad):
    with h5py.File(h5ad, "r") as handle:
        return ad.io.read_elem(handle["obs"]), ad.io.read_elem(handle["var"])


def matrix_cache(h5ad, directory):
    """Read CSR arrays once; subsequent stages memory-map the immutable arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5ad, "r") as handle:
        shape = tuple(handle["X"].attrs["shape"])
        for key in ("data", "indices", "indptr"):
            path = directory / f"X_{key}.npy"
            dataset = handle[f"X/{key}"]
            if not path.exists():
                array = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dataset.dtype, shape=dataset.shape,
                )
                for start in range(0, len(dataset), 4_000_000):
                    array[start:start+4_000_000] = dataset[start:start+4_000_000]
                array.flush()
                del array
        return sp.csr_matrix((
            np.load(directory / "X_data.npy", mmap_mode="r"),
            np.load(directory / "X_indices.npy", mmap_mode="r"),
            np.load(directory / "X_indptr.npy", mmap_mode="r"),
        ), shape=shape, copy=False)


def load_prepared(directory, load_x=True):
    directory = Path(directory)
    meta = json.loads((directory / "dataset_manifest.json").read_text())
    obs, var = metadata(meta["h5ad"])
    roles = np.load(directory / "cell_roles.npz", allow_pickle=False)
    graph = torch.load(directory / "graphs.pt", map_location="cpu", weights_only=False)
    x = matrix_cache(meta["h5ad"], meta["matrix_cache"]) if load_x else None
    return meta, obs, var, roles, graph, x


def build_model(graph):
    from gears.model import GEARS_Model
    return GEARS_Model(graph["model_config"])


def publish(source, destination):
    """Copy only a completed file; never append to an object-backed mount."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
