"""Record only scientific software/runtime metadata, never credentials."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import GEARS_COMMIT,write_json


def capture(output):
    packages=[
        "torch","numpy","pandas","scipy","scanpy","anndata","h5py",
        "torch-geometric","dcor","networkx","scikit-learn","tqdm","psutil",
        "requests","numba","llvmlite","joblib","threadpoolctl","umap-learn",
        "pynndescent","seaborn","statsmodels","matplotlib","xxhash","cell-gears",
    ]
    versions={p:importlib.metadata.version(p) for p in packages}
    write_json(output,dict(
        python=platform.python_version(),platform=platform.platform(),
        architecture=platform.machine(),packages=versions,gears_commit=GEARS_COMMIT,
        cpu_only=True,training_threads=4,python_hash_seed=42,
        paid_resources_requested=False,
        reproducibility_scope="Same software versions, CPU operation/threading, fixed seed and frozen input/split/graph hashes",
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    capture(parser.parse_args().output)
