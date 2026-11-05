"""Check optimized reads on complete existing count files, not simulated data."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
from data.csr_rows import CSRRowReader
from io_utils import write_json


def main():
    base = Path("/workspace/gears_raw_eval/full_unseen")
    observations = []
    for filename in ["gears_epoch_8_unseen_test.h5ad","measured_unseen_test.h5ad"]:
        data = ad.read_h5ad(base/filename,backed="r")
        reader = CSRRowReader(base/filename)
        try:
            labels = data.obs.target.astype(str).to_numpy()
            targets = [x for x in sorted(set(labels)) if x!="non-targeting"][:8]
            old_seconds = new_seconds = 0
            for target in targets:
                rows = np.flatnonzero(labels==target)
                start=time.monotonic()
                old=data[rows].to_memory()
                old_seconds += time.monotonic()-start
                start=time.monotonic()
                new=reader.read(rows,data.obs,data.var)
                new_seconds += time.monotonic()-start
                assert old.obs.equals(new.obs) and old.var.equals(new.var)
                assert (old.X != new.X).nnz == 0
                assert old.X.dtype == new.X.dtype
            observations.append(dict(filename=filename,targets=8,
                                     exact_counts_and_axes=True,
                                     old_read_seconds=old_seconds,new_read_seconds=new_seconds))
        finally:
            data.file.close()
            reader.close()
    result=dict(passed=True,panels=observations,no_count_regeneration=True)
    write_json("/mnt/results/gears_recovery_20261104/exact_csr_row_read_parity.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
