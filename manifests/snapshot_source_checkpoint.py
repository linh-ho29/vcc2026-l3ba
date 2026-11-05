"""Checkpoint current project source, excluding machine caches and datasets."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from io_utils import digest, timestamp


def snapshot():
    root = Path(__file__).resolve().parents[1]
    work = Path("/workspace/gears_recovery_20261104")
    work.mkdir(parents=True,exist_ok=True)
    files = [p for p in root.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and ".git" not in p.parts
             and p.suffix != ".pyc"]
    name = "gears_recovery_source_"+str(time.time_ns())+".tar.gz"
    local = work/name
    expected = {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=digest(p))
                for p in files}
    with tarfile.open(local,"w:gz",compresslevel=6) as archive:
        for path in files:
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    destination = Path("/mnt/shared-workspace/shared/gears_rpe1_recovery_sources")/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(local,destination)
    receipt = dict(created_at=timestamp(),filename=name,sha256=digest(destination),
                   bytes=destination.stat().st_size,files=expected)
    destination.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="files"}),flush=True)


if __name__ == "__main__":
    snapshot()
