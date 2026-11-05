"""Framework-free completed-file I/O for the isolated evaluation environment."""
import hashlib
import json
import time
import uuid
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def write_progress_snapshot(prefix, value):
    """Write immutable progress objects; never rewrite a shared checkpoint key."""
    path = Path(str(prefix)+"_"+str(time.time_ns())+"_"+uuid.uuid4().hex[:12]+".json")
    write_json(path, value)
    return path
