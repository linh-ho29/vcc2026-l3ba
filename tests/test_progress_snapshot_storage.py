"""Check immutable shared progress writes using an actual campaign record."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import digest, write_json, write_progress_snapshot


def main():
    old = Path("/mnt/shared-workspace/shared/gears_rpe1_official_de_cache/"
               "campaign_progress/1793806878778324710_4193ec4d0972e12d0d29.json")
    original_hash = digest(old)
    record = json.loads(old.read_text())
    prefix = old.parent/("immutable_progress_storage_test_"+str(time.time_ns()))
    first = write_progress_snapshot(prefix, record)
    second = write_progress_snapshot(prefix, record)
    assert first != second
    assert json.loads(first.read_text()) == json.loads(second.read_text()) == record
    assert digest(old) == original_hash
    verification = dict(passed=True, original_progress_file_unchanged=True,
                        original_sha256=original_hash,
                        snapshots=[dict(path=str(p), bytes=p.stat().st_size,
                                        sha256=digest(p)) for p in [first, second]],
                        scientific_inputs_or_settings_changed=False)
    write_json("/mnt/results/gears_recovery_20261104/immutable_progress_storage_test.json",
               verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
