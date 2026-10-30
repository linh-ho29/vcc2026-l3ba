#!/usr/bin/env python3
"""Resumable, sequential experiment command runner; standard-library only.

Deliberately does NOT implement biology, model selection, or unattended code editing.
Each experiment executes an explicit argv list, not a shell string.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/experiments.json')
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    config_path = root / args.config
    cfg = json.loads(config_path.read_text())
    state_path = root / 'harness/state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'experiments': {}, 'runs': 0}
    max_runs = int(cfg.get('max_runs', 20))
    max_retries = int(cfg.get('max_retries', 2))
    consecutive_failures = 0

    for exp in cfg['experiments']:
        ident = exp['id']; record = state['experiments'].get(ident, {})
        if not exp.get('enabled', False) or record.get('status') == 'COMPLETED':
            continue
        if state['runs'] >= max_runs or consecutive_failures >= 3:
            break
        deps = exp.get('depends_on', [])
        if any(state['experiments'].get(dep, {}).get('status') != 'COMPLETED' for dep in deps):
            state['experiments'][ident] = {'status':'BLOCKED', 'reason':'unmet dependencies', 'updated_at':utc()}
            atomic_json(state_path, state)
            continue
        command = exp.get('command', [])
        if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
            state['experiments'][ident] = {'status':'BLOCKED','reason':'no valid argv command configured', 'updated_at':utc()}
            atomic_json(state_path, state)
            continue
        timeout = max(1, int(exp.get('timeout_seconds', 3600)))
        attempts = int(record.get('attempts', 0))
        if attempts > max_retries:
            continue
        logdir = root / 'harness/logs'; logdir.mkdir(parents=True, exist_ok=True)
        state['experiments'][ident] = {'status':'RUNNING','attempts':attempts+1,'started_at':utc(), 'config_sha256':sha256_file(config_path)}
        atomic_json(state_path, state)
        started = time.monotonic(); status = 'FAILED'; reason = ''; exit_code = None
        outpath = logdir / f'{ident}.attempt{attempts+1}.stdout.txt'
        errpath = logdir / f'{ident}.attempt{attempts+1}.stderr.txt'
        try:
            with outpath.open('w') as stdout, errpath.open('w') as stderr:
                # POSIX-only process group enables termination of subprocess descendants on timeout.
                proc = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                        start_new_session=(os.name == 'posix'))
                try:
                    exit_code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        import signal
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
                    reason = 'timeout'
            required = [root / p for p in exp.get('required_outputs', [])]
            if exit_code == 0 and all(p.is_file() and p.stat().st_size > 0 for p in required):
                status = 'COMPLETED'
            elif exit_code == 0:
                reason = 'required output missing or empty'
            else:
                reason = reason or f'exit code {exit_code}'
        except Exception as exc:
            reason = repr(exc)
        elapsed = round(time.monotonic() - started, 2)
        result = {'id': ident, 'status': status, 'attempts': attempts+1, 'duration_seconds': elapsed,
                  'exit_code':exit_code, 'reason': reason, 'updated_at':utc(),
                  'stdout':str(outpath.relative_to(root)), 'stderr':str(errpath.relative_to(root))}
        state['experiments'][ident] = result
        state['runs'] += 1
        atomic_json(state_path, state)
        append_jsonl(root / 'experiments/results.jsonl', result)
        consecutive_failures = consecutive_failures + 1 if status != 'COMPLETED' else 0
        # Status summary is intentionally concise: metrics/acceptance decisions belong to an independent evaluator.
        summary = ['# Current status', '', f'Last experiment: {ident}', f'Result: {status}',
                   f'Duration: {elapsed}s', f'Log: {result["stderr"]}',
                   f'Failure reason: {reason or "none"}', '',
                   'No automatic model promotion. Consult frozen evaluation metrics.']
        status_path = root / 'reports/CURRENT_STATUS.md'
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text('\n'.join(summary)+'\n')
        if status != 'COMPLETED':
            print(f'{ident}: {status} ({reason})', file=sys.stderr)
        else:
            print(f'{ident}: COMPLETED')

if __name__ == '__main__':
    main()
