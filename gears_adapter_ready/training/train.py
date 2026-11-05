"""Official GEARS CPU training with lazy sparse inputs and complete restart state."""
import argparse
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import psutil
import torch
from torch_geometric.data import Batch, Data

from common import (build_model, digest, load_prepared, publish, rss_gb, seed_all,
                    timestamp, write_json)
from training.checkpoint_utils import (load_checkpoint, restore_rng, rng_state,
                                       save_checkpoint, state_dict)


class CellPairs:
    """Fixed control pairing as in PertData; only fitting controls may be inputs."""
    def __init__(self, obs, var, roles, x, targets):
        self.x = x
        self.conditions = obs.condition.astype(str).to_numpy()
        self.targets = {g: i for i, g in enumerate(targets)}
        self.fit = roles["fit"]
        self.ctrl = roles["fit_control"]
        rng = np.random.RandomState(42)
        self.pairs = rng.choice(self.ctrl, size=len(obs), replace=True)
        self.pairs[self.ctrl] = self.ctrl
        self.g = len(var)

    def batch(self, rows, include_truth=True, de=None):
        rows = np.asarray(rows)
        inputs = self.x[self.pairs[rows]].toarray()
        truths = self.x[rows].toarray() if include_truth else None
        objects = []
        for i, row in enumerate(rows):
            condition = self.conditions[row]
            indices = [self.targets[g] for g in condition.split("+") if g != "ctrl"]
            values = dict(x=torch.from_numpy(inputs[i, :, None]),
                          pert_idx=indices if indices else [-1], pert=condition)
            if include_truth:
                values["y"] = torch.from_numpy(truths[i:i+1])
            if de is not None:
                values["de_idx"] = de.get(condition, [-1])
            objects.append(Data(**values))
        return Batch.from_data_list(objects)


def validation(model, pairs, rows, de, batch_size=16):
    model.eval()
    sums, means, n = {}, {}, {}
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start+batch_size]
            batch = pairs.batch(selected, de=de)
            pred = model(batch).numpy()
            truth = batch.y.numpy()
            for condition in set(batch.pert):
                mask = np.asarray(batch.pert) == condition
                if condition not in sums:
                    sums[condition] = np.zeros(pairs.g, np.float64)
                    means[condition] = np.zeros(pairs.g, np.float64)
                    n[condition] = 0
                sums[condition] += pred[mask].sum(axis=0, dtype=np.float64)
                means[condition] += truth[mask].sum(axis=0, dtype=np.float64)
                n[condition] += int(mask.sum())
    per_condition = {
        p: dict(mse=float(np.mean((sums[p]/n[p]-means[p]/n[p])**2)),
                mse_de=float(np.mean((sums[p]/n[p]-means[p]/n[p])[de[p]]**2)),
                n_cells=n[p])
        for p in sums
    }
    return dict(
        mse=float(np.mean([r["mse"] for r in per_condition.values()])),
        mse_de=float(np.mean([r["mse_de"] for r in per_condition.values()])),
        n_targets=len(per_condition), seconds=time.time()-started,
        per_target=per_condition,
    )


def make_training(prepared, threads):
    # Import upstream before setting seed: GEARS has an import-time manual_seed(0).
    from gears.utils import loss_fct
    meta, obs, var, roles, graph, x = load_prepared(prepared)
    seed_all(42, threads)
    model = build_model(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    targets = pd.read_csv(Path(prepared)/"target_index.csv").gene_symbol.to_numpy()
    meta["frozen_split"] = json.loads((Path(prepared)/"split.json").read_text())
    meta["cell_role_rows"] = {k: roles[k].astype(np.int32) for k in roles.files if k != "roles"}
    meta["gene_ids"] = var.index.astype(str).tolist()
    meta["gene_symbols"] = var.gene_name.astype(str).tolist()
    meta["target_symbols"] = targets.tolist()
    pairs = CellPairs(obs, var, roles, x, targets)
    filters = json.loads((Path(prepared)/"training_nonzero.json").read_text())
    ctrl = torch.from_numpy(np.load(Path(prepared)/"training_statistics.npz")["ctrl_mean"])
    de = json.loads((Path(prepared)/"validation_de.json").read_text())
    return meta, graph, roles, model, optimizer, scheduler, pairs, filters, ctrl, de, loss_fct


def training_step(model, optimizer, batch, loss_fct, filters, ctrl):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch)
    loss = loss_fct(prediction, batch.y, batch.pert, ctrl=ctrl,
                    dict_filter=filters, direction_lambda=0.1)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    loss.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("Nonfinite gradient")
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def benchmark(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    context = make_training(args.prepared, args.threads)
    meta, graph, roles, model, opt, scheduler, pairs, filters, ctrl, de, loss_fct = context
    rng = np.random.default_rng(42)
    rows = rng.permutation(roles["fit"])
    initial = copy.deepcopy(state_dict(model, opt, scheduler, epoch=0))
    timing = []
    cpu_start = psutil.Process().cpu_times()
    for i in range(100):
        batch = pairs.batch(rows[i*args.batch_size:(i+1)*args.batch_size])
        start = time.perf_counter()
        loss = training_step(model, opt, batch, loss_fct, filters, ctrl)
        timing.append(time.perf_counter()-start)
        if i % 25 == 0:
            print("BENCHMARK", i, "loss", loss, "seconds", timing[-1],
                  "RSS GiB", rss_gb(), flush=True)
    save_checkpoint(out / "smoke_resume.pt",
                    state_dict(model, opt, scheduler, epoch=0, cursor=100))
    second_batch = pairs.batch(rows[100*args.batch_size:101*args.batch_size])
    loss_a = training_step(model, opt, second_batch, loss_fct, filters, ctrl)
    parameters_a = copy.deepcopy(model.state_dict())
    load_checkpoint(out / "smoke_resume.pt", model, opt, scheduler)
    loss_b = training_step(model, opt, second_batch, loss_fct, filters, ctrl)
    error = max(float((parameters_a[k]-model.state_dict()[k]).abs().max())
                for k in parameters_a)
    assert loss_a == loss_b and error == 0, (loss_a, loss_b, error)
    # Check the actual upstream Data API against its direct eager construction.
    row = rows[0]
    explicit = [
        Data(x=torch.tensor(pairs.x[pairs.pairs[r]].toarray().T),
             y=torch.tensor(pairs.x[r].toarray()), pert_idx=[pairs.targets[pairs.conditions[r].split("+")[0]]],
             pert=pairs.conditions[r])
        for r in rows[:args.batch_size] if pairs.conditions[r] != "ctrl"
    ]
    model.eval()
    non_control = [r for r in rows[:args.batch_size] if pairs.conditions[r] != "ctrl"]
    with torch.no_grad():
        eager = model(Batch.from_data_list(explicit))
        lazy = model(pairs.batch(non_control))
    assert torch.equal(eager, lazy)
    trial = validation(model, pairs, roles["validation"][:320], de)
    # Reset all benchmark changes; save the authentic training initialization.
    model.load_state_dict(initial["model"])
    opt.load_state_dict(initial["optimizer"])
    scheduler.load_state_dict(initial["scheduler"])
    restore_rng(initial["rng"])
    save_checkpoint(out / "initial_state.pt", state_dict(
        model, opt, scheduler, epoch=0, cursor=0, graph=graph, dataset=meta,
        batch_size=args.batch_size, threads=args.threads, global_step=0,
    ))
    steady = np.asarray(timing[10:])
    n_batches = len(roles["fit"]) // args.batch_size
    report = dict(
        created=timestamp(), batches=100, warmup_batches=10,
        batch_size=args.batch_size, threads=args.threads,
        median_batch_seconds=float(np.median(steady)),
        mean_batch_seconds=float(steady.mean()),
        p90_batch_seconds=float(np.quantile(steady, .9)),
        train_batches_per_epoch=n_batches,
        projected_eight_epoch_training_hours=float(steady.mean()*n_batches*8/3600),
        rss_gib=rss_gb(), peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        resume_next_loss_equal=loss_a == loss_b, resume_parameter_max_error=error,
        eager_lazy_prediction_equal=True, validation_smoke=trial,
        parameters=sum(p.numel() for p in model.parameters()),
        cpu_seconds=psutil.Process().cpu_times().user+psutil.Process().cpu_times().system
                    -cpu_start.user-cpu_start.system,
        timing_seconds=timing, dataset=meta["dataset"], training_cell_cap=meta["training_cell_cap"],
    )
    write_json(out / f"benchmark_t{args.threads}.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["timing_seconds", "validation_smoke"]}, indent=2), flush=True)


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    meta, graph, roles, model, opt, scheduler, pairs, filters, ctrl, de, loss_fct = make_training(
        args.prepared, args.threads)
    state = load_checkpoint(args.resume, model, opt, scheduler)
    assert state["dataset"]["split_sha256"] == meta["split_sha256"]
    assert state["dataset"]["fit_row_sha256"] == meta["fit_row_sha256"]
    assert state["batch_size"] == args.batch_size
    epoch, cursor = state["epoch"], state.get("cursor", 0)
    global_step = state.get("global_step", 0)
    best_val = state.get("best_validation_mse_de", float("inf"))
    best_epoch = state.get("best_epoch", 0)
    history = state.get("history", [])
    elapsed = state.get("elapsed_training_seconds", 0.0)
    started, last_checkpoint = time.time(), time.time()
    permutation = state.get("permutation")
    generator = torch.Generator().manual_seed(42)
    if "permutation_rng" in state:
        generator.set_state(state["permutation_rng"])
    # Logging occurs locally; copy closed, complete snapshots to output storage.
    logfile = out / "training_log.jsonl"
    process = psutil.Process()
    if args.checkpoint_seconds <= 0:
        raise ValueError("Recovery checkpoint interval must be positive")
    attempt_id = args.attempt_id or f"pid_{process.pid}"
    attempt_record = dict(
        attempt_id=attempt_id, started_at=timestamp(), pid=process.pid,
        resumed_from=str(args.resume), completed_epochs=epoch,
        current_epoch=epoch+1, completed_batches=cursor, global_step=global_step,
        checkpoint_seconds=args.checkpoint_seconds, threads=args.threads,
        batch_size=args.batch_size,
    )
    attempt_path = out / "attempts" / f"{attempt_id}.json"
    write_json(attempt_path, attempt_record)
    if args.publish_dir:
        publish(attempt_path, Path(args.publish_dir).parent/"reports"/attempt_path.name)
    print("TRAINING_ATTEMPT_STARTED", json.dumps(attempt_record), flush=True)
    def checkpoint(name, finished_epoch, position):
        current = state_dict(
            model, opt, scheduler, epoch=finished_epoch, cursor=position,
            global_step=global_step, permutation=permutation,
            permutation_rng=generator.get_state(),
            best_validation_mse_de=best_val, best_epoch=best_epoch,
            history=history, elapsed_training_seconds=elapsed+time.time()-started,
            graph=graph, dataset=meta, batch_size=args.batch_size,
            threads=args.threads,
        )
        return save_checkpoint(out/"checkpoints"/name, current, args.publish_dir)
    while epoch < 8:
        epoch_start = time.time()
        if permutation is None:
            permutation = roles["fit"][torch.randperm(len(roles["fit"]), generator=generator).numpy()]
        n_batches = len(permutation)//args.batch_size
        loss_sum = 0.0
        steps_this_chunk = 0
        for batch_index in range(cursor, n_batches):
            rows = permutation[batch_index*args.batch_size:(batch_index+1)*args.batch_size]
            step_start = time.time()
            batch = pairs.batch(rows)
            loss = training_step(model, opt, batch, loss_fct, filters, ctrl)
            global_step += 1
            steps_this_chunk += 1
            loss_sum += loss
            cursor = batch_index+1
            if global_step % 50 == 0:
                record = dict(
                    attempt_id=attempt_id,
                    time=timestamp(), epoch=epoch+1, batch=cursor, total_batches=n_batches,
                    global_step=global_step, loss=loss, step_seconds=time.time()-step_start,
                    elapsed_training_seconds=elapsed+time.time()-started,
                    rss_gib=rss_gb(), cpu_seconds=sum(process.cpu_times()[:2]),
                    learning_rate=opt.param_groups[0]["lr"],
                )
                with logfile.open("a") as handle:
                    handle.write(json.dumps(record)+"\n")
                print(json.dumps(record), flush=True)
            if time.time()-last_checkpoint >= args.checkpoint_seconds:
                checkpoint("recovery.pt", epoch, cursor)
                last_checkpoint = time.time()
                if args.publish_dir and logfile.exists():
                    publish(logfile, Path(args.publish_dir).parent/"reports"/logfile.name)
            if args.chunk_minutes and time.time()-started > args.chunk_minutes*60:
                checkpoint("recovery.pt", epoch, cursor)
                write_json(out/"progress.json", dict(
                    status="healthy_chunk_checkpoint", completed_epochs=epoch,
                    current_epoch=epoch+1, cursor=cursor, global_step=global_step,
                    elapsed_training_seconds=elapsed+time.time()-started))
                print("HEALTHY_CHUNK_COMPLETED", flush=True)
                return
        scheduler.step()
        val = validation(model, pairs, roles["validation"], de, 16)
        if val["mse_de"] < best_val:
            best_val, best_epoch = val["mse_de"], epoch+1
        record = dict(epoch=epoch+1, validation_mse=val["mse"],
                      validation_mse_de=val["mse_de"], validation_seconds=val["seconds"],
                      train_loss_chunk_mean=loss_sum/max(steps_this_chunk,1),
                      epoch_seconds_this_chunk=time.time()-epoch_start,
                      elapsed_training_seconds=elapsed+time.time()-started,
                      global_step=global_step, rss_gib=rss_gb(),
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                      fitting_cells_per_epoch=n_batches*args.batch_size,
                      dropped_last_cells=len(permutation)%args.batch_size)
        history.append(record)
        epoch += 1
        cursor, permutation = 0, None
        checkpoint(f"epoch_{epoch:02d}.pt", epoch, 0)
        write_json(out/"validation"/f"epoch_{epoch:02d}.json", val)
        write_json(out/"progress.json", dict(
            status="training" if epoch < 8 else "trained_eight_epochs",
            completed_epochs=epoch, best_epoch=best_epoch, history=history))
        if args.publish_dir:
            publish(out/"progress.json", Path(args.publish_dir).parent/"reports"/"training_progress.json")
            publish(out/"validation"/f"epoch_{epoch:02d}.json",
                    Path(args.publish_dir).parent/"reports"/f"validation_epoch_{epoch:02d}.json")
            if logfile.exists():
                publish(logfile, Path(args.publish_dir).parent/"reports"/logfile.name)
        print("EPOCH_COMPLETE", json.dumps(record), flush=True)
    print("TRAINING_COMPLETE_EIGHT_EPOCHS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "train"])
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume")
    parser.add_argument("--publish-dir")
    parser.add_argument("--chunk-minutes", type=int, default=100)
    parser.add_argument("--checkpoint-seconds", type=int, default=900)
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark(args)
    else:
        assert args.resume
        train(args)
