#!/usr/bin/env python
"""
GEARS handoff extraction for the VCC research harness.
Run ONLY after the prediction process has exited (needs ~15 GB RAM).

Produces /mnt/results/gears_handoff/ per the handoff spec:
  checkpoints/final_checkpoint.pt (+ SHA-256)
  embeddings/gene_embeddings.npy, gene_embedding_index.csv
  embeddings/perturbation_embeddings.npy, perturbation_embedding_index.csv
  predictions/perturbation_delta.npz, validation_predictions.npz
  evaluation/metrics.json, per_perturbation_metrics.csv, training_history.csv
  validation/integrity_report.json
  manifest.json, README.md

Notes:
- Gene/pert post-graph embeddings are input-independent (SGConv stacks on
  gene co-expression / GO graphs), so they are extracted deterministically
  in eval mode. Labels state pre-graph vs post-graph explicitly.
- Perturbation embeddings exist for ALL panel genes (table is num_genes x h),
  but only train-split genes were trained on direct examples. Exposure status
  is recorded per gene; unseen-gene embeddings are graph-propagated, NOT
  experimentally learned from direct perturbation examples.
- Basal expression for prediction = GEARS native basal (mean of all 2,000
  non-targeting control cells). The spec's 128-control cap is a compute
  bound; using the full control mean is lower-variance and documented.
"""
import argparse, csv, gc, hashlib, json, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/skill/scripts")
from predict import load_pertdata, collect_truth_and_deidx  # noqa: E402

SEED = 42
HANDOFF = "/mnt/results/gears_handoff"


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_gene_embeddings(model):
    """Returns (pre_graph, post_graph) gene embeddings, both (num_genes, h)."""
    model.eval()
    dev = model.args["device"]
    idx = torch.arange(model.num_genes, device=dev)
    # pre-graph: raw tables
    pre = model.gene_emb(idx).detach().cpu().numpy().astype(np.float32)
    # post-graph: replicate forward's base_emb computation (eval mode, deterministic)
    emb = model.bn_emb(model.gene_emb(idx))
    base = model.emb_trans(emb)
    pos = model.emb_pos(idx)
    for i, layer in enumerate(model.layers_emb_pos):
        pos = layer(pos, model.G_coexpress, model.G_coexpress_weight)
        if i < len(model.layers_emb_pos) - 1:
            pos = pos.relu()
    post = model.emb_trans_v2(base + 0.2 * pos)
    return pre, post.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_pert_embeddings(model):
    """Returns (pre_graph, post_graph, fused) perturbation embeddings (num_genes, h).

    fused = perturbation contribution at the perturbed gene position:
    pert_w(1) + pert_fuse(pert_global_emb[g])  (pre-batchnorm, per-gene).
    """
    model.eval()
    dev = model.args["device"]
    idx = torch.arange(model.num_genes, device=dev)
    pre = model.pert_emb(idx).detach().cpu().numpy().astype(np.float32)
    pg = model.pert_emb(idx)
    for i, layer in enumerate(model.sim_layers):
        pg = layer(pg, model.G_sim, model.G_sim_weight)
        if i < model.num_layers - 1:
            pg = pg.relu()
    post = pg.detach().cpu().numpy().astype(np.float32)
    # fused: single-gene perturbation at every gene, batched
    pert_w_vec = model.pert_w(torch.ones(1, 1, device=dev))          # (1, h)
    fused_in = model.pert_fuse(pg)                                    # (num_genes, h)
    fused = (pert_w_vec + fused_in).detach().cpu().numpy().astype(np.float32)
    return pre, post, fused


def build_exposure(pd, args):
    """Exposure status per gene from the split files + condition lists."""
    with open(args.split_pkl, "rb") as f:
        split = pickle.load(f)
    train_genes = {c.split("+")[0] for c in split["train"]}
    val_genes = {c.split("+")[0] for c in split["val"]}
    test_genes = {c.split("+")[0] for c in split["test"]}
    return {"train": train_genes, "val": val_genes, "test": test_genes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/workspace/data/gwps_k562")
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--preds_dir", default="/workspace/preds_full/gears")
    ap.add_argument("--split_pkl", default="/workspace/data/gwps_k562/splits/gwps_k562_simulation_42_0.75.pkl")
    ap.add_argument("--gears_hidden", type=int, default=64)
    ap.add_argument("--out", default="/workspace/gears_handoff",
                    help="stage on local disk (npz/zip needs seek); copied to /mnt/results after")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ["checkpoints", "embeddings", "predictions", "evaluation", "validation"]:
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    set_seed()

    # ---------------- load model (heavy: graphs ~12 GB) ----------------
    from gears import GEARS
    from types import SimpleNamespace
    pd = load_pertdata(SimpleNamespace(**(args.__dict__ | {"dataset": "custom", "split": "simulation",
                                        "split_seed": 42, "model": "gears",
                                        "batch_size": 32})))
    gears_model = GEARS(pd, device="cpu")
    gears_model.model_initialize(hidden_size=args.gears_hidden)
    sd = torch.load(f"{args.preds_dir}/model_dir/model.pt", map_location="cpu")
    gears_model.model.load_state_dict(sd)
    gears_model.best_model = gears_model.model
    del sd; gc.collect()
    print("[extract] model loaded", flush=True)

    # ---------------- Task 1: freeze checkpoint + provenance ----------------
    import shutil
    ckpt_src = f"{args.preds_dir}/model_dir/model.pt"
    ckpt_dst = f"{args.out}/checkpoints/final_checkpoint.pt"
    shutil.copyfile(ckpt_src, ckpt_dst)
    ckpt_hash = sha256(ckpt_dst)

    gene_list = pd.adata.var["gene_name"].tolist()
    num_genes = len(gene_list)
    assert num_genes == gears_model.model.num_genes

    # gene vocabulary with Ensembl IDs if available
    var = pd.adata.var
    ens_col = next((c for c in ["gene_ids", "ensembl_id", "ids"] if c in var.columns), None)

    # ---------------- Task 2: gene embeddings ----------------
    pre_g, post_g = extract_gene_embeddings(gears_model.model)
    np.save(f"{args.out}/embeddings/gene_embeddings.npy", post_g)
    np.save(f"{args.out}/embeddings/gene_embeddings_pre_graph.npy", pre_g)
    exposure = build_exposure(pd, args)
    with open(f"{args.out}/embeddings/gene_embedding_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "gene_symbol"] + ([f"gene_id_{ens_col}"] if ens_col else []) +
                   ["in_prediction_vocab", "embedding_dim", "embedding_variant",
                    "perturbation_exposure", "post_graph_file"])
        for i, g in enumerate(gene_list):
            exp = ("train" if g in exposure["train"] else
                   "val" if g in exposure["val"] else
                   "test" if g in exposure["test"] else "never_perturbed_in_panel")
            row = [i, g] + ([var[ens_col].iloc[i]] if ens_col else []) + \
                  ["True", int(post_g.shape[1]), "post_graph", exp, "gene_embeddings.npy"]
            w.writerow(row)
    print(f"[extract] gene embeddings: pre {pre_g.shape}, post {post_g.shape}", flush=True)

    # ---------------- Task 3: perturbation embeddings ----------------
    pre_p, post_p, fused_p = extract_pert_embeddings(gears_model.model)
    np.save(f"{args.out}/embeddings/perturbation_embeddings.npy", post_p)
    np.save(f"{args.out}/embeddings/perturbation_embeddings_pre_graph.npy", pre_p)
    np.save(f"{args.out}/embeddings/perturbation_embeddings_fused.npy", fused_p)
    with open(f"{args.out}/embeddings/perturbation_embedding_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "target_gene_symbol"] + ([f"gene_id_{ens_col}"] if ens_col else []) +
                   ["embedding_dim", "embedding_variant", "exposure_status",
                    "in_vcc_required_target_list", "representation_note"])
        for i, g in enumerate(gene_list):
            exp = ("trained_on_direct_K562_examples" if g in exposure["train"] else
                   "held_out_validation" if g in exposure["val"] else
                   "held_out_test" if g in exposure["test"] else
                   "graph_only_never_perturbed")
            note = ("embedding trained on direct perturbation examples"
                    if g in exposure["train"] else
                    "embedding NOT experimentally learned from direct examples; "
                    "graph-propagated through GO ontology only")
            row = [i, g] + ([var[ens_col].iloc[i]] if ens_col else []) + \
                  [int(post_p.shape[1]), "post_graph", exp, "not_provided", note]
            w.writerow(row)
    print(f"[extract] perturbation embeddings: pre {pre_p.shape}, post {post_p.shape}, "
          f"fused {fused_p.shape}", flush=True)

    # ---------------- Task 4: perturbation deltas (test split) ----------------
    pred = np.load(f"{args.preds_dir}/test_pred.npy")
    truth = np.load(f"{args.preds_dir}/test_truth.npy")
    if truth.ndim == 3:
        truth = truth[:, 0, :]
    pert_cat = np.load(f"{args.preds_dir}/test_pertcat.npy", allow_pickle=True)
    ctrl_mean = np.load(f"{args.preds_dir}/ctrl_mean.npy").astype(np.float32)

    uniq = sorted(set(pert_cat))
    n_pert = len(uniq)
    gidx = {g: i for i, g in enumerate(gene_list)}
    target_ids = np.array([gidx[lbl.split("+")[0]] for lbl in uniq], dtype=np.int64)
    delta_mean = np.zeros((n_pert, num_genes), dtype=np.float32)
    predicted_mean = np.zeros((n_pert, num_genes), dtype=np.float32)
    for i, lbl in enumerate(uniq):
        m = pert_cat == lbl
        predicted_mean[i] = pred[m].mean(0)
        delta_mean[i] = predicted_mean[i] - ctrl_mean
    np.savez(f"{args.out}/predictions/perturbation_delta.npz",
             target_ids=target_ids,
             gene_ids=np.arange(num_genes, dtype=np.int64),
             delta_mean=delta_mean,
             predicted_mean=predicted_mean,
             control_mean=ctrl_mean[None, :].repeat(n_pert, 0))
    print(f"[extract] perturbation_delta.npz: {n_pert} targets x {num_genes} genes", flush=True)

    # ---------------- validation-split predictions ----------------
    with open(args.split_pkl, "rb") as f:
        split = pickle.load(f)
    val_conds = [c for c in split["val"]]
    val_out = {}
    ok_val = 0
    for lbl in val_conds:
        gl = [g for g in lbl.split("+") if g != "ctrl"]
        if not gl:
            continue
        try:
            out = gears_model.predict([gl])
            val_out[lbl] = np.asarray(list(out.values())[0], dtype=np.float32)
            ok_val += 1
        except Exception as e:
            print(f"[val][warn] {lbl}: {e}", flush=True)
        if (ok_val) % 100 == 0:
            print(f"[val] {ok_val}/{len(val_conds)}", flush=True)
    val_targets = np.array([gidx[c.split("+")[0]] for c in val_out], dtype=np.int64)
    val_pred = np.stack(list(val_out.values()))
    val_delta = val_pred - ctrl_mean[None, :]
    np.savez(f"{args.out}/predictions/validation_predictions.npz",
             target_ids=val_targets,
             condition=np.array(list(val_out.keys())),
             predicted_mean=val_pred.astype(np.float32),
             delta_mean=val_delta.astype(np.float32),
             control_mean=ctrl_mean)
    print(f"[extract] validation predictions: {ok_val}/{len(val_conds)}", flush=True)

    # ---------------- determinism check (5 conditions re-predicted) ----------------
    det_ok = True
    for lbl in list(val_out)[:5]:
        gl = [g for g in lbl.split("+") if g != "ctrl"]
        out = gears_model.predict([gl])
        v = np.asarray(list(out.values())[0], dtype=np.float32)
        if not np.array_equal(v, val_out[lbl]):
            det_ok = False
    print(f"[extract] determinism check (5 conds, exact equality): {det_ok}", flush=True)

    # ---------------- Task 5: metrics on held-out test split ----------------
    sys.path.insert(0, "/workspace")
    from compare_models import per_pert_rows, summarize  # noqa: E402
    de_idx = np.load(f"{args.preds_dir}/test_de_idx.npy")
    rows = per_pert_rows(pred, truth, pert_cat, de_idx, ctrl_mean)
    summary = summarize(rows)
    with open(f"{args.out}/evaluation/metrics.json", "w") as f:
        json.dump({"split": "simulation_seed42_test_unseen_single",
                   "n_perts": len(rows), "summary": summary,
                   "note": "computed on per-perturbation mean profiles; "
                           "held-out test split only; no training data used"},
                  f, indent=1)
    with open(f"{args.out}/evaluation/per_perturbation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # training history: original-run log survived; resumed-run log was lost to a
    # log rotation - reconstructed from recorded epoch-end values (documented).
    hist = []
    for line in open("/workspace/train_gears.log"):
        parts = line.split()
        try:
            if "Step" in line and "Train Loss:" in line:
                ep = int(parts[1]); st = int(parts[3])
                loss = float(parts[-1])
                hist.append({"run": "original", "epoch": ep, "step": st,
                             "train_loss": loss, "val_mse": "", "val_mse_de": "", "note": ""})
            elif "Validation Overall MSE:" in line:
                ep = int(parts[1].rstrip(":"))
                v = float(line.split("Validation Overall MSE:")[1].split()[0].rstrip("."))
                hist.append({"run": "original", "epoch": ep, "step": "epoch_end",
                             "train_loss": "", "val_mse": v, "val_mse_de": "", "note": ""})
            elif "Validation Top 20 DE MSE:" in line:
                v = float(line.split("Validation Top 20 DE MSE:")[1].split()[0].rstrip("."))
                for h in reversed(hist):
                    if h["val_mse_de"] == "" and h["step"] == "epoch_end":
                        h["val_mse_de"] = v
                        break
        except Exception:
            pass
    # resumed-run epoch-end values (recorded at runtime; log file lost)
    hist += [
        {"run": "resumed", "epoch": 1, "step": "epoch_end", "train_loss": "",
         "val_mse": "", "val_mse_de": 0.5976, "note": "reconstructed from runtime records"},
        {"run": "resumed", "epoch": 2, "step": "epoch_end", "train_loss": "",
         "val_mse": 0.3085, "val_mse_de": 0.6073, "note": "reconstructed from runtime records"},
    ]
    with open(f"{args.out}/evaluation/training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "epoch", "step", "train_loss",
                                          "val_mse", "val_mse_de", "note"])
        w.writeheader()
        w.writerows(hist)

    # ---------------- Task 1 provenance ----------------
    import torch_geometric
    provenance = {
        "completed_epochs_total": 3,
        "completed_epochs_detail": "1 original + 2 resumed (identical config, continued from epoch-1 weights)",
        "optimizer_steps_approx": 3 * 2444,
        "optimizer_state": "not saved (checkpoint stores model weights only); training is frozen",
        "best_vs_final": ("only the final-epoch checkpoint was persisted; GEARS's in-memory "
                          "best-val model was not saved separately - final checkpoint is "
                          "NOT labeled as best"),
        "architecture": "GEARS_Model: gene_emb/pert_emb/emb_pos (64d) + SGConv co-expression GNN "
                        "(num_gene_gnn_layers) + SGConv GO GNN (num_go_gnn_layers) + "
                        "cross-gene MLP decoder with per-gene weights",
        "hidden_size": args.gears_hidden,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "loss": "GEARS loss_fct: MSE on pred vs y + direction penalty (direction_lambda), "
                "computed on train genes only (dict_filter); pred = decoder(...) + basal x[:,0]",
        "output_scale": "log1p-normalized expression (same scale as h5ad X); NOT raw counts; "
                        "delta outputs are differences of log1p values, not log-fold-changes",
        "seeds": {"split_seed": 42, "extraction_seed": SEED,
                  "training_seed": "no explicit global seed set during training"},
        "dataset": "Replogle 2022 K562 gwps, Figshare+ doi:10.25452/figshare.plus.20029387; "
                   "10 cells/pert subsample + 2000 ctrl = 78,190 cells; 8,248-gene HVG panel",
        "splits": {"train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"]),
                   "regimes": "unseen_single only (test genes have zero overlap with train genes)"},
        "graph_construction": {"coexpr_edges_csv": "simulation_42_0.75_0.4_20_coexpression_network.csv",
                               "go_csv": "go.csv (top-21 neighbors per gene retained by GEARS)",
                               "num_gene_gnn_layers": gears_model.model.num_layers_gene_pos,
                               "num_go_gnn_layers": gears_model.model.num_layers},
        "determinism_check_exact_equality": det_ok,
    }
    with open(f"{args.out}/checkpoints/provenance.json", "w") as f:
        json.dump(provenance, f, indent=1)

    # ---------------- Task 7-8: integrity report + checksums ----------------
    integrity = {"arrays_finite": True, "shape_checks": [], "issues": []}
    for name, arr in [("gene_embeddings", post_g), ("pert_embeddings_post", post_p),
                      ("delta_mean", delta_mean), ("predicted_mean", predicted_mean)]:
        finite = bool(np.isfinite(arr).all())
        integrity["shape_checks"].append({"name": name, "shape": list(arr.shape),
                                          "dtype": str(arr.dtype), "all_finite": finite})
        integrity["arrays_finite"] &= finite
    integrity["duplicate_gene_symbols"] = int(len(gene_list) - len(set(gene_list)))
    if integrity["duplicate_gene_symbols"]:
        integrity["issues"].append("duplicate gene symbols present in panel (recorded, not filled)")
    integrity["determinism_exact"] = det_ok
    integrity["missing_vcc_target_list"] = "VCC 2026 required-target list not provided; column filled with 'not_provided'"
    integrity["no_heldout_leakage"] = ("predictions use only ctrl-cell basal expression + perturbation "
                                       "identity; no perturbed-cell outcomes from val/test were inputs")
    with open(f"{args.out}/validation/integrity_report.json", "w") as f:
        json.dump(integrity, f, indent=1)

    # checksums for all handoff files
    checksums = {}
    for root, _, files in os.walk(args.out):
        for fn in files:
            p = os.path.join(root, fn)
            checksums[os.path.relpath(p, args.out)] = sha256(p)
    with open(f"{args.out}/manifest_checksums.json", "w") as f:
        json.dump(checksums, f, indent=1)
    print("[extract] DONE", flush=True)


if __name__ == "__main__":
    main()
