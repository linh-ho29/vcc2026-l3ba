import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

ROOT = Path.home() / 'vcc2026_cpu'
VAL_DIR = ROOT / 'vcc_2026_controls'
FINAL_DIR = ROOT / 'vcc_2026_controls_final'
ARTIFACTS = ROOT / 'artifacts'
PRED_DIR = ROOT / 'predictions'
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
SHRINKAGE = 0.65
CLIP_LOG_FOLD = 1.0
CELLS_PER_PERT = 400
MAX_CELL_COUNTS = 1_000_000

bank = np.load(ARTIFACTS / 'signature_bank_k562gwps.npz', allow_pickle=True)
source_genes = bank['genes'].astype(str)
source_perts = bank['perts'].astype(str)
source_effects = bank['effects']
source_gene_index = {x: i for i, x in enumerate(source_genes)}
source_pert_index = {x: i for i, x in enumerate(source_perts)}
print(f"Loaded bank: {len(source_perts)} trained perturbations, {len(source_genes)} genes", flush=True)


def bundle_metadata(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    gene_list = pd.read_csv(directory / 'gene_names.csv').iloc[:, 0].astype(str).tolist()
    pert_list = pd.read_csv(directory / 'pert_counts.csv').iloc[:, 0].astype(str).tolist()
    assert len(gene_list) == len(set(gene_list))
    assert len(pert_list) == len(set(pert_list))
    assert len(gene_list) == manifest['n_genes']
    assert len(pert_list) == manifest['n_constructs']
    return manifest, gene_list, pert_list


def report_coverage(directory):
    manifest, genes, perts = bundle_metadata(directory)
    print(directory, 'contexts:', manifest['contexts'],
          'gene intersection:', sum(x in source_gene_index for x in genes), '/', len(genes),
          'perturbations with trained signature:', sum(p in source_pert_index for p in perts), '/', len(perts),
          flush=True)
    return [x for x in perts if x not in source_pert_index]


def make_predicted_rows(X_ctrl, sample_idx, log_effect, strength, rng):
    subset = X_ctrl[sample_idx, :].tocsr(copy=True).astype(np.float32)
    if log_effect is not None:
        factor = np.exp(np.clip(strength * log_effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD))
        subset.data *= factor[subset.indices]
    lo = np.floor(subset.data)
    subset.data = (lo + (rng.random(len(lo)) < (subset.data - lo))).astype(np.float32)
    subset.eliminate_zeros()
    totals = np.asarray(subset.sum(axis=1)).ravel()
    if np.any(totals > MAX_CELL_COUNTS):
        raise ValueError('Cell count cap violated')
    assert np.isfinite(subset.data).all() and np.all(subset.data >= 0)
    return subset


def generate_context(directory, context, outpath, only_n_perts=None):
    manifest, genes, perts = bundle_metadata(directory)
    if only_n_perts is not None:
        perts = perts[:only_n_perts]
    a = ad.read_h5ad(directory / f'context_{context}.h5ad', backed='r')
    assert a.n_vars == len(genes)
    a_genes = a.var_names.astype(str).tolist()
    if a_genes != genes:
        raise ValueError('Context gene axis differs from gene_names.csv')
    assert a.obs['context'].astype(str).eq(context).all()
    assert a.obs['target_gene'].astype(str).eq(manifest['control_label']).all()
    src_idx = np.asarray([source_gene_index.get(g, -1) for g in genes], dtype=np.int64)
    rng = np.random.default_rng(SEED + sum(map(ord, context)))
    X = a.X[:, :]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    out_parts = []
    out_labels = []
    missing = []
    for j, p in enumerate(perts):
        sel = rng.choice(a.n_obs, size=CELLS_PER_PERT, replace=True)
        if p not in source_pert_index:
            delta = None
            missing.append(p)
        else:
            delta = np.zeros(len(genes), dtype=np.float32)
            match = src_idx >= 0
            delta[match] = source_effects[source_pert_index[p], src_idx[match]]
        part = make_predicted_rows(X, sel, delta, SHRINKAGE, rng)
        out_parts.append(part)
        out_labels.extend([p] * CELLS_PER_PERT)
        if (j + 1) % 50 == 0:
            print(context, 'generated', j + 1, '/', len(perts), 'targets', flush=True)
    X_out = sp.vstack(out_parts, format='csr')
    X_out.eliminate_zeros()
    obs = pd.DataFrame({
        'target_gene': pd.Categorical(out_labels),
        'context': pd.Categorical([context] * len(out_labels)),
    }, index=[f'{context}_{i:07d}' for i in range(len(out_labels))])
    result = ad.AnnData(X=X_out, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name='gene_name')))
    print(context, 'shape:', result.shape, 'nnz:', result.X.nnz, 'unseen perturbations:', len(missing), flush=True)
    if outpath is not None:
        result.write_h5ad(outpath, compression='gzip')
        print('Wrote', outpath, round(outpath.stat().st_size / 1e9, 2), 'GB', flush=True)
    a.file.close()
    del a, X, X_out, out_parts
    gc.collect()


def generate_panel(directory, outfile):
    manifest, genes, perts = bundle_metadata(directory)
    assert len(manifest['contexts']) == 3
    context_files = []
    for ctx in manifest['contexts']:
        p = PRED_DIR / f'{manifest["partition"]}_{ctx}_prediction.h5ad'
        generate_context(directory, ctx, outpath=p)
        context_files.append(p)
    ads = [ad.read_h5ad(p) for p in context_files]
    combo = ad.concat(ads, axis=0, join='exact', merge='same', index_unique=None)
    assert combo.n_obs == len(manifest['contexts']) * len(perts) * CELLS_PER_PERT
    combo.X = combo.X.tocsr()
    combo.X.eliminate_zeros()
    combo.write_h5ad(outfile, compression='gzip')
    print('Complete panel:', outfile, combo.shape, 'nnz:', combo.X.nnz, flush=True)
    del ads, combo
    gc.collect()


print("=== Coverage check ===", flush=True)
report_coverage(VAL_DIR)
report_coverage(FINAL_DIR)

print("=== Generating validation predictions (A/B/C) ===", flush=True)
generate_panel(VAL_DIR, PRED_DIR / 'validation_predictions_ABC_gwps.h5ad')

print("=== Generating final predictions (D/E/F) ===", flush=True)
generate_panel(FINAL_DIR, PRED_DIR / 'final_predictions_DEF_gwps.h5ad')

print("ALL DONE", flush=True)
