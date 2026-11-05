# GEARS track - organized by cell type

Reorganized to match the Phase 1 ("offline precomputation") box of the
multi-context fusion architecture: per-cell-type pretrained GEARS models,
each producing extracted perturbation vectors `e_p^(k)` for cell type `k`.
Phase 1's other half (control-cell VAE/PCA -> NTC latents `z_0`) and the
rest of Phase 2 now live on the `gated-fusion-work` branch (the
`hybrid_adapter` package), not here - this track is purely the per-cell-type
GEARS models and the tooling to extract/merge their embeddings. RPE1
(Courtney's track) has no checkpoint or extracted embeddings here yet.

## Layout

```
gears_track/
├── per_celltype/
│   ├── k562/        # complete - see per_celltype/k562/README.md
│   ├── hepg2/        # partial - see per_celltype/hepg2/README.md and the gap noted below
│   └── jurkat/       # complete - see per_celltype/jurkat/README.md
├── merged/
│   └── gears_embeddings_merged.h5   # output of shared/scripts/extract_gears_embeddings.py
└── shared/
    ├── gears_patched_source/   # vendored, patched upstream `gears` library (v0.0.2) - used by any cell type
    └── scripts/                 # data-prep, graph-cache builders, training/inference, metrics/reporting
```

## k562 (`per_celltype/k562/`)

Complete: checkpoint (weights only, no optimizer state), embeddings
(pre-graph/post-graph/fused), evaluation metrics, integrity validation. 3
total epochs, paused for GPU infrastructure reasons. See its own README for
full detail (determinism caveats, honest metrics, resumption instructions).

## hepg2 (`per_celltype/hepg2/`)

From a separate handoff (Lawrence's `LMA/` folder, originally uploaded
directly to the `main` branch via GitHub's web UI, now on the `lawrence`
branch). **Three files did not survive that upload and are not included
here**, confirmed directly rather than assumed:

- `best_checkpoint.pt`, `epoch8_checkpoint.pt`, and the standalone
  `perturbation_embeddings.npz` were all exactly 2 bytes in the source
  branch - all three resolve to the identical git blob hash
  (`d3f5a12f...`), i.e. an empty placeholder, not real content. Per the
  folder's own `manifests/MANIFEST.json`, the real sizes should have been
  ~13.1 MB (each checkpoint) and ~3.1 MB (the embeddings file) - consistent
  with GitHub's web-upload size ceiling silently truncating a drag-and-drop
  upload rather than any error on this end.
- Everything else transferred correctly (verified against
  `MANIFEST.json`'s checksums where that manifest itself was current -
  see caveat below): `gene_embeddings.npz`, `heldout_effect_predictions.npz`
  (which does include a small `perturbation_embedding` array, but only for
  the 20 held-out test perturbations, not the full panel - not a substitute
  for the missing standalone embeddings file), all evaluation CSVs/JSONs,
  and the dataset/split manifests.
- **This means hepg2 cannot yet feed Phase 1's "Extracted Cell-Type
  Perturbation Vectors" step** (no full-panel embeddings, no usable
  checkpoint to resume or re-export from) until those two files are
  re-transferred - ideally via `git push` rather than the web UI, since the
  actual sizes exceed GitHub's browser-upload limit.
- Separately, `manifests/MANIFEST.json` appears to be from an earlier/
  different experimental configuration than what's actually here (it
  self-labels `"arm": "depth_100x150"`, a 100-target/150-cell design, while
  the folder's own README documents a 500-target/30-cell-cap design) - 4 of
  its 12 checksums don't match the files actually present. Not fixed here;
  just don't treat that manifest as a reliable reference for this folder's
  current contents.

See `per_celltype/hepg2/README.md` for the full original design
documentation (data source, split, training config, reproduction steps).

## jurkat (`per_celltype/jurkat/`)

Complete, no corruption (checked directly - every file's size and shape
matches its own README/MANIFEST). The most rigorously evaluated of the
three real cell-type models here - worth reading
`per_celltype/jurkat/README.md` in full before using it for anything beyond
embedding extraction: under response-blind, biology-diverse target
sampling, **GEARS underperforms a naive mean-shift baseline**
(pearson_delta 0.073 vs. 0.239) on the held-out split, and a dedicated
ablation (stretched LR schedule, 4x wider hidden layer, 2x more epochs)
shows this is **structural, not under-training** - every metric is
unchanged across arms. Consistent with similar weak-GEARS-vs-baseline
findings elsewhere in this project (see `multisource_experiments/` on the
`linhho` branch) - a third, independently-arrived-at data point for the
same pattern, not a one-off.

Not copied here: `heldout_effect_predictions.npz` (190 MB, exceeds
GitHub's limit) and the checkpoint-only `checkpoints/epoch{1..8}` sweep
(only `best_checkpoint.pt`, epoch 2, and `epoch8_checkpoint.pt` are kept,
matching the hepg2/k562 convention) - available locally in
`~/Downloads/jurkat_gears/` if needed.

## `merged/gears_embeddings_merged.h5`

Output of `shared/scripts/extract_gears_embeddings.py`: standardizes each
cell type's post-GNN perturbation embedding to a fixed `d_pert=64` (identity
for k562, already 64-d; a fixed/seeded - not learned - Gaussian random
projection for hepg2 and jurkat, both 32-d) and merges into one
`[n_genes, 4, 64]` tensor in the fixed order `[K562, HepG2, RPE1, Jurkat]`,
with a `[n_genes, 4]` binary mask (1 = real, 0 = imputed). Missing
cell-line/gene combinations are imputed with the cross-line mean of
whichever lines ARE available for that gene - never zero-filled, per this
project's established convention. RPE1 has mask=0 for every single gene
(no RPE1 model exists anywhere in this project yet); 11,632 genes have at
least one real source, mean coverage 0.496 across all (gene, cell-line)
pairs. "post-GNN" is defined precisely in the script's own docstring, not
assumed - it's the output of GEARS's `pert_fuse(sim_layers(pert_emb(...)))`,
matching what each handoff already saved as `*_fused`/`post_graph` arrays.
