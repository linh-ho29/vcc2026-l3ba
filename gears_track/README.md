# GEARS track - organized by cell type

Reorganized to match the Phase 1 ("offline precomputation") box of the
multi-context fusion architecture: per-cell-type pretrained GEARS models,
each producing extracted perturbation vectors `e_p^(k)` for cell type `k`.
Phase 1's other half (control-cell VAE/PCA -> NTC latents `z_0`) and all of
Phase 2 (gated fusion, input conditioning, latent flow block, decoder,
losses) are **not yet present** - awaiting the adapter code and additional
trained cell types the user will point to separately.

## Layout

```
gears_track/
├── per_celltype/
│   ├── k562/        # complete - see per_celltype/k562/README.md
│   └── hepg2/        # partial - see per_celltype/hepg2/README.md and the gap noted below
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
