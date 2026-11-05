## System architecture: gated latent multi-context fusion

Two phases. Phase 1 runs once, offline, CPU-only (or a free T4) and produces
two kinds of precomputed inputs; Phase 2 is the actual trainable model,
lightweight enough to train online in ~1-2 hours of GPU time.

### Phase 1: offline precomputation

```
[Control Single Cells]                         [Target Gene Perturbation (p)]
        |                                                   |
Frozen Lightweight VAE / PCA                      Pretrained GEARS Models
  (Trained on all NTCs)                           (Trained per cell type k)
        |                                                   |
        v                                                   v
   Saved NTC Latents                      Extracted Cell-Type Perturbation Vectors
   z_0 ~ P_ctrl  in R^128                   { e_p^(1), e_p^(2), ..., e_p^(K) }
```

- **Control-cell side**: a frozen VAE/PCA trained once on all non-targeting
  control cells, producing a saved latent `z_0` per control cell (R^128).
- **Perturbation side**: one pretrained GEARS model per cell type `k`
  (currently k562 and hepg2 - see `GEARS` branch's `gears_track/per_celltype/`;
  rpe1 and others to follow per the contributor list above), each
  contributing an extracted perturbation vector `e_p^(k)` for target gene `p`.

### Phase 2: online lightweight GPU training (~1-2 hours)

**Gated latent multi-context fusion** - combines the per-cell-type
perturbation vectors into one fused vector, weighted by how relevant each
cell type's signal is to the current context:

```
Inputs: context c (cell line/type embedding in R^32)
        perturbation vectors { e_p^(k) } for k in {1..K}

Gating weights:
    alpha_k = Softmax( W_gate . [ e_p^(k) || c ] )

Fused perturbation vector:
    e_p^(fused) = sum_k alpha_k . Linear(e_p^(k))     in R^64
```

**Input conditioning layer**: concatenate `[ z_0 (128) || e_p^(fused) (64)
|| c (32) ]`, linear projection (224 -> 256) + LayerNorm.

**Discrete latent flow block** (2-3 steps), an explicit Euler step - no ODE
solver overhead, O(1) memory, no trajectory-stiffness bugs:

```
z(t+1) = z(t) + MLP_flow( z(t), e_p^(fused), c, t )
  - Pre-LayerNorm -> Linear(256->256) -> GELU -> Dropout(0.1)
```

Produces predicted latent `z_pert` (R^128), which a light linear decoder
head projects to `R^G` (G genes) as predicted expression `y_hat_pert`.

**Losses:**
- Reconstruction: `MSE(y_hat_pert, y_true)` - mean transcriptomic fit.
- Sliced Wasserstein: `SWD({z_pert}, {z_true})` - preserves cell-to-cell
  heterogeneity that a pure MSE loss on mean profiles would wash out.

### Status

Phase 1's per-cell-type GEARS models (k562, hepg2) exist on the `GEARS`
branch; rpe1 (Courtney) is tracked on its own branch and not yet merged in
here.

Everything else in this diagram - the gated multi-context fusion, the
control-cell branch (frozen scGPT pooling + PCA, since no VAE was needed:
sampling real encoded control cells empirically makes a generative prior
unnecessary), the discrete latent flow block, the decoder, and the Sliced
Wasserstein loss - is implemented on top of Aretha's adapter package
(`src/hybrid_adapter`), as a second model (`GatedFusionFlowAdapter`)
alongside the original `HybridAdapter` (`corrected = stat + residual`,
untouched). Verified end to end against real data - 2,000 real HepG2
control cells, the real fine-tuned scGPT embeddings (frozen, used as an
offline feature extractor per direction - though this project's own
evaluation found the fine-tune didn't measurably beat frozen embeddings
under its probe, worth knowing before expecting much from that choice
alone), and real k562+hepg2 GEARS embeddings for 2,122 overlapping genes -
but only as a forward-pass compatibility check (no training, no gradients
against real data, per the package's synthetic-phase scope). This work is
on the `gated-fusion-work` branch (forked from `Aretha_phylo`, now pushed -
not yet merged into `Aretha_phylo` or here) - the new modules live together
under `src/hybrid_adapter/gated_fusion_flow/`, separate from the original
Module A-D files, which that branch leaves untouched.



Contributions: 
1. Linh: project manager, data context and engineering as well as advanced inference engineering and contribute to form a scientifically sound hypothesis
2. Aretha: train GEARS with K562 data, form scientifically sound hypothesis and build a fusion adapter to fit with the hypothesis. 
3. Courtney: train GEARS with RPE1 data
4. Lawrence: train GEARS with HepG2 with broad pertubation sampling
5. Buu: form scientifically sound hypothesis regarding gene ontology graph and Markov chains
6. Lerie: pinpoint LLM platforms that can be helpful for this project
7. Sang: for training team members on basic concepts of bioinformatics and strategizing data engineering
