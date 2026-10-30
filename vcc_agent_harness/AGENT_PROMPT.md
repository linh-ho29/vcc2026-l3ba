# VCC 2026 — Resumable, Context-Aware K562 Baseline Research Agent

## Mission
Extend the **existing, completed Replogle 2022 K562 statistical linear baseline** to predict perturbations in unseen VCC 2026 contexts D/E/F. Preserve the baseline and previous submissions. Do **not** introduce GEARS or Tahoe-100M in this development cycle.

Current Phase 0 observations (reported by existing coding agent): 2,162–2,975 of 18,533 output genes undetected in each target context; ungated effects currently transfer even to these genes; mean proposed gate weight ~0.58–0.59; all 275 covered perturbations change under gating; HBZ is a conspicuous K562 erythroid-specific example. Treat these as diagnostics, **not established accuracy improvements**.

## First read, then act
Read `reports/CURRENT_STATUS.md`, `harness/state.json`, `configs/experiments.json`, and code relevant to the first pending experiment. Inspect the repository to discover actual baseline locations and current exporter/validator. Do not invent paths, checkpoints, scores, accessions, or column names. Maintain a compact status report <=600 words, and persist all detailed data outside the LLM context.

## Non-negotiable rules
- Preserve the existing baseline and submitted artifacts; changes belong in new modules / versioned outputs.
- Experimental comparison must hold model coefficients, gene alignment, validation splits, count generation, and random seeds fixed unless deliberately testing one of them.
- Never use private D/E/F perturbation outcomes for training/tuning. Control cells are permitted for deriving context features.
- Use **labeled public perturbation responses** from an unseen line (prefer Replogle RPE1, where legally/technically available) for cross-context validation. Split tuning vs final assessment to prevent leakage.
- scBaseCount contributes *observational context knowledge*, not causal CRISPRi labels. Prove value through ablation, not assumption.
- CPU and sparse/chunked operations first. No paid GPU or cloud provisioning. Never download >5 GB or >12 source files without approval.
- Honor exact VCC schema as confirmed from current official documentation, rather than hardcoded assumptions.

## ScBaseCount selection plan (hypothetical quotas; identifiers MUST be verified)
Official source: https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/scBaseCount
Atlas: https://arcinstitute.org/tools/virtualcellatlas
Preferred reference release: 2026-01-12 **if present and verified**, otherwise record verified release and ask before substituting. Metadata-first selection in the currently available official bucket/Marketplace access. Restrict to human, nonperturbed, well annotated samples. Inspect actual count layer, technology, file sizes, gene identifiers, sample accessions, study provenance, and licensing before transfer. Target up to 30,000 cells across:

| Group | Metadata search terms | Cell target |
|---|---|---:|
| K562-like/hematopoietic | K562, hematopoietic | 5,000 |
| Erythroid | erythroblast, erythroid progenitor | 5,000 |
| Immune | T cell, B cell, monocyte | 6,000 |
| Epithelial | epithelial, cultured epithelial | 5,000 |
| Stromal | fibroblast, stromal, mesenchymal | 4,000 |
| Stem/progenitor | pluripotent, stem, progenitor | 5,000 |

Aim for <=2,000 cells per independent study, <=12 source files, <=5 GB download. These quotas are not verified availability. Never fabricate accession IDs. Build `data/manifests/scbasecount_selection.csv` with study/accession, cell population, annotation confidence, protocol, object path, file size, expected cells, reason chosen, exclusion/leakage checks. If no exact match, record a justified substitution and missing category. If access fails, mark BLOCKED and continue independent local experiments; do not retry indefinitely.

## Planned experiments
- E00: reproduce existing K562 baseline and current exported predictions.
- E01: constant global effect shrinkage (controls for overall attenuation).
- E02: binary gate for detected/undetected downstream genes.
- E03: smooth percentile gate + detection-fraction hybrid.
- E04: direction-aware gating (do not preclude possible induction from undetected genes).
- E05: scBaseCount metadata manifest, curated reference, reference-group pseudobulks, detection fractions, pathway features, and study-balanced PCA.
- E06: context-control-only representation vs context-control + scBaseCount representation; regularized adjustment with lambda=0 recovering E03, plus mismatched/shuffled reference negative control.

## Gating scientific safeguards
Let delta[g,j] be a K562 perturbation effect on **its documented scale**, and w[c,j] be a bounded context-specific gate. Do not exponentiate log1p-normalized differences as if they were log fold changes. Audit scale before mapping delta to multiplicative count-rate effects. Zero detected expression constrains negative/downregulation but does not prove biological absence; test positive-induction behavior separately. Report both mean gate and effect-weighted magnitude retained. An improvement in plausibility alone does not establish predictive accuracy.

## Integer count generation
Inspect and reuse the existing working count exporter if valid. If implementation needs improvement, use seeded control-cell sampling, documented scale conversion to positive expected rates, and fixed-depth multinomial draws to conserve per-cell observed UMI total as a first baseline; compare against calibrated Poisson/NB where appropriate. Preserve between-cell variation and produce sparse nonnegative integer matrices. Validate exact current CLI limits (previous brief used 400 cells/perturbation/context, 18,533 genes, per-cell total <=1,000,000, <=4.75e9 stored entries, no explicitly stored zeros); confirm rather than assume. Use CSR `.eliminate_zeros()` before writing, check gene order, labels, count totals, and run official dry-run validator when accessible.

## Evaluation protocol
Run controlled contrasts against the original baseline and global shrinkage. Metrics: mean-expression MAE/RMSE, DE direction and recovery, perturbation discrimination, sparsity, library-size/dispersion realism, and official metrics where available. Distinguish prediction-difference diagnostics on D/E/F (no ground truth) from accuracy metrics on genuinely held-out public perturbation responses. Compare paired random seeds. Reject or mark inconclusive if improvements are not robust, if a protected metric materially regresses, or if ablation suggests scBaseCount only acts as generic shrinkage.

## Agent harness execution requirements
Create/extend a deterministic local Python runner with state persisted atomically after every experiment. States: PENDING, RUNNING, COMPLETED, REJECTED, FAILED, BLOCKED. Each experiment has a declared command, dependencies, time/memory limits, output paths and required metric file. Capture stdout/stderr, exit code, run duration, code/config/data manifest hashes, seed and artifacts. Timeouts must kill the process; only retry recoverable failures twice. On restart, stale RUNNING experiments must be recovered safely and partial artifacts verified or discarded. Do not auto-promote models without a frozen evaluator and prespecified acceptance rules. Stop after 20 runs, three consecutive non-improvements, or resource/approval blockers. Make supervisor usage optional; do not promise the LLM session will remain alive indefinitely.

## Initial steps
1. Audit repository; map real baseline model, output format, existing gating, evaluator and dependencies.
2. Adapt the provided `harness/runner.py` scaffold (it runs **configured commands**, does not invent training logic) to actual scripts and write `configs/experiments.json`.
3. Run E00 end-to-end with deterministic results. Fix test failures before further work.
4. Perform E01–E04 with independent held-out public responses where available.
5. Audit scBaseCount metadata and produce verified accession manifest before downloading.
6. Implement E05 and E06 under download and resource caps.
7. Produce a final reproducible ablation table and recommendations.

## Reporting/continuation contract
After each run update `reports/CURRENT_STATUS.md` (<=600 words): champion, completed runs, verified metrics, blockers, exact next command. Record detailed results in `experiments/results.jsonl` and full rationale in `reports/decisions.md`. On context exhaustion, persist state and provide a small handoff. A fresh coding-agent session reads only this prompt, status, state, next experiment config and relevant code. Never claim autonomous infinite execution; durable local processes can continue only while host and job supervisor are available.
