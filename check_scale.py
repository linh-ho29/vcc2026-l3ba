import numpy as np

k562 = np.load("predictions/signature_bank_k562gwps.npz", allow_pickle=True)
rpe1 = np.load("predictions/signature_bank_rpe1.npz", allow_pickle=True)
k562_genes, k562_perts, k562_effects = k562["genes"].astype(str), k562["perts"].astype(str), k562["effects"]
rpe1_genes, rpe1_perts, rpe1_effects = rpe1["genes"].astype(str), rpe1["perts"].astype(str), rpe1["effects"]
k562_gene_idx = {g: i for i, g in enumerate(k562_genes)}
k562_pert_idx = {p: i for i, p in enumerate(k562_perts)}
rpe1_gene_idx = {g: i for i, g in enumerate(rpe1_genes)}
rpe1_pert_idx = {p: i for i, p in enumerate(rpe1_perts)}

overlap_perts = sorted(set(k562_pert_idx) & set(rpe1_pert_idx))
print("Perturbations with trained signatures in both banks:", len(overlap_perts))

k562_ontarget, rpe1_ontarget = [], []
for p in overlap_perts:
    if p in k562_gene_idx and p in rpe1_gene_idx:
        k562_ontarget.append(k562_effects[k562_pert_idx[p], k562_gene_idx[p]])
        rpe1_ontarget.append(rpe1_effects[rpe1_pert_idx[p], rpe1_gene_idx[p]])
k562_ontarget = np.array(k562_ontarget)
rpe1_ontarget = np.array(rpe1_ontarget)
valid = (~np.isnan(k562_ontarget)) & (~np.isnan(rpe1_ontarget))
k562_ontarget = k562_ontarget[valid]
rpe1_ontarget = rpe1_ontarget[valid]

print("n=", len(k562_ontarget))
print("K562 on-target: mean=", k562_ontarget.mean(), "median=", np.median(k562_ontarget), "std=", k562_ontarget.std())
print("RPE1 on-target: mean=", rpe1_ontarget.mean(), "median=", np.median(rpe1_ontarget), "std=", rpe1_ontarget.std())
print("Pearson:", np.corrcoef(k562_ontarget, rpe1_ontarget)[0, 1])
print("both negative frac:", np.mean((k562_ontarget < 0) & (rpe1_ontarget < 0)))
print("scale ratio std (RPE1/K562):", rpe1_ontarget.std() / k562_ontarget.std())
print("slope (RPE1 ~ K562, through origin):", np.sum(k562_ontarget * rpe1_ontarget) / np.sum(k562_ontarget ** 2))
