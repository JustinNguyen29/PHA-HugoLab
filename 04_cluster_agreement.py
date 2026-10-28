# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: panhumanpy (py3.12)
#     language: python
#     name: panhumanpy
# ---

# %% [markdown]
# # 04 — How close is PHA to our manual clustering? (label-free)
#
# `02` scored PHA through a hand-built Cell Ontology crosswalk, which assumes both the crosswalk and the
# manual labels are ground truth. Here neither is assumed: we treat the manual annotation and PHA as
# two alternative partitions of the same cells and ask how they relate geometrically.
#
# Requires `03_qc_and_pha` (QC-passing cells, PHA predictions + 128-d embedding).
#
# **Spaces** (all distances are cosine):
# - `pha`: PHA's 128-d embedding. Biased towards PHA's own labels, since it was trained to separate them.
# - `pca`: 50 PCs of log-normalised expression (2,000 HVGs, batch-aware). Independent of both
#   labelings, so it is the neutral judge.
#
# **Analyses**
# 1. Cluster tightness: cell → own-centroid distance, silhouette, centroid–centroid distances
# 2. Label-free agreement: ARI / NMI / homogeneity / completeness, manual vs PHA at each level
# 3. Does PHA split manual clusters, and is the split real substructure? (permutation test)
# 4. Suspect cells: closer to another manual cluster, and does PHA agree they're mislabelled?
# 5. Paper summary table
#
# Glioma malignant states are included here (nothing depends on the ontology).
# Outputs → `results/cluster_agreement/`.

# %%
import os, sys
from pathlib import Path

# Run from the repo root (found via config.py) so relative paths are stable
_root = Path.cwd()
while not (_root / "config.py").exists():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root))

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
                             completeness_score, silhouette_samples, silhouette_score)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from config import DATASETS, RESULTS_DIR, load_counts_adata

QC_DIR = RESULTS_DIR / "qc_pha"
OUT_DIR = RESULTS_DIR / "cluster_agreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPACES = ["pca", "pha"]
PHA_LEVELS = ["azimuth_broad", "azimuth_medium", "azimuth_fine", "final_level_labels"]
N_PCS, N_HVG, KNN_K = 50, 2000, 15
SPLIT_MIN_FRAC, SPLIT_MIN_CELLS, N_PERM = 0.05, 10, 100
rng = np.random.default_rng(0)
pd.set_option("display.max_rows", 200, "display.max_colwidth", 80)


def l2norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def centroids(Xn, labels):
    """Unit-length centroid per label (mean of L2-normalised vectors, renormalised)."""
    df = pd.DataFrame(Xn).groupby(np.asarray(labels)).mean()
    return pd.DataFrame(l2norm(df.values), index=df.index)


# %% [markdown]
# ## 0. Load cells and build both spaces

# %%
def load(name):
    meta = pd.read_csv(QC_DIR / f"qc_pha_{name}_predictions.csv", index_col=0)
    meta["true_label"] = meta["true_label"].astype(str)
    emb = np.load(QC_DIR / f"qc_pha_{name}_embed.npy")

    a = load_counts_adata(name)[meta.index].copy()
    batch = "patient_id" if "patient_id" in a.obs else "patient"
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, batch_key=batch, subset=True)
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=N_PCS, random_state=0)
    return {"meta": meta, "pca": l2norm(a.obsm["X_pca"]), "pha": l2norm(emb)}


data = {name: load(name) for name in DATASETS}
{name: d["meta"].shape[0] for name, d in data.items()}

# %% [markdown]
# ## 1. Cluster tightness
#
# **Intra-cluster distance**: cosine distance of each cell to its own manual-cluster centroid.
# Tight clusters have small, narrow distributions; a long tail means heterogeneous or mixed clusters.

# %%
rows = []
for name, d in data.items():
    m = d["meta"]
    for space in SPACES:
        C = centroids(d[space], m["true_label"])
        own = C.loc[m["true_label"]].values
        dist = 1 - (d[space] * own).sum(axis=1)
        m[f"dist_own_{space}"] = dist
        rows.append(pd.DataFrame({"dataset": name, "space": space,
                                  "true_label": m["true_label"].values, "dist_to_centroid": dist}))
intra = pd.concat(rows, ignore_index=True)

intra_summary = (intra.groupby(["dataset", "true_label", "space"], sort=False)["dist_to_centroid"]
                 .describe(percentiles=[0.25, 0.5, 0.75])[["count", "mean", "25%", "50%", "75%"]]
                 .unstack("space").round(3))
intra_summary.to_csv(OUT_DIR / "intra_cluster_distance.csv")
intra_summary

# %%
for name in DATASETS:
    d = intra[intra["dataset"] == name]
    order = (d[d["space"] == "pca"].groupby("true_label")["dist_to_centroid"]
             .median().sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * len(order) + 1.5), sharey=True)
    for ax, space in zip(axes, SPACES):
        sns.violinplot(d[d["space"] == space], y="true_label", x="dist_to_centroid", order=order,
                       cut=0, inner="quartile", density_norm="width", color="#a5d6a7", ax=ax)
        ax.set_title(f"{space} space"); ax.set_xlabel("cosine distance to own centroid"); ax.set_ylabel("")
    fig.suptitle(f"{name}: intra-cluster distance (manual clusters)")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"intra_distance_{name}.png", dpi=150); plt.show()

# %% [markdown]
# **Silhouette** (cosine): how much closer each cell is to its own group than to the nearest other group,
# in [-1, 1]. Compared for the manual labels vs PHA's `final_level_labels`, in both spaces.
# The `pca` column is the fair comparison; in `pha` space PHA has a home advantage.
# Also shown: PHA `azimuth_medium` (closer in granularity to our labels). For a like-for-like comparison,
# every labeling is scored on the same cells: those whose PHA final label has ≥ 10 cells (PHA emits many
# tiny labels; singletons would score 0 by definition). Silhouette still depends on how many groups
# there are, so read the gaps alongside `n_groups`.

# %%
LABELINGS = [("manual", "true_label"), ("PHA_medium", "azimuth_medium"), ("PHA", "final_level_labels")]

sil_rows, sil_class = [], []
for name, d in data.items():
    m = d["meta"]
    vc = m["final_level_labels"].value_counts()
    mask = m["final_level_labels"].isin(vc[vc >= SPLIT_MIN_CELLS].index).values
    for space in SPACES:
        # per-cluster silhouette of the manual labels uses all cells
        m[f"sil_manual_{space}"] = silhouette_samples(d[space], m["true_label"].values, metric="cosine")
        sil_class.append(m.groupby("true_label")[f"sil_manual_{space}"].mean()
                         .rename(space).to_frame().assign(dataset=name))
        for labeling, col in LABELINGS:
            lbl = m[col].astype(str).values[mask]
            sil_rows.append({"dataset": name, "space": space, "labeling": labeling,
                             "n_groups": len(set(lbl)), "n_cells_scored": int(mask.sum()),
                             "silhouette": silhouette_score(d[space][mask], lbl, metric="cosine")})

sil = pd.DataFrame(sil_rows)
sil_wide = sil.pivot_table(index="dataset", columns=["space", "labeling"], values="silhouette",
                           sort=False).round(3)
for labeling, _ in LABELINGS:
    sil_wide[("n_groups", labeling)] = sil[sil["labeling"] == labeling].groupby("dataset", sort=False)["n_groups"].first()
sil_wide[("n_cells_scored", "")] = sil.groupby("dataset", sort=False)["n_cells_scored"].first()
sil_wide.to_csv(OUT_DIR / "silhouette_by_dataset.csv")
sil_wide

# %%
sil_per_class = (pd.concat(sil_class).reset_index().rename(columns={"index": "true_label"})
                 .groupby(["dataset", "true_label"], sort=False)[SPACES].first().round(3))
sil_per_class.to_csv(OUT_DIR / "silhouette_per_cluster.csv")
sil_per_class.sort_values("pca")

# %%
g = sns.catplot(sil, x="silhouette", y="dataset", hue="labeling", col="space", kind="bar",
                height=3.5, aspect=1.3, palette=["#5c6bc0", "#ffb74d", "#ef6c00"])
g.set_titles("{col_name} space"); g.set_axis_labels("mean cosine silhouette", "")
plt.savefig(OUT_DIR / "silhouette_manual_vs_pha.png", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# **Centroid–centroid distances** between manual clusters (PCA space). Pairs with small distance are
# the ones most likely to be confused / over-split in the manual annotation.

# %%
for name, d in data.items():
    C = centroids(d["pca"], d["meta"]["true_label"])
    D = pd.DataFrame(cosine_distances(C.values), index=C.index, columns=C.index)
    D.round(3).to_csv(OUT_DIR / f"centroid_distance_{name}.csv")
    Z = linkage(squareform(D.values, checks=False), method="average")
    cg = sns.clustermap(D, row_linkage=Z, col_linkage=Z, cmap="viridis_r", annot=D.round(2), fmt="g", annot_kws={"size": 6},
                        figsize=(0.5 * len(D) + 4, 0.5 * len(D) + 4),
                        cbar_kws={"label": "cosine distance"})
    cg.fig.suptitle(f"{name}: manual-cluster centroid distances (PCA)", y=1.02)
    cg.savefig(OUT_DIR / f"centroid_distance_{name}.png", dpi=150); plt.show()

# %% [markdown]
# ## 2. Label-free agreement
#
# Manual labels as `labels_true`, PHA as `labels_pred`:
# - **ARI / NMI**: overall agreement of the two partitions (chance-corrected / information-based)
# - **homogeneity**: each PHA label contains cells from only one manual cluster
# - **completeness**: each manual cluster falls into a single PHA label
#
# High homogeneity + low completeness ⇒ PHA *splits* our clusters (finer); the reverse ⇒ PHA *merges* them.
# The `shuffled` row is a null (manual labels permuted) and should be ≈ 0 for ARI.

# %%
agree_rows = []
for name, d in data.items():
    m = d["meta"]
    shuffled = rng.permutation(m["true_label"].values)
    for level in PHA_LEVELS + ["shuffled"]:
        pred = m["final_level_labels"].values if level == "shuffled" else m[level].astype(str).values
        true = shuffled if level == "shuffled" else m["true_label"].values
        agree_rows.append({"dataset": name, "pha_level": level, "n_pha_labels": len(set(pred)),
                           "ARI": adjusted_rand_score(true, pred),
                           "NMI": normalized_mutual_info_score(true, pred),
                           "homogeneity": homogeneity_score(true, pred),
                           "completeness": completeness_score(true, pred)})
agree = pd.DataFrame(agree_rows).set_index(["dataset", "pha_level"]).round(3)
agree.to_csv(OUT_DIR / "partition_agreement.csv")
agree

# %%
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(agree.drop("shuffled", level="pha_level")["ARI"].unstack("pha_level")[PHA_LEVELS]
            .reindex(list(DATASETS)), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("ARI: manual vs PHA, by PHA level"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(OUT_DIR / "ari_heatmap.png", dpi=150); plt.show()

# %% [markdown]
# ## 3. Does PHA split manual clusters — and is the split real?
#
# For each manual cluster: PHA labels covering ≥ 5% of its cells, and the normalised entropy of PHA labels
# (0 = one label, 1 = evenly spread). Then, within the cluster only and in **PCA space**, the silhouette
# of PHA's sub-groups (≥ 10 cells each) vs a null of 100 label permutations.
# A large z / small p ⇒ PHA's sub-labels pick out genuine expression substructure inside our cluster
# (candidate sub-clusters); z ≈ 0 ⇒ PHA's split is noise relative to expression.

# %%
def split_stats(Xn, pha):
    vc = pd.Series(pha).value_counts()
    p = (vc / vc.sum()).values
    out = {"n": len(pha), "n_pha_labels_5pct": int((p >= SPLIT_MIN_FRAC).sum()),
           "entropy_norm": float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0,
           "top_pha": ", ".join(f"{k} ({100 * v / vc.sum():.0f}%)" for k, v in vc.head(3).items())}
    keep_lbl = vc[vc >= SPLIT_MIN_CELLS].index
    mask = np.isin(pha, keep_lbl)
    if len(keep_lbl) < 2:
        return {**out, "sub_silhouette": np.nan, "null_mean": np.nan, "z": np.nan, "p_perm": np.nan}
    D = cosine_distances(Xn[mask])
    lbl = np.asarray(pha)[mask]
    obs = silhouette_score(D, lbl, metric="precomputed")
    null = np.array([silhouette_score(D, rng.permutation(lbl), metric="precomputed")
                     for _ in range(N_PERM)])
    return {**out, "sub_silhouette": obs, "null_mean": null.mean(),
            "z": (obs - null.mean()) / null.std(ddof=1),
            "p_perm": (1 + (null >= obs).sum()) / (N_PERM + 1)}


split_rows = []
for name, d in data.items():
    m = d["meta"]
    for lbl, idx in m.groupby("true_label").indices.items():
        split_rows.append({"dataset": name, "true_label": lbl,
                           **split_stats(d["pca"][idx], m["final_level_labels"].values[idx])})
splits = pd.DataFrame(split_rows).set_index(["dataset", "true_label"])
splits.round(3).to_csv(OUT_DIR / "pha_splits_within_clusters.csv")
splits.round(3).sort_values("z", ascending=False)

# %% [markdown]
# ## 4. Suspect cells
#
# In PCA space, `margin = d(own centroid) − min d(other centroid)`; **suspect** if margin > 0 (the cell is
# closer to another manual cluster than to its own). `knn_purity` = fraction of the 15 nearest
# neighbours (cosine) sharing the cell's manual label.
#
# PHA evidence, without the ontology: each manual cluster's *modal* PHA label is its PHA "signature".
# - `pha_disagrees`: the cell's PHA label ≠ its own cluster's modal PHA label
# - `pha_supports_switch`: the cell's PHA label = the modal PHA label of its **nearest other** cluster
#   (and that differs from its own cluster's). Suspect + supports_switch is the strongest mislabel signal.

# %%
susp_rows = []
for name, d in data.items():
    m = d["meta"]
    X, labels = d["pca"], m["true_label"].values
    C = centroids(X, labels)
    S = X @ C.values.T                                   # cosine similarity to each centroid
    own_i = C.index.get_indexer(labels)
    d_all = 1 - S
    d_own = d_all[np.arange(len(m)), own_i]
    d_other = d_all.copy(); d_other[np.arange(len(m)), own_i] = np.inf
    near_i = d_other.argmin(axis=1)

    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine").fit(X)
    nbr = nn.kneighbors(X, return_distance=False)[:, 1:]
    knn_purity = (labels[nbr] == labels[:, None]).mean(axis=1)

    mode_pha = m.groupby("true_label")["final_level_labels"].agg(lambda s: s.value_counts().index[0])
    own_mode = mode_pha.loc[labels].values
    near_label = C.index.values[near_i]
    near_mode = mode_pha.loc[near_label].values
    pha = m["final_level_labels"].values

    susp_rows.append(pd.DataFrame({
        "dataset": name, "true_label": labels, "nearest_other_cluster": near_label,
        "margin": d_own - d_other.min(axis=1), "knn_purity": knn_purity,
        "pha_label": pha, "pha_conf": m["final_level_confidence"].values,
        "own_cluster_modal_pha": own_mode, "nearest_cluster_modal_pha": near_mode,
        "pha_disagrees": pha != own_mode,
        "pha_supports_switch": (pha == near_mode) & (near_mode != own_mode),
    }, index=m.index))

susp = pd.concat(susp_rows)
susp["suspect"] = susp["margin"] > 0
susp["strong_suspect"] = susp["suspect"] & susp["pha_supports_switch"]
susp.to_csv(OUT_DIR / "per_cell_suspect.csv")
susp[susp["suspect"]].sort_values("margin", ascending=False).to_csv(OUT_DIR / "suspect_cells.csv")
print(f"{susp['suspect'].sum():,} suspect cells ({100 * susp['suspect'].mean():.1f}%), "
      f"{susp['strong_suspect'].sum():,} with PHA supporting the switch")

# %% [markdown]
# Is PHA disagreement concentrated on geometrically suspect cells? If PHA tracks real biology, cells that
# sit closer to another cluster should be much more likely to get that cluster's PHA signature.

# %%
(susp.groupby(["dataset", "suspect"])[["pha_disagrees", "pha_supports_switch", "pha_conf", "knn_purity"]]
 .mean().mul([100, 100, 1, 1]).round(2)
 .rename(columns={"pha_disagrees": "pha_disagrees_%", "pha_supports_switch": "pha_supports_switch_%",
                  "pha_conf": "mean_pha_conf", "knn_purity": "mean_knn_purity"}))

# %%
susp_by_cluster = (susp.groupby(["dataset", "true_label"], sort=False)
                   .agg(n=("suspect", "size"),
                        suspect_pct=("suspect", lambda s: 100 * s.mean()),
                        strong_suspect_pct=("strong_suspect", lambda s: 100 * s.mean()),
                        median_knn_purity=("knn_purity", "median"),
                        top_nearest_other=("nearest_other_cluster",
                                           lambda s: s.value_counts().index[0]))
                   .round(2))
susp_by_cluster.to_csv(OUT_DIR / "suspect_by_cluster.csv")
susp_by_cluster.sort_values("suspect_pct", ascending=False)

# %% [markdown]
# Where do strong suspects want to move? (manual label → nearest other cluster, counts)

# %%
moves = (susp[susp["strong_suspect"]].groupby(["dataset", "true_label", "nearest_other_cluster"])
         .size().rename("n_strong_suspects").sort_values(ascending=False))
moves.to_csv(OUT_DIR / "strong_suspect_moves.csv")
moves.head(30)

# %% [markdown]
# ## 5. Paper summary
# One row per dataset. `sil_*`: mean cosine silhouette; `ARI`/`NMI`: manual vs PHA `final_level_labels`.

# %%
fl = agree.xs("final_level_labels", level="pha_level")
paper = pd.DataFrame({
    "n_cells": susp.groupby("dataset", sort=False).size(),
    "n_manual_clusters": susp.groupby("dataset", sort=False)["true_label"].nunique(),
    "n_pha_labels": fl["n_pha_labels"],
    "median_intra_dist_pca": intra[intra["space"] == "pca"].groupby("dataset", sort=False)["dist_to_centroid"].median(),
    "median_intra_dist_pha": intra[intra["space"] == "pha"].groupby("dataset", sort=False)["dist_to_centroid"].median(),
    "sil_manual_pca": sil_wide[("pca", "manual")], "sil_PHA_pca": sil_wide[("pca", "PHA")],
    "sil_manual_pha": sil_wide[("pha", "manual")], "sil_PHA_pha": sil_wide[("pha", "PHA")],
    "ARI": fl["ARI"], "NMI": fl["NMI"],
    "homogeneity": fl["homogeneity"], "completeness": fl["completeness"],
    "suspect_%": susp.groupby("dataset", sort=False)["suspect"].mean() * 100,
    "strong_suspect_%": susp.groupby("dataset", sort=False)["strong_suspect"].mean() * 100,
    "clusters_with_real_PHA_split": splits.groupby(level="dataset", sort=False)
        .apply(lambda s: int(((s["p_perm"] < 0.05) & (s["z"] > 3)).sum())),
    "median_pha_conf": susp.groupby("dataset", sort=False)["pha_conf"].median(),
}).reindex(list(DATASETS)).round(3)
paper.to_csv(OUT_DIR / "paper_summary.csv")
paper.T
