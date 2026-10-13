"""Shared paths and dataset table for the Pan-human Azimuth benchmark.

Data stays in the sibling Cell2Sentence-HugoLab repo; nothing is copied here.
"""
from pathlib import Path

import anndata as ad

REPO_ROOT = Path(__file__).resolve().parent
C2S_ROOT = REPO_ROOT.parent / "Cell2Sentence-HugoLab"
LAB_DATA_DIR = C2S_ROOT / "lab-data"
RESULTS_DIR = REPO_ROOT / "results"

# Existing C2S results we compare against (per-cell CSVs, see 02_compare)
V11_RESULTS_DIR = C2S_ROOT / "multi_tissue_v11_results"
LR_RESULTS_DIR = C2S_ROOT / "multi_tissue_lr_baseline_results"

MODEL_VERSION = "v1"

# name -> (path, where raw counts live, label column)
#   counts_source: "raw"            -> adata.raw.X, gene symbols in raw.var['_index']
#                  "layers:counts"  -> adata.layers['counts'], gene symbols in var_names
DATASETS = {
    "glioma":            (C2S_ROOT / "All_cells.h5ad",                "layers:counts", "predicted.high_hierarchy"),
    "Brain_normal":      (LAB_DATA_DIR / "Brain_normal.h5ad",         "raw",           "cell_type"),
    "Liver_normal":      (LAB_DATA_DIR / "Liver_normal.h5ad",         "raw",           "cell_type"),
    "Lymph_node_normal": (LAB_DATA_DIR / "Lymph_node_normal.h5ad",    "raw",           "cell_type"),
    "lymphoid":          (LAB_DATA_DIR / "lymphoid.h5ad",             "raw",           "cell_type"),
    "myeloid":           (LAB_DATA_DIR / "myeloid.h5ad",              "raw",           "cell_type"),
}


def load_counts_adata(name):
    """Return an AnnData with integer raw counts in .X, gene symbols as var_names,
    the original obs, and the ground-truth label copied to obs['true_label']."""
    path, counts_source, label_col = DATASETS[name]
    adata = ad.read_h5ad(path)

    if counts_source == "raw":
        raw = adata.raw
        var = raw.var.copy()
        if "_index" in var.columns:
            var.index = var["_index"].astype(str).values
            var = var.drop(columns="_index")
        counts = ad.AnnData(X=raw.X.copy(), obs=adata.obs.copy(), var=var)
    elif counts_source == "layers:counts":
        counts = ad.AnnData(X=adata.layers["counts"].copy(), obs=adata.obs.copy(), var=adata.var.copy())
    else:
        raise ValueError(f"unknown counts_source {counts_source!r}")

    counts.var_names_make_unique()
    counts.obs["true_label"] = adata.obs[label_col].astype(str).values
    counts.obs["dataset"] = name
    return counts
