# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: panhumanpy (py3.12)
#     language: python
#     name: panhumanpy
# ---

# %% [markdown]
# # 01 — Run Pan-human Azimuth on the lab datasets (zero-shot)
#
# For each lab dataset: build a raw-counts AnnData (panhumanpy detects integer counts and applies
# CP10k + log1p itself, on all query genes, before subsetting to its 5,055-gene panel; missing panel
# genes are zero-filled), run `AzimuthNN` (model v1), map labels to Cell Ontology, and save per-cell
# predictions + 128-d embeddings to `results/`.
#
# Notes on outputs:
# - `azimuth_broad/medium/fine == 'False'` means panhumanpy's label refinement found no match
#   (mostly cells with `full_consistent_hierarchy == False`); `None` means the lineage has no label at
#   that resolution. `final_level_labels` is always populated.
# - `Unassigned` is PHA's built-in QC label (empty droplet / ambient / doublet-like profile).

# %%
import os, sys
from pathlib import Path

# Run from the repo root (found via config.py) so relative paths are stable
_root = Path.cwd()
while not (_root / "config.py").exists():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# %%
import time
import numpy as np
import pandas as pd
import panhumanpy as ph

from config import DATASETS, RESULTS_DIR, MODEL_VERSION, load_counts_adata

RESULTS_DIR.mkdir(exist_ok=True)
CL_COLS = ["final_level_labels", "azimuth_broad", "azimuth_medium", "azimuth_fine"]


# %%
def run_dataset(name):
    q = load_counts_adata(name)
    t0 = time.time()
    az = ph.AzimuthNN(q, model_version=MODEL_VERSION)
    for col in CL_COLS:
        # panhumanpy 1.0.0 bug: map_to_cell_ontology sorts unmapped labels and crashes on None
        # (lineages with no label at that resolution), so make them an explicit string first
        az.cells_meta[col] = az.cells_meta[col].fillna("None").astype(str)
        az.map_to_cell_ontology(col, include_cl_id=True)
    emb = az.azimuth_embed()
    elapsed = time.time() - t0

    meta = az.cells_meta.copy()
    meta.index.name = "cell_id"
    meta.to_csv(RESULTS_DIR / f"pha_{name}_predictions.csv")
    np.save(RESULTS_DIR / f"pha_{name}_embed.npy", emb)

    panel = set(az.inference_feature_panel)
    overlap = len(panel & set(q.var_names))
    return {
        "dataset": name,
        "n_cells": q.n_obs,
        "n_query_genes": q.n_vars,
        "panel_genes": len(panel),
        "panel_overlap": overlap,
        "panel_overlap_pct": round(100 * overlap / len(panel), 1),
        "unassigned_pct": round(100 * (meta["final_level_labels"] == "Unassigned").mean(), 2),
        "inconsistent_hierarchy_pct": round(100 * (~meta["full_consistent_hierarchy"].astype(bool)).mean(), 2),
        "median_final_conf": round(meta["final_level_confidence"].median(), 3),
        "seconds": round(elapsed, 1),
    }


# %%
summary = []
for name in DATASETS:
    print(f"\n===== {name} =====")
    summary.append(run_dataset(name))

summary = pd.DataFrame(summary)
summary.to_csv(RESULTS_DIR / "pha_run_summary.csv", index=False)
summary

# %% [markdown]
# ## Quick look: top PHA fine label per true label

# %%
for name in DATASETS:
    m = pd.read_csv(RESULTS_DIR / f"pha_{name}_predictions.csv", index_col=0)
    top = (m.groupby("true_label")["final_level_labels"]
             .agg(lambda s: ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(3).items())))
    print(f"\n### {name}")
    print(top.to_string())
