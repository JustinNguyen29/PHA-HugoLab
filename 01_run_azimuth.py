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
import numpy as np
import pandas as pd

from config import DATASETS, RESULTS_DIR, load_counts_adata
from pha_utils import run_pha

RESULTS_DIR.mkdir(exist_ok=True)


# %%
def run_dataset(name):
    meta, emb, info = run_pha(load_counts_adata(name))
    meta.to_csv(RESULTS_DIR / f"pha_{name}_predictions.csv")
    np.save(RESULTS_DIR / f"pha_{name}_embed.npy", emb)
    return {"dataset": name, **info}


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
