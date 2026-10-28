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
# # 03 — scanpy QC → Pan-human Azimuth → confidence
#
# Paper pipeline for PHA as an alternative annotation: starting from raw counts, remove low-quality
# cells with scanpy, run PHA on the cells that pass, and report PHA's per-cell confidence.
#
# QC filters (each recorded as its own column so the funnel is reportable):
# - `qc_low_genes`: fewer than 200 detected genes
# - `qc_high_mt`: > 10% mitochondrial counts (the lab's own threshold)
# - `qc_mad_outlier`: > 5 MADs from the median on log total counts or log detected genes
# - `qc_doublet`: scrublet doublet call, run per patient
#
# **Input caveat:** the current lab files are already QC-filtered (and glioma is subsampled to 200
# cells/class), so few cells are expected to be removed. Pointing `config.DATASETS` at unfiltered
# matrices is all that's needed to run this on truly raw data.
#
# Outputs → `results/qc_pha/`: `qc_pha_<dataset>_predictions.csv` (PHA + QC columns),
# `qc_pha_<dataset>_embed.npy`, `qc_funnel.csv`, `confidence_by_cluster.csv`, figures.

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
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATASETS, RESULTS_DIR, load_counts_adata
from pha_utils import run_pha

QC_DIR = RESULTS_DIR / "qc_pha"
QC_DIR.mkdir(parents=True, exist_ok=True)

MIN_GENES, MAX_PCT_MT, N_MADS = 200, 10.0, 5
QC_FLAGS = ["qc_low_genes", "qc_high_mt", "qc_mad_outlier", "qc_doublet"]
CONF_THRESHOLDS = [0.5, 0.75, 0.9]


def patient_col(obs):
    return "patient_id" if "patient_id" in obs else "patient"


def mad_outlier(x, n_mads):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.abs(x - med) > n_mads * 1.4826 * mad


# %% [markdown]
# ## 1. QC

# %%
def run_qc(name):
    a = load_counts_adata(name)
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)

    a.obs["qc_low_genes"] = a.obs["n_genes_by_counts"] < MIN_GENES
    a.obs["qc_high_mt"] = a.obs["pct_counts_mt"] > MAX_PCT_MT
    a.obs["qc_mad_outlier"] = (mad_outlier(a.obs["log1p_total_counts"].values, N_MADS)
                               | mad_outlier(a.obs["log1p_n_genes_by_counts"].values, N_MADS))
    # scrublet normalises internally; run per patient so doublet simulation stays within a sample
    sc.pp.scrublet(a, batch_key=patient_col(a.obs), random_state=0)
    a.obs["qc_doublet"] = a.obs["predicted_doublet"].astype(bool)
    a.obs["qc_pass"] = ~a.obs[QC_FLAGS].any(axis=1)
    return a


qc = {name: run_qc(name) for name in DATASETS}

# %%
funnel = pd.DataFrame({
    name: {"cells_in": a.n_obs,
           **{f"flagged_{f.removeprefix('qc_')}": int(a.obs[f].sum()) for f in QC_FLAGS},
           "removed": int((~a.obs["qc_pass"]).sum()),
           "kept": int(a.obs["qc_pass"].sum()),
           "kept_%": round(100 * a.obs["qc_pass"].mean(), 1),
           "has_mt_genes": bool(a.var["mt"].any())}
    for name, a in qc.items()}).T
funnel.index.name = "dataset"
funnel.to_csv(QC_DIR / "qc_funnel.csv")
funnel

# %% [markdown]
# Flags overlap, so the `flagged_*` columns can sum to more than `removed`.
# Removed cells per manual cluster (does QC hit some labels disproportionately?):

# %%
removed_by_cluster = pd.concat([
    a.obs.groupby("true_label", observed=True)["qc_pass"]
     .agg(n="size", removed_pct=lambda s: round(100 * (~s).mean(), 1))
     .assign(dataset=name).reset_index()
    for name, a in qc.items()])
removed_by_cluster[removed_by_cluster["removed_pct"] > 0].sort_values("removed_pct", ascending=False)

# %% [markdown]
# ## 2. PHA on QC-passing cells

# %%
QC_OBS_COLS = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "doublet_score", *QC_FLAGS]

run_info, preds = [], {}
for name, a in qc.items():
    print(f"\n===== {name} =====")
    kept = a[a.obs["qc_pass"].values].copy()
    meta, emb, info = run_pha(kept)
    # cells_meta keeps obs columns, but make sure the QC columns travel with the predictions
    for c in QC_OBS_COLS:
        meta[c] = kept.obs[c].values
    meta.to_csv(QC_DIR / f"qc_pha_{name}_predictions.csv")
    np.save(QC_DIR / f"qc_pha_{name}_embed.npy", emb)
    preds[name] = meta
    run_info.append({"dataset": name, **info})

run_info = pd.DataFrame(run_info).set_index("dataset")
run_info.to_csv(QC_DIR / "qc_pha_run_summary.csv")
run_info

# %% [markdown]
# Sanity check: on pre-filtered input, kept cells should get the same labels as in `01` (PHA
# normalises each cell independently, so removing other cells shouldn't change a cell's call).

# %%
for name, meta in preds.items():
    old = pd.read_csv(RESULTS_DIR / f"pha_{name}_predictions.csv", index_col=0).loc[meta.index]
    same = (old["final_level_labels"].values == meta["final_level_labels"].values).mean()
    dconf = np.abs(old["final_level_confidence"].values - meta["final_level_confidence"].values).max()
    print(f"{name:18s} same final label: {100 * same:6.2f}% | max |Δconf|: {dconf:.2e}")

# %% [markdown]
# ## 3. Confidence

# %%
conf = pd.concat([m[["dataset", "true_label", "final_level_labels", "final_level_confidence"]]
                  for m in preds.values()])

fig, ax = plt.subplots(figsize=(8, 4))
for name in DATASETS:
    sns.kdeplot(conf.loc[conf["dataset"] == name, "final_level_confidence"], ax=ax, label=name,
                clip=(0, 1), cut=0)
ax.set_xlabel("PHA final-level confidence"); ax.set_xlim(0, 1)
ax.legend(); ax.set_title("PHA confidence on QC-passing cells")
plt.tight_layout(); plt.savefig(QC_DIR / "confidence_by_dataset.png", dpi=150); plt.show()

# %%
def conf_table(df):
    c = df["final_level_confidence"]
    return pd.Series({"n": len(c), "median_conf": c.median(),
                      **{f"conf>={t}_%": 100 * (c >= t).mean() for t in CONF_THRESHOLDS}})

conf_by_dataset = conf.groupby("dataset", sort=False).apply(conf_table, include_groups=False)
conf_by_dataset.loc["ALL"] = conf_table(conf)
conf_by_dataset.round(3).to_csv(QC_DIR / "confidence_by_dataset.csv")
conf_by_dataset.round(3)

# %%
conf_by_cluster = conf.groupby(["dataset", "true_label"], sort=False).apply(conf_table, include_groups=False).round(3)
conf_by_cluster.to_csv(QC_DIR / "confidence_by_cluster.csv")
pd.set_option("display.max_rows", 200)
conf_by_cluster

# %%
for name in DATASETS:
    d = conf[conf["dataset"] == name]
    order = d.groupby("true_label")["final_level_confidence"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(order) + 1.5))
    sns.violinplot(d, y="true_label", x="final_level_confidence", order=order, cut=0,
                   inner="quartile", density_norm="width", color="#90caf9", ax=ax)
    ax.set_xlim(0, 1); ax.set_xlabel("PHA final-level confidence"); ax.set_ylabel("manual label")
    ax.set_title(f"{name}: PHA confidence per manual cluster")
    plt.tight_layout(); plt.savefig(QC_DIR / f"confidence_violin_{name}.png", dpi=150); plt.show()
