"""Thin wrapper around panhumanpy's AzimuthNN, shared by 01 and 03."""
import time

import numpy as np
import panhumanpy as ph

from config import MODEL_VERSION

CL_COLS = ["final_level_labels", "azimuth_broad", "azimuth_medium", "azimuth_fine"]


def run_pha(adata):
    """Run PHA on an AnnData with integer raw counts in .X and gene symbols as var_names.

    Returns (meta, embed, info): per-cell predictions (PHA columns + adata.obs), the 128-d
    embedding (rows aligned with meta), and a dict of run statistics.
    """
    t0 = time.time()
    az = ph.AzimuthNN(adata, model_version=MODEL_VERSION)
    for col in CL_COLS:
        # panhumanpy 1.0.0 bug: map_to_cell_ontology sorts unmapped labels and crashes on None
        # (lineages with no label at that resolution), so make them an explicit string first
        az.cells_meta[col] = az.cells_meta[col].fillna("None").astype(str)
        az.map_to_cell_ontology(col, include_cl_id=True)
    emb = az.azimuth_embed()
    elapsed = time.time() - t0

    meta = az.cells_meta.copy()
    meta.index.name = "cell_id"
    panel = set(az.inference_feature_panel)
    overlap = len(panel & set(adata.var_names))
    info = {
        "n_cells": adata.n_obs,
        "n_query_genes": adata.n_vars,
        "panel_genes": len(panel),
        "panel_overlap": overlap,
        "panel_overlap_pct": round(100 * overlap / len(panel), 1),
        "unassigned_pct": round(100 * (meta["final_level_labels"] == "Unassigned").mean(), 2),
        "inconsistent_hierarchy_pct": round(100 * (~meta["full_consistent_hierarchy"].astype(bool)).mean(), 2),
        "median_final_conf": round(meta["final_level_confidence"].median(), 3),
        "seconds": round(elapsed, 1),
    }
    return meta, np.asarray(emb), info
