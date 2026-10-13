# panhumanpy-HugoLab

Zero-shot benchmark of **Pan-human Azimuth** (PHA; Sarkar, Li et al., *Organism-scale annotation with
Pan-human Azimuth*, bioRxiv 2026, doi:10.64898/2026.07.16.738997) on the HugoLab lab datasets, compared
against our Cell2Sentence + HCE models (v11) and the C2S-embedding LR baseline from the sibling repo
[`Cell2Sentence-HugoLab`](../Cell2Sentence-HugoLab).

PHA is a supervised MLP (5,055 genes → 128-d embedding → 8 hierarchical heads, 13 → 382 classes)
trained on ~9.7M curated cells from 23 tissues, with calibrated per-level confidence, an "Unassigned"
QC class, and a Cell Ontology crosswalk. It lives in its own repo because it needs TensorFlow 2.17 /
Python 3.12, which conflicts with the C2S torch / Python 3.13 environment.

## Setup

```bash
~/miniconda3/bin/conda create -p ./.venv python=3.12
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name panhumanpy --display-name "panhumanpy (py3.12)"
```

Model weights (v1) download automatically to `~/.cache/panhumanpy/` on first run. CPU is fine
(~10–20 s per lab dataset); TF 2.17 does not see the RTX 5060 Ti.

Data is read in place from `../Cell2Sentence-HugoLab` (see `config.py`); nothing is copied.

## Notebooks

Notebooks are paired with `py:percent` scripts via jupytext — edit either, then `jupytext --sync`.

| Notebook | What it does |
|---|---|
| `01_run_azimuth` | Runs PHA on the 6 lab sets from raw counts, maps labels to CL, writes `results/pha_<dataset>_predictions.csv`, `pha_<dataset>_embed.npy`, `pha_run_summary.csv` |
| `02_evaluate_vs_ground_truth` | Scores PHA against the lab labels in Cell Ontology space (exact / finer / coarser / wrong / unassigned), with per-class tables, confusion heatmaps, and confidence analysis → `results/eval_vs_truth/` |

`label_crosswalk.csv` maps each lab label to the most specific CL term it justifies (hand-curated;
edit it to change what counts as correct). `cl_utils.py` holds the CL is_a hierarchy helpers.

## Input contract (verified in panhumanpy 1.0.0 source)

- Reads `adata.X`; if integer counts are detected it applies CP10k + log1p over **all query genes**,
  then subsets / zero-fills to the 5,055-gene panel. We therefore pass raw counts
  (`.raw.X` for lab-data files, `layers['counts']` for `All_cells.h5ad`).
- Gene symbols as `var_names`.
- `azimuth_broad/medium/fine == 'False'` = refinement found no match (mostly hierarchy-inconsistent
  cells); `'None'` = lineage has no label at that resolution. `final_level_labels` is always set.
- Workaround: `map_to_cell_ontology` crashes on `None` values (sorting str vs None), so those
  columns are cast to str first.
