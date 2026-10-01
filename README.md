![C3PRO — a mosaic of cells stained in H&E pink, seen through two lenses that recolour each cell by its morphological phenotype](.github/c3pro_banner.jpg)

# C3PRO

C3PRO is a cell-level pipeline for H&E whole-slide images of pancreatic adenocarcinoma. It detects every
nucleus, embeds each cell with a pathology foundation model and clusters the cells into a vocabulary of
phenotypes. Slide-level phenotype proportions and cell graphs then feed diagnosis, prognosis and
resectability models.

## Requirements

- Steps 01–02 need an NVIDIA GPU; steps 04 and 05 run k-means on the GPU (cupy / cuML).
- The weights of CellViT++ (step 01) and UNI2-h (step 02) are not redistributed here: obtain them from
  their authors, under their licences. UNI2-h is gated on Hugging Face; log in with `huggingface-cli login`
  or set `HF_TOKEN`.

## Installation (Python 3.11)

```bash
pip install -r requirements.txt
```

The GPU packages (cupy, cuML, RMM, faiss-gpu, CellViT++, PathoPatch) are listed at the end of
`requirements.txt`; install them following their own instructions for your CUDA version.

## Scripts

| Step | Script | What it does |
|---|---|---|
| 01 | `01_nuclei_detection_cellvit.py` | nucleus detection with CellViT++ |
| 02 | `02_cell_features_uni2.py` | one UNI2-h embedding per detected cell |
| 03 | `03_features_csv_to_parquet.py` | converts feature CSVs to parquet |
| 04 | `04_cell_clustering_streaming.py` | prognostic vocabulary: streaming spherical k-means |
| 05 | `05_cell_clustering_from_h5ad.py` | diagnostic vocabulary: spherical k-means on an .h5ad |
| 06 | `06_spatial_cluster_maps.py` | per-slide maps of cells coloured by cluster |
| 07 | `07_cluster_tile_sampling.py` | tiles of sampled cells for each cluster |
| 08 | `08_cellular_niches.py` | cellular neighbourhoods (niches) from the clusters |
| 09 | `09_niche_vignettes.py` | representative regions of each niche |
| 10 | `10_diagnosis_xgboost_shap.py` | diagnosis from cluster proportions, XGBoost and SHAP |
| 11 | `11_diagnosis_cell_shap_scores.py` | slide-level SHAP values projected onto cells |
| 12 | `12_diagnosis_shap_heatmap.py` | SHAP heatmaps over the slides |
| 13 | `13_gnn_graph_builder.py` | one cell graph per slide |
| 14 | `14_benchmark_fold_indices.py` | cross-validation folds shared by all models |
| 15 | `15_gnn_survival_nested_cv.py` | survival graph network, one per fold |
| 16 | `16_ai_score_cv.py` | prognosis and resectability models on shared folds |
| 17 | `17_resectability.py` | resectability analyses and classifiers |

Each script lists its options with `--help`. Steps 13 to 16 run in this order: `16 --dry-run` (cohort
table and folds), then 14, then 13 and 15, then 16.

## Inputs

A slide list, `slide_split.csv`:

```
slide_name,label,split
21AB000123-A01-1,Cancer,train
22AB000456-A01-1,NoCancer,test
```

Slide names start with a 10-character patient identifier (two-digit year, two letters, six digits); the
year separates the two scanner eras. A clinical table, one row per patient (`.xlsx`, columns below):

```
PATHID,DEATH,MONTHS_FU,AGE_AT_DIAGNOSIS,SEX,PATH_GRADE,SURGERY_SPECIMEN,LOCALISATION,PRETX_LARGE_VASCULAR,PRETX_SIZE,NEO_FOLFORINOX,NEO_GEMCITABINE,TX_GEMCITABINE,TX_FOLFORINOX,TX_NONE
21AB000123,1,14.2,67,1,2,0,1,0,32,0,0,1,0,0
```

The prognostic vocabulary (step 04) and the diagnostic vocabulary (step 05) are built independently:
their cluster numbers do not correspond.

## Tests

`pytest` runs the tests on CPU with synthetic data.

## Licence

MIT, see `LICENSE`.

Preprint: link to be added.
