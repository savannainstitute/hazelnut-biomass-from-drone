# hazelnut-biomass-from-drone

A modular Python pipeline for estimating per-bush above-ground biomass (AGB) and stored carbon in hazelnut orchards from aerial LiDAR or Structure-from-Motion (SfM) point clouds.

---

## Overview

The pipeline converts a raw LAS point cloud into bush-level biomass and carbon storage estimates in three stages:

1. **LiDAR preprocessing** — Classify ground returns, generate DTM/DSM rasters, and compute a Canopy Height Model (CHM).
2. **Canopy segmentation** — Segment individual hazelnut canopies from the CHM using a marker-controlled watershed algorithm, seeded by user-supplied tree-top points.
3. **Biomass estimation** — Compute per-bush canopy volume from the CHM, then apply a fitted allometric equation to estimate wet AGB and carbon.

Both LiDAR and SfM acquisition modes are supported with separate allometric models.

---

## Repository Structure

```
hazelnut-biomass-from-drone/
├── main.py                          # End-to-end pipeline entry point
├── hazelnut-biomass.yml             # Conda environment specification
├── lidar_preprocessing/
│   ├── preprocessing.py             # Ground classification, DTM/DSM/CHM generation
│   └── sample_data/
│       ├── inputs/sample_orchard.las
│       └── outputs/                 # Pre-computed sample outputs
├── canopy_segmentation/
│   ├── segmentation.py              # Tree top refinement + watershed segmentation
│   └── sample_data/
│       ├── inputs/                  # Sample marker shapefile (sample_orchard_trees.shp)
│       └── outputs/                 # Pre-computed sample segments and treetops
├── biomass_estimation/
│   ├── biomass.py                   # Volume calculation + allometric equations
│   └── sample_data/outputs/         # Pre-computed sample biomass results
├── notebooks/
│   └── hazelnut_biomass_from_drone.ipynb   # Full pipeline walkthrough notebook
└── supplemental/
    └── lr.py                        # Regression diagnostics for allometric model fitting
```

---

## Environment Setup

> **Note:** PDAL must be installed via conda-forge. It is not available through pip and is not compatible with a pip-only install.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Mambaforge.
2. Clone this repository:
   ```powershell
   git clone https://github.com/<your-org>/hazelnut-biomass-from-drone.git
   cd hazelnut-biomass-from-drone
   ```
3. Create and activate the conda environment:
   ```powershell
   conda env create -f hazelnut-biomass.yml
   conda activate hazelnut-biomass
   ```

The environment installs Python 3.10, PDAL 2.8.4, rasterio 1.4.3, geopandas 1.1.1, scikit-image 0.25.2, laspy 2.6.1, scipy 1.15.2, and related dependencies (see `hazelnut-biomass.yml` for pinned versions).

**Hardware:** All processing is CPU-based. No GPU is required. For full-orchard datasets, 16–64 GB RAM and a local SSD are recommended to handle large LAS files and raster operations.

---

## Inputs

| Input | Description |
|---|---|
| Raw LAS file | Aerial LiDAR or SfM point cloud in LAS/LAZ format |
| Tree-top marker shapefile | Point shapefile with one point per hazelnut bush (e.g., from RTK GPS survey or manual digitization over imagery) |
| Extent shapefile *(optional)* | Polygon shapefile used to crop/mask all raster and vector outputs to an orchard boundary |

Tree-top markers are required and must be supplied by the user. They are used as watershed seeds. The pipeline refines each marker to the local CHM maximum within a 1.75 m radius.

---

## Running the Full Pipeline

`main.py` chains all three stages and accepts a single LAS file and a marker shapefile:

```powershell
python main.py `
    --input-las "lidar_preprocessing/sample_data/inputs/sample_orchard.las" `
    --tree-tops-shp "canopy_segmentation/sample_data/inputs/sample_orchard_trees.shp" `
    --method lidar `
    --output-dir "outputs"
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--input-las` | Yes | Path to input LAS file |
| `--tree-tops-shp` | Yes | Path to tree-top marker shapefile |
| `--output-dir` | Yes | Directory for all outputs |
| `--method` | No | Allometric model: `lidar` (default) or `sfm` |
| `--extent-shapefile` | No | Polygon shapefile for spatial cropping/masking |
| `--res` | No | Raster resolution in meters (default: auto-estimated from point spacing) |

---

## Pipeline Details

### Stage 1 — LiDAR Preprocessing (`lidar_preprocessing/preprocessing.py`)

1. **Ground classification** — Runs PDAL's [SMRF filter](https://pdal.io/en/stable/stages/filters.smrf.html) on the raw LAS file. Default parameters: `scalar=1.2`, `slope=0.15`, `threshold=0.07`, `window=2.5`.
2. **DTM** — Rasterizes ground-classified points (LAS class 2) using inverse-distance weighting (IDW, power=2).
3. **DSM** — Rasterizes first returns (`ReturnNumber == 1`) using IDW.
4. **CHM** — Computed as `DSM − DTM`. Negative values are clamped to zero. If DSM and DTM extents differ, the DTM is bilinearly resampled to match the DSM grid before subtraction.
5. **Resolution** — If `--res` is not specified, it is estimated as `1 / sqrt(point_density)` from the LAS header.

**Outputs:**

| File | Description |
|---|---|
| `{prefix}_classified.las` | Ground-classified point cloud |
| `{prefix}_dtm.tif` | Digital Terrain Model (float32 GeoTIFF) |
| `{prefix}_dsm.tif` | Digital Surface Model (float32 GeoTIFF) |
| `{prefix}_chm.tif` | Canopy Height Model (float32 GeoTIFF, LZW compressed) |

---

### Stage 2 — Canopy Segmentation (`canopy_segmentation/segmentation.py`)

1. **Marker refinement** — Each input tree-top point is snapped to the CHM local maximum within a 1.75 m buffer window.
2. **Watershed segmentation** — Runs scikit-image's `watershed` on the inverted, Gaussian-smoothed CHM (sigma=0.5), using the refined markers as seeds. Only pixels with CHM > 0.1 m are included in the segmentation mask.
3. **Polygon extraction** — Each segment label is converted to a polygon. Small holes (< 8 px) are filled; small objects (< 8 px) are removed.

**Outputs:**

| File | Description |
|---|---|
| `{prefix}_treetops.shp` | Refined tree-top point locations with `tree_id` and `height` attributes |
| `{prefix}_segments.shp` | Canopy polygons with `tree_id`, `area_m2`, `max_h` (m), `mean_h` (m) |

---

### Stage 3 — Biomass Estimation (`biomass_estimation/biomass.py`)

**Volume calculation:** For each canopy polygon, the CHM is masked to that polygon and volume is computed as:

$$V = \sum_i h_i \cdot A_{px}$$

where $h_i$ is the CHM height of pixel $i$ and $A_{px}$ is the pixel area in m².

**Allometric equations:**

| Method | Equation | Relative RMSE |
|---|---|---|
| `lidar` | $\text{AGB} = 4.674 \times V$ | 19.3% |
| `sfm` | $\text{AGB} = 4.021 \times V^{0.841}$ | 22.6% |

AGB is wet above-ground biomass in kg; volume $V$ is in m³. Coefficients were derived by regression against destructive harvest measurements (ground-truth dataset not yet publicly released; see `supplemental/lr.py`).

**Carbon estimation:**

$$C = \text{AGB} \times 0.548 \times 0.5$$

where 0.548 is the dry-matter fraction and 0.5 is the carbon fraction, both determined from lab analysis of destructively harvested hazelnut bushes (publication pending).

**Error bounds** are added as AGB and carbon columns at ±1 relative RMSE.

**Output columns added to the segment shapefile/CSV:**

| Column | Description |
|---|---|
| `volume_m3` | Canopy volume (m³) |
| `agb_kg` | Estimated wet above-ground biomass (kg) |
| `agb_kg_lo` | Lower bound at −1 relative RMSE |
| `agb_kg_up` | Upper bound at +1 relative RMSE |
| `c_kg` | Estimated stored carbon (kg) |
| `c_kg_lo` | Carbon lower bound |
| `c_kg_up` | Carbon upper bound |

**Outputs:**

| File | Description |
|---|---|
| `{prefix}_biomass_carbon.shp` | Canopy polygons with all biomass/carbon attributes |
| `{prefix}_biomass_carbon.csv` | Same data in tabular form (no geometry) |

---

## Notebooks

- `notebooks/hazelnut_biomass_from_drone.ipynb` — Full pipeline walkthrough with step-by-step commentary.

Run with the `hazelnut-biomass` kernel (installed as `ipykernel` in the conda environment).

---

## Supplemental: Regression Analysis

`supplemental/lr.py` provides CLI-accessible regression diagnostics used to derive the allometric coefficients embedded in `biomass.py`. It fits raw linear, log-transformed linear, and pooled linear models to a `biomass_kg` vs. `volume_m3` CSV and reports R², Shapiro-Wilk normality, RMSE, and relative RMSE.

The ground-truth CSV (`ground_truth.csv`) will be released in `supplemental/` upon publication. Once available:

```powershell
python supplemental/lr.py --csv "supplemental/ground_truth.csv" --plot
```

The CSV must contain at minimum `biomass_kg` and `volume_m3` columns. Optional `site` and `treeID` columns are dropped before fitting.

---

## Limitations and Assumptions

- **Tree-top markers are required.** The segmentation is marker-controlled; fully automated tree detection is not implemented. Marker quality directly affects segmentation accuracy.
- **LAS coordinate units assumed to be meters.** The SMRF window, IDW radius, and buffer parameters are all specified in meters.
- **Single-return or classified inputs.** The DSM uses `ReturnNumber == 1` (first returns). Multi-return data where first returns are not classified will work; unclassified single-return data (e.g., some SfM outputs) should also work if all returns are flagged as return 1.
- **Allometric equations are hazelnut-specific.** Coefficients were fitted to hazelnut destructive harvest data and are not expected to generalize to other species or canopy architectures without re-fitting.
- **PDAL must be conda-installed.** The pipeline shells out to the PDAL CLI via `subprocess`. PDAL is not pip-installable.
- **No GPU usage.** All computation is CPU-based (numpy, rasterio, scikit-image, scipy).

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---
