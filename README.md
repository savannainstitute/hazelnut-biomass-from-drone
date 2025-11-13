# hazelnut-biomass-from-drone

Pipeline for estimating hazelnut above-ground biomass and stored carbon using aerial LiDAR.

---

## Overview

This repository provides a modular pipeline to convert LiDAR point clouds into bush-level biomass and carbon storage estimates. The main steps are:

1. **LiDAR preprocessing**: Classify ground, build DTM/DSM, and produce a CHM.
2. **Canopy segmentation**: Segment individual hazelnut canopies from the CHM using a proximity-aware watershed algorithm with tree-top markers.
3. **Biomass estimation**: Compute per-bush volume metrics and apply a volume-based allometric equation to estimate biomass and carbon.

---

## Hardware Requirements

- Windows OS
- Recommended: 8+ GB GPU VRAM
- RAM: 64+ GB for large datasets
- CPU: multi-core (8+ cores recommended)
- Disk: ~100 GB free for full orchard datasets

---

## Sample Data

Sample data for testing is provided in the `*/sample_data/` folders within each subproject (e.g., `lidar_preprocessing/sample_data`, `canopy_segmentation/sample_data`).

---

## Environment Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Clone this repository.
3. Create and activate the conda environment:

    ```powershell
    conda env create -f hazelnut-biomass.yml
    conda activate hazelnut-biomass
    ```

- All required dependencies are installed via the provided YAML file.

---

## Pipeline Steps: Inputs & Outputs

### 1. LiDAR Preprocessing

**Inputs:**
- `sample_orchard.las` (Raw LAS file)

**Outputs:**
- `sample_orchard_classified.las` (ground-classified LAS)
- `sample_orchard_dtm.tif` (digital terrain model)
- `sample_orchard_dsm.tif` (digital surface model)
- `sample_orchard_chm.tif` (canopy height model)

### 2. Canopy Segmentation

**Inputs:**
- `sample_orchard_chm.tif` (from preprocessing)
- Tree points shapefile (RTK or digitized points)

**Outputs:**
- `sample_orchard_segments.shp` (canopy polygons)

### 3. Biomass Estimation

**Inputs:**
- `sample_orchard_segments.shp` (from segmentation)
- `sample_orchard_chm.tif` (from preprocessing)

**Outputs:**
- `sample_orchard_biomass_results.shp` (polygons with biomass/carbon attributes)
- `sample_orchard_biomass_results.csv` (tabular results)

---

## Running the Full Pipeline

You can run the entire pipeline (preprocessing, segmentation, biomass estimation) with a single command using `main.py`.

### Example

```powershell
python main.py --input-las "sample_data/inputs/hazelnuts_valleyFarm_091625_clip.las" `
               --tree-tops-shp "canopy_segmentation/sample_data/inputs/hazelnuts_valleyFarm_markers.shp" `
               --output-dir "outputs"
```

**Optional arguments:**
- `--extent-shapefile`: Path to extent shapefile for cropping/masking
- `--res`: Raster resolution in meters (default: inferred from input LAS point spacing)

**Outputs:**  
All outputs will be saved in the specified `--output-dir` and named based on the input LAS file.  
For example, if your input LAS is `sample_orchard.las`, the outputs will be:

- `sample_orchard_classified.las`  
- `sample_orchard_dtm.tif`  
- `sample_orchard_dsm.tif`  
- `sample_orchard_chm.tif`  
- `sample_orchard_segments.shp`  
- `sample_orchard_biomass_results.shp`  
- `sample_orchard_biomass_results.csv`  

---

## Advanced: Running Individual Steps

You can also run each step separately using the scripts in their respective subfolders. See the script docstrings or use `--help` for details.

---

## Supplemental Regression Analysis

The regression code for deriving the volume→biomass coefficient is in `supplemental/lr.py`.  
The ground-truth CSV (`ground_truth.csv`) will be placed in the `supplemental/` folder after this work is published. Once available, you can run:

```powershell
python supplemental/lr.py --csv "supplemental/ground_truth.csv" --plot
```

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---
