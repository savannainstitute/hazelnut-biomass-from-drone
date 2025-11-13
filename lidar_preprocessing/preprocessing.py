"""
Preprocess LiDAR point cloud data for hazelnut biomass estimation using PDAL and rasterio.

Steps:
1. Ground classification (PDAL)
2. Digital Terrain Model (DTM) generation (PDAL)
3. Digital Surface Model (DSM) generation (PDAL)
4. Canopy Height Model (CHM) calculation (DSM - DTM)
5. Save outputs as GeoTIFFs (rasterio)
"""

import os
import logging
import rasterio
import laspy
import numpy as np
import subprocess

def run_pdal_pipeline(pipeline_json):
    """
    Run a PDAL pipeline from a JSON object.

    Args:
        pipeline_json (dict or list): PDAL pipeline definition as a Python object.

    Raises:
        RuntimeError: If the PDAL pipeline fails.

    Returns:
        None
    """
    import json
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        f.write(json.dumps(pipeline_json))
        pipeline_path = f.name
    try:
        result = subprocess.run(['pdal', 'pipeline', pipeline_path], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"PDAL pipeline failed: {result.stderr}")
            raise RuntimeError(f"PDAL pipeline failed: {result.stderr}")
        else:
            logging.info(f"PDAL pipeline succeeded: {result.stdout}")
    finally:
        os.remove(pipeline_path)

def classify_ground(input_las, output_las, scalar=1.2, slope=0.15, threshold=0.07, window=2.5):
    """
    Classify ground points using PDAL SMRF filter.

    Args:
        input_las (str): Path to input LAS file.
        output_las (str): Path to output classified LAS file.
        scalar (float): Multiplier for the mean absolute deviation (MAD) for ground threshold.
        slope (float): Maximum allowed slope between neighboring points.
        threshold (float): Maximum allowed height difference for ground classification.
        window (float): Neighborhood window size in meters (suggested: max canopy diameter).

    Returns:
        None
    """
    ground_pipeline = [
        {
            "type": "readers.las",
            "filename": input_las
        },
        {
            "type": "filters.smrf",
            "scalar": scalar,
            "slope": slope,
            "threshold": threshold,
            "window": window
        },
        {
            "type": "writers.las",
            "filename": output_las
        }
    ]
    run_pdal_pipeline(ground_pipeline)
    logging.info(f"Classified LAS saved to {output_las}")

def estimate_point_spacing(las_path):
    """
    Estimate average point spacing from a LAS file.

    Args:
        las_path (str): Path to LAS file.

    Returns:
        float: Estimated average point spacing (meters).
    """
    las = laspy.read(las_path)
    x, y = las.x, las.y
    area = (x.max() - x.min()) * (y.max() - y.min())
    if area == 0 or len(x) < 2:
        return 0.25  # fallback
    density = len(x) / area  # points per m^2
    spacing = 1 / np.sqrt(density)
    return spacing

def create_dtm(classified_las, dtm_tif, res=None):
    """
    Create Digital Terrain Model (DTM) from ground-classified LAS.

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dtm_tif (str): Output path for DTM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None, estimated from point spacing.

    Returns:
        None
    """
    if res is None:
        res = estimate_point_spacing(classified_las)
        logging.info(f"Inferred DTM resolution: {res:.3f} m")
    dtm_pipeline = [
        {
            "type": "readers.las",
            "filename": classified_las
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.gdal",
            "filename": dtm_tif,
            "resolution": res,
            "output_type": "idw",
            "power": 2,
            "radius": res * 10,
            "window_size": 100,
            "data_type": "float32"
        }
    ]
    run_pdal_pipeline(dtm_pipeline)
    logging.info(f"DTM saved to {dtm_tif}")

def create_dsm(classified_las, dsm_tif, res=None):
    """
    Create Digital Surface Model (DSM) from ground-classified LAS.

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dsm_tif (str): Output path for DSM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None, estimated from point spacing.

    Returns:
        None
    """
    if res is None:
        res = estimate_point_spacing(classified_las)
        logging.info(f"Inferred DSM resolution: {res:.3f} m")
    dsm_pipeline = [
        {
            "type": "readers.las",
            "filename": classified_las
        },
        {
            "type": "filters.range",
            "limits": "ReturnNumber[1:1]"
        },
        {
            "type": "writers.gdal",
            "filename": dsm_tif,
            "resolution": res,
            "output_type": "idw",      # Use IDW for interpolation
            "power": 2,
            "radius": res * 10,        # Match DTM settings
            "window_size": 100,
            "data_type": "float32"
        }
    ]
    run_pdal_pipeline(dsm_pipeline)
    logging.info(f"DSM saved to {dsm_tif}")

def create_chm(dsm_tif, dtm_tif, chm_tif):
    """
    Create Canopy Height Model (CHM) by subtracting DTM from DSM.

    Args:
        dsm_tif (str): Path to DSM GeoTIFF.
        dtm_tif (str): Path to DTM GeoTIFF.
        chm_tif (str): Output path for CHM GeoTIFF.

    Returns:
        np.ndarray: CHM array (DSM - DTM).
    """
    with rasterio.open(dsm_tif) as dsm_src, rasterio.open(dtm_tif) as dtm_src:
        dsm = dsm_src.read(1)
        dtm = dtm_src.read(1)
        chm = dsm - dtm
        chm[chm < 0] = 0  # Remove negative values
        meta = dsm_src.meta.copy()
        meta.update(dtype='float32', compress='lzw')
        with rasterio.open(chm_tif, 'w', **meta) as dst:
            dst.write(chm.astype('float32'), 1)
    logging.info(f"CHM saved to {chm_tif}")
    return chm

def preprocess_lidar(input_las, output_dir, res=None):
    """
    Run all preprocessing steps and return file paths.

    Args:
        input_las (str): Path to input LAS file.
        output_dir (str): Output directory for all results.
        res (float, optional): Raster resolution in meters (default 0.25).

    Returns:
        dict: {
            "classified_las": path to ground-classified LAS,
            "dtm": path to DTM GeoTIFF,
            "dsm": path to DSM GeoTIFF,
            "chm": path to CHM GeoTIFF,
            "chm_array": CHM array (numpy)
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    prefix = os.path.splitext(os.path.basename(input_las))[0]
    ground_las = os.path.join(output_dir, f"{prefix}_classified.las")
    dtm_tif = os.path.join(output_dir, f"{prefix}_dtm.tif")
    dsm_tif = os.path.join(output_dir, f"{prefix}_dsm.tif")
    chm_tif = os.path.join(output_dir, f"{prefix}_chm.tif")

    classify_ground(input_las, ground_las)

    if res is None:
        res = estimate_point_spacing(ground_las)
        
    create_dtm(ground_las, dtm_tif, res)
    create_dsm(ground_las, dsm_tif, res)
    chm = create_chm(dsm_tif, dtm_tif, chm_tif)

    return {
        "classified_las": ground_las,
        "dtm": dtm_tif,
        "dsm": dsm_tif,
        "chm": chm_tif,
        "chm_array": chm
    }