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
from rasterio.warp import reproject, Resampling
import geopandas as gpd
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

def get_bounds_from_shapefile(shapefile_path):
    """
    Returns PDAL bounds string from a shapefile.
    """
    gdf = gpd.read_file(shapefile_path)
    minx, miny, maxx, maxy = gdf.total_bounds
    return f"([{minx},{maxx}],[{miny},{maxy}])"

def classify_ground(input_las, output_las, scalar=1.2, slope=0.15, threshold=0.07, window=2.5, bounds=None):
    """
    Classify ground points using PDAL SMRF filter. https://pdal.io/en/stable/stages/filters.smrf.html

    Args:
        input_las (str): Path to input LAS file.
        output_las (str): Path to output classified LAS file.
        scalar (float): Multiplier for the mean absolute deviation (MAD) for ground threshold.
        slope (float): Maximum allowed slope between neighboring points.
        threshold (float): Maximum allowed height difference for ground classification.
        window (float): Neighborhood window size in meters (suggested: max canopy diameter).
        bounds (str, optional): Optional bounds to crop the input LAS (e.g., "([xmin,xmax],[ymin,ymax])").
    Returns:
        None
    """
    logging.info("Classifying ground points with SMRF...")
    readers_las = {
        "type": "readers.las",
        "filename": input_las
    }
    if bounds:
        readers_las["bounds"] = bounds
    ground_pipeline = [
        readers_las,
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
    Estimate average point spacing from a LAS file using header info.
    """
    logging.info("Estimating point spacing...")
    with laspy.open(las_path) as las:
        header = las.header
        x_min, x_max = header.mins[0], header.maxs[0]
        y_min, y_max = header.mins[1], header.maxs[1]
        area = (x_max - x_min) * (y_max - y_min)
        if area == 0 or header.point_count < 2:
            return 0.025  # fallback
        density = header.point_count / area
        spacing = 1 / np.sqrt(density)
        logging.info(f"Estimated point spacing: {spacing:.3f} m (density: {density:.2f} pts/m²)")
        return spacing

def create_dtm(classified_las, dtm_tif, res=None, bounds=None):
    """
    Create Digital Terrain Model (DTM) from ground-classified LAS. Uses inverse-distance weighting

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dtm_tif (str): Output path for DTM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None, estimated from point spacing.
        bounds (str, optional): Optional bounds to crop the input LAS (e.g., "([xmin,xmax],[ymin,ymax])").
    Returns:
        None
    """
    logging.info("Creating DTM...")
    if res is None:
        res = estimate_point_spacing(classified_las)
    logging.info(f"DTM resolution: {res*100:.3f} cm")
    readers_las = {
        "type": "readers.las",
        "filename": classified_las
    }
    if bounds:
        readers_las["bounds"] = bounds
    dtm_pipeline = [
        readers_las,
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
            "radius": res * 5,
            "window_size": 64,
            "data_type": "float32"
        }
    ]
    run_pdal_pipeline(dtm_pipeline)
    logging.info(f"DTM saved to {dtm_tif}")

def create_dsm(classified_las, dsm_tif, res=None, bounds=None):
    """
    Create Digital Surface Model (DSM) from ground-classified LAS.

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dsm_tif (str): Output path for DSM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None, estimated from point spacing.
        bounds (str, optional): Optional bounds to crop the input LAS (e.g., "([xmin,xmax],[ymin,ymax])").
    Returns:
        None
    """
    logging.info("Creating DSM...")
    if res is None:
        res = estimate_point_spacing(classified_las)
    logging.info(f"DSM resolution: {res*100:.3f} cm")
    readers_las = {
        "type": "readers.las",
        "filename": classified_las
    }
    if bounds:
        readers_las["bounds"] = bounds
    dsm_pipeline = [
        readers_las,
        {
            "type": "filters.range",
            "limits": "ReturnNumber[1:1]"
        },
        {
            "type": "writers.gdal",
            "filename": dsm_tif,
            "resolution": res,
            "output_type": "idw",
            "power": 2,
            "radius": res * 5,
            "window_size": 64,
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
    logging.info("Creating CHM...")
    with rasterio.open(dsm_tif) as dsm_src, rasterio.open(dtm_tif) as dtm_src:
        dsm = dsm_src.read(1)
        dtm = dtm_src.read(1)

        # Align DTM to DSM if needed
        if (dsm.shape != dtm.shape) or (dsm_src.transform != dtm_src.transform):
            logging.info("Aligning DTM to DSM before calculating...")
            aligned_dtm = np.empty_like(dsm)
            reproject(
                source=dtm,
                destination=aligned_dtm,
                src_transform=dtm_src.transform,
                src_crs=dtm_src.crs,
                dst_transform=dsm_src.transform,
                dst_crs=dsm_src.crs,
                resampling=Resampling.bilinear
            )
            logging.info("DTM aligned to DSM.")
            dtm = aligned_dtm

        chm = dsm - dtm
        chm[chm < 0] = 0  # Remove negative values
        meta = dsm_src.meta.copy()
        meta.update(dtype='float32', compress='lzw')
        with rasterio.open(chm_tif, 'w', **meta) as dst:
            dst.write(chm.astype('float32'), 1)
    logging.info(f"CHM saved to {chm_tif}")
    return chm

def preprocess_lidar(input_las, output_dir, res=None, extent_shapefile=None):
    """
    Run all preprocessing steps and return file paths.
    Args:
        input_las (str): Path to input LAS file.
        output_dir (str): Output directory for all results.
        res (float, optional): Raster resolution in meters (default 0.25).
        extent_shapefile (str, optional): Path to extent shapefile for cropping/masking.
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

    bounds = None
    if extent_shapefile:
        bounds = get_bounds_from_shapefile(extent_shapefile)

    classify_ground(input_las, ground_las, bounds=bounds)

    if res is None:
        res = estimate_point_spacing(ground_las)
        
    create_dtm(ground_las, dtm_tif, res, bounds=bounds)
    create_dsm(ground_las, dsm_tif, res, bounds=bounds)
    chm = create_chm(dsm_tif, dtm_tif, chm_tif)

    return {
        "classified_las": ground_las,
        "dtm": dtm_tif,
        "dsm": dsm_tif,
        "chm": chm_tif,
        "chm_array": chm
    }