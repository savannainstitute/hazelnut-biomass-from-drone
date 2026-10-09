"""
Preprocess LiDAR point cloud data for hazelnut biomass estimation using
PDAL and rasterio.

Steps:
1. Ground classification (PDAL)
2. Digital Terrain Model (DTM) generation (PDAL)
3. Digital Surface Model (DSM) generation (PDAL)
4. Canopy Height Model (CHM) calculation (DSM - DTM)
5. Save outputs as GeoTIFFs (rasterio)
"""

import logging
import os
import subprocess

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def run_pdal_pipeline(pipeline_json):
    """
    Run a PDAL pipeline from a JSON object.

    Args:
        pipeline_json (dict or list): PDAL pipeline definition as a Python
            object.

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
        result = subprocess.run(
            ['pdal', 'pipeline', pipeline_path], capture_output=True, text=True
        )
        if result.returncode != 0:
            logging.error(f"PDAL pipeline failed: {result.stderr}")
            raise RuntimeError(f"PDAL pipeline failed: {result.stderr}")
        else:
            logging.info(f"PDAL pipeline succeeded: {result.stdout}")
    finally:
        os.remove(pipeline_path)


# The SMRF window default, documented as the largest canopy diameter.
MAX_CANOPY_DIAMETER_M = 2.5


def get_crop_polygon_wkt(shapefile_path, las_path, margin_m=0.0):
    """
    Return the extent polygon as WKT in the CRS of the LAS file.

    All features in the shapefile are unioned into one geometry and
    reprojected to the LAS CRS when the two differ.

    Args:
        shapefile_path (str): Path to the extent polygon shapefile.
        las_path (str): Path to the LAS file the polygon will crop.
        margin_m (float): Distance in meters to grow the polygon by, so a
            bush seeded just inside the extent keeps the points of its
            whole crown.

    Raises:
        ValueError: If the shapefile or the LAS file has no CRS. Assign one
            with geopandas `set_crs` or PDAL `filters.assign` first.

    Returns:
        str: WKT of the extent polygon in the LAS CRS.
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError(
            f"Extent shapefile has no CRS: {shapefile_path}. Assign one "
            "with geopandas set_crs before cropping."
        )
    with laspy.open(las_path) as las:
        las_crs = las.header.parse_crs()
    if las_crs is None:
        raise ValueError(
            f"LAS file has no CRS: {las_path}. Assign one with PDAL "
            "filters.assign before cropping."
        )
    if gdf.crs != las_crs:
        logging.info(f"Reprojecting extent from {gdf.crs} to the LAS CRS")
        gdf = gdf.to_crs(las_crs)
    return gdf.geometry.union_all().buffer(margin_m).wkt


def classify_ground(
    input_las,
    output_las,
    scalar=1.2,
    slope=0.15,
    threshold=0.07,
    window=MAX_CANOPY_DIAMETER_M,
    crop_polygon=None,
):
    """
    Classify ground points using PDAL SMRF filter. https://pdal.io/en/stable/stages/filters.smrf.html

    Args:
        input_las (str): Path to input LAS file.
        output_las (str): Path to output classified LAS file.
        scalar (float): Multiplier for the mean absolute deviation (MAD) for
            ground threshold.
        slope (float): Maximum allowed slope between neighboring points.
        threshold (float): Maximum allowed height difference for ground
            classification.
        window (float): Neighborhood window size in meters (suggested: max
            canopy diameter).
        crop_polygon (str, optional): WKT polygon in the LAS CRS. Points
            outside it are dropped before classification; see
            get_crop_polygon_wkt.
    Returns:
        None
    """
    logging.info("Classifying ground points with SMRF...")
    ground_pipeline = [{"type": "readers.las", "filename": input_las}]
    if crop_polygon:
        ground_pipeline.append(
            {"type": "filters.crop", "polygon": crop_polygon}
        )
    ground_pipeline += [
        {
            "type": "filters.smrf",
            "scalar": scalar,
            "slope": slope,
            "threshold": threshold,
            "window": window,
        },
        {"type": "writers.las", "filename": output_las},
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
        logging.info(
            f"Estimated point spacing: {spacing:.3f} m "
            f"(density: {density:.2f} pts/m²)"
        )
        return spacing


def create_dtm(classified_las, dtm_tif, res=None):
    """
    Create Digital Terrain Model (DTM) from ground-classified LAS. Uses
    inverse-distance weighting

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dtm_tif (str): Output path for DTM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None,
            estimated from point spacing.
    Returns:
        None
    """
    logging.info("Creating DTM...")
    if res is None:
        res = estimate_point_spacing(classified_las)
    logging.info(f"DTM resolution: {res * 100:.3f} cm")
    dtm_pipeline = [
        {"type": "readers.las", "filename": classified_las},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {
            "type": "writers.gdal",
            "filename": dtm_tif,
            "resolution": res,
            "output_type": "idw",
            "power": 2,
            "radius": res * 10,
            "window_size": 128,
            "data_type": "float32",
        },
    ]
    run_pdal_pipeline(dtm_pipeline)
    logging.info(f"DTM saved to {dtm_tif}")


def create_dsm(classified_las, dsm_tif, res=None):
    """
    Create Digital Surface Model (DSM) from ground-classified LAS.

    Args:
        classified_las (str): Path to ground-classified LAS file.
        dsm_tif (str): Output path for DSM GeoTIFF.
        res (float, optional): Raster resolution in meters. If None,
            estimated from point spacing.
    Returns:
        None
    """
    logging.info("Creating DSM...")
    if res is None:
        res = estimate_point_spacing(classified_las)
    logging.info(f"DSM resolution: {res * 100:.3f} cm")
    dsm_pipeline = [
        {"type": "readers.las", "filename": classified_las},
        {"type": "filters.range", "limits": "ReturnNumber[1:1]"},
        {
            "type": "writers.gdal",
            "filename": dsm_tif,
            "resolution": res,
            "output_type": "idw",
            "power": 2,
            "radius": res * 10,
            "window_size": 128,
            "data_type": "float32",
        },
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
        np.ndarray: CHM array (DSM - DTM), NaN where either input has no
            data.
    """
    logging.info("Creating CHM...")
    with rasterio.open(dsm_tif) as dsm_src, rasterio.open(dtm_tif) as dtm_src:
        dsm = dsm_src.read(1).astype('float64')
        dtm = dtm_src.read(1).astype('float64')
        if dsm_src.nodata is not None:
            dsm[dsm == dsm_src.nodata] = np.nan
        if dtm_src.nodata is not None:
            dtm[dtm == dtm_src.nodata] = np.nan

        # Align DTM to DSM if needed
        if (dsm.shape != dtm.shape) or (
            dsm_src.transform != dtm_src.transform
        ):
            logging.info("Aligning DTM to DSM before calculating...")
            aligned_dtm = np.full(dsm.shape, np.nan)
            reproject(
                source=dtm,
                destination=aligned_dtm,
                src_transform=dtm_src.transform,
                src_crs=dtm_src.crs,
                src_nodata=np.nan,
                dst_transform=dsm_src.transform,
                dst_crs=dsm_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            logging.info("DTM aligned to DSM.")
            dtm = aligned_dtm

        chm = dsm - dtm
        chm[chm < 0] = 0  # Remove negative values
        n_nodata = int(np.isnan(chm).sum())
        if n_nodata:
            logging.info(
                f"{n_nodata} CHM cells have no DSM or DTM value and are "
                "written as nodata"
            )
        meta = dsm_src.meta.copy()
        meta.update(dtype='float32', compress='lzw', nodata=np.nan)
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
        extent_shapefile (str, optional): Path to extent shapefile. The
            cloud is cropped to it grown by MAX_CANOPY_DIAMETER_M.
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

    crop_polygon = None
    if extent_shapefile:
        crop_polygon = get_crop_polygon_wkt(
            extent_shapefile, input_las, margin_m=MAX_CANOPY_DIAMETER_M
        )

    classify_ground(input_las, ground_las, crop_polygon=crop_polygon)

    if res is None:
        # A cropped cloud keeps a bounding-box header, which understates
        # its density, so a cropped run takes the spacing from the input.
        res = estimate_point_spacing(input_las if crop_polygon else ground_las)

    create_dtm(ground_las, dtm_tif, res)
    create_dsm(ground_las, dsm_tif, res)
    chm = create_chm(dsm_tif, dtm_tif, chm_tif)

    return {
        "classified_las": ground_las,
        "dtm": dtm_tif,
        "dsm": dsm_tif,
        "chm": chm_tif,
        "chm_array": chm,
    }
