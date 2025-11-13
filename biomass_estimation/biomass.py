"""
Estimate hazelnut bush biomass and carbon sequestration from segmented CHM rasters and bush polygons.

This module provides functions to:
1. Calculate per-bush canopy volume (m³) by overlaying bush polygons on a CHM raster,
   summing pixel-wise (x-size * y-size * height) within each polygon.
2. Estimate above-ground biomass (kg) from canopy volume using an allometric equation.
3. Estimate carbon (kg) as a fixed proportion of biomass.
4. Add error bounds to biomass and carbon estimates.
5. Return a GeoDataFrame with all results for further export or analysis.
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

VOLUME_COEF = 4.674
CARBON_CONVERSION = 0.52  # 52% of above-ground biomass is carbon
REL_RMSE = 0.193  # Relative RMSE is 19.3% for biomass estimation

def estimate_biomass(volume_m3):
    """
    Estimate above-ground biomass (kg) from canopy volume (m³) using an allometric equation.

    Args:
        volume_m3 (float or np.ndarray): Canopy volume in cubic meters.

    Returns:
        float or np.ndarray: Estimated above-ground biomass in kilograms.
    """
    return VOLUME_COEF * volume_m3

def estimate_carbon(biomass_kg):
    """
    Estimate carbon content (kg) from above-ground biomass (kg).

    Args:
        biomass_kg (float or np.ndarray): Above-ground biomass in kilograms.

    Returns:
        float or np.ndarray: Estimated carbon in kilograms.
    """
    return biomass_kg * CARBON_CONVERSION

def calculate_polygon_volumes(polygons_gdf, chm_path):
    """
    Calculate canopy volume (m³) for each polygon by overlaying polygons on a CHM raster.

    Args:
        polygons_gdf (GeoDataFrame): GeoDataFrame with bush polygons.
        chm_path (str): Path to CHM raster (GeoTIFF).

    Returns:
        GeoDataFrame: Copy of input with new 'volume_m3' column.
    """
    with rasterio.open(chm_path) as src:
        chm = src.read(1)
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])  # xres * yres (yres is negative)
        volumes = []
        for geom in polygons_gdf.geometry:
            mask = features.geometry_mask([geom], out_shape=chm.shape, transform=transform, invert=True)
            heights = chm[mask]
            heights = heights[~np.isnan(heights)]  # Remove NaN (nodata)
            volume = np.sum(heights * pixel_area)
            volumes.append(volume)
    polygons_gdf = polygons_gdf.copy()
    polygons_gdf['volume_m3'] = volumes
    return polygons_gdf

def add_biomass_and_carbon(gdf):
    """
    Add biomass and carbon columns (with relative error bounds) to a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with 'volume_m3' column.

    Returns:
        GeoDataFrame: Copy with added columns:
            - 'agb_kg', 'c_kg'
            - 'agb_kg_lo', 'agb_kg_up'
            - 'c_kg_lo', 'c_kg_up'
    """
    gdf['agb_kg'] = estimate_biomass(gdf['volume_m3']).clip(lower=0)
    gdf['c_kg'] = estimate_carbon(gdf['agb_kg'])
    gdf['agb_kg_lo'] = (gdf['agb_kg'] * (1 - REL_RMSE)).clip(lower=0)
    gdf['agb_kg_up'] = gdf['agb_kg'] * (1 + REL_RMSE)
    gdf['c_kg_lo'] = gdf['agb_kg_lo'] * CARBON_CONVERSION
    gdf['c_kg_up'] = gdf['agb_kg_up'] * CARBON_CONVERSION
    return gdf

def save_results(gdf, output_shp, output_csv):
    """
    Save the GeoDataFrame with biomass and carbon columns to a shapefile and CSV.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with biomass/carbon columns.
        output_shp (str): Output path for shapefile.
        output_csv (str): Output path for CSV (geometry dropped).

    Returns:
        None
    """
    gdf.to_file(output_shp)
    gdf.drop(columns='geometry').to_csv(output_csv, index=False)

def run_allometry(polygons_path, chm_path, output_shp, output_csv):
    """
    Full workflow: load polygons, calculate volume from CHM, add biomass/carbon,
    save results to shapefile and CSV, and return GeoDataFrame.

    Args:
        polygons_path (str): Path to bush/canopy polygons (shapefile).
        chm_path (str): Path to CHM raster (GeoTIFF).
        output_shp (str): Output path for results shapefile.
        output_csv (str): Output path for results CSV.

    Returns:
        GeoDataFrame: Results with all columns.
    """
    polygons_gdf = gpd.read_file(polygons_path)
    polygons_gdf = calculate_polygon_volumes(polygons_gdf, chm_path)
    polygons_gdf = add_biomass_and_carbon(polygons_gdf)
    save_results(polygons_gdf, output_shp, output_csv)
    return polygons_gdf