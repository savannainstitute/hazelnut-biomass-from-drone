"""
Proximity-based segmentation of tree canopies from a CHM raster for hazelnut biomass estimation.

Steps:
1. Load CHM raster (optionally crop to extent)
2. Refine bush/tree top points to local maxima within a buffer
3. Adaptive watershed segmentation using tree tops
4. Save results as shapefiles (canopy polygons and refined tree tops)
"""

import os
import logging
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes, rasterize
from rasterio import mask as rio_mask
from rasterio.transform import rowcol, xy
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import gaussian
from shapely.geometry import shape, Point

def setup_logging():
    """
    Set up logging for the module.
    No inputs or outputs.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_chm(chm_path, extent_shapefile=None):
    """
    Load a Canopy Height Model (CHM) raster, optionally cropping to a shapefile extent.

    Args:
        chm_path (str): Path to the CHM raster (.tif).
        extent_shapefile (str, optional): Path to a shapefile for cropping extent.

    Returns:
        chm (np.ndarray): 2D array of CHM values (float64, np.nan for nodata).
        profile (dict): Rasterio profile dictionary (metadata).
        res_m_per_px (float): Pixel resolution in meters.
        extent_gdf (GeoDataFrame or None): Extent geometry if cropped, else None.
    """
    with rasterio.open(chm_path) as src:
        if extent_shapefile:
            gdf = gpd.read_file(extent_shapefile)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            geoms = [geom for geom in gdf.geometry]
            arr, out_transform = rio_mask.mask(src, geoms, crop=True, nodata=np.nan)
            profile = src.profile.copy()
            profile.update({
                "height": arr.shape[1],
                "width": arr.shape[2],
                "transform": out_transform,
                "nodata": np.nan
            })
            chm = arr[0].astype(np.float64)
            extent_gdf = gdf
        else:
            chm = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            extent_gdf = None
        nodata = profile.get("nodata", None)
        if nodata is not None:
            chm = np.where(chm == nodata, np.nan, chm)
        transform = profile["transform"]
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        res_m_per_px = float((px_w + px_h) / 2.0)
        profile["chm_path"] = chm_path
        logging.info(f"Loaded CHM: shape={chm.shape}, resolution={res_m_per_px:.4f} m/px")
        return chm, profile, res_m_per_px, extent_gdf

def meters_to_pixels(distance_meters, res_m_per_px):
    """
    Convert a distance in meters to pixels, given the raster resolution.

    Args:
        distance_meters (float): Distance in meters.
        res_m_per_px (float): Raster resolution in meters per pixel.

    Returns:
        int: Distance in pixels (rounded).
    """
    return int(round(distance_meters / res_m_per_px))

def mask_markers_within_extent(markers, profile, extent_gdf):
    """
    Mask out marker pixels that fall outside the extent geometry.

    Args:
        markers (np.ndarray): 2D marker array.
        profile (dict): Rasterio profile.
        extent_gdf (GeoDataFrame): Extent geometry.

    Returns:
        np.ndarray: Masked marker array.
    """
    if extent_gdf is None:
        return markers
    transform = profile["transform"]
    rows, cols = np.where(markers > 0)
    for r, c in zip(rows, cols):
        x, y = xy(transform, int(r), int(c), offset="center")
        pt = Point(x, y)
        if not any(extent_gdf.contains(pt)):
            markers[r, c] = 0
    return markers

def mask_segments_within_extent(segments, profile, extent_gdf):
    """
    Mask out segment pixels that fall outside the extent geometry.

    Args:
        segments (np.ndarray): 2D segment label array.
        profile (dict): Rasterio profile.
        extent_gdf (GeoDataFrame): Extent geometry.

    Returns:
        np.ndarray: Masked segment array.
    """
    if extent_gdf is None:
        return segments
    transform = profile["transform"]
    mask_shape = segments.shape
    extent_mask = rasterize(
        [(geom, 1) for geom in extent_gdf.geometry],
        out_shape=mask_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return np.where(extent_mask == 1, segments, 0)

def refine_tree_tops(chm, profile, shapefile_path, buffer_meters=1.75, extent_gdf=None):
    """
    Refine tree/bush top points to local maxima within a buffer on the CHM.

    Args:
        chm (np.ndarray): CHM raster array.
        profile (dict): Rasterio profile.
        shapefile_path (str): Path to input marker shapefile.
        buffer_meters (float): Buffer radius in meters for local maxima search.
        extent_gdf (GeoDataFrame, optional): Extent geometry for filtering.

    Returns:
        tuple: (refined_rows, refined_cols, markers, refined_gdf)
            - refined_rows (np.ndarray): Row indices of refined points.
            - refined_cols (np.ndarray): Column indices of refined points.
            - markers (np.ndarray): Marker array for segmentation.
            - refined_gdf (GeoDataFrame): Refined points with attributes and new geometry.
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])
    if extent_gdf is not None:
        gdf = gdf[gdf.geometry.within(extent_gdf.unary_union)]
    transform = profile["transform"]
    res_m_per_px = float((abs(transform.a) + abs(transform.e)) / 2.0)
    skipped = 0
    base_buf_px = max(1, meters_to_pixels(buffer_meters, res_m_per_px))
    refined_rows = []
    refined_cols = []
    refined_indices = []
    for idx, row in gdf.iterrows():
        pt = row.geometry
        r, c = rowcol(transform, pt.x, pt.y)
        r = int(r); c = int(c)
        if not (0 <= r < chm.shape[0] and 0 <= c < chm.shape[1]):
            skipped += 1
            continue
        rmin = max(0, r - base_buf_px)
        rmax = min(chm.shape[0], r + base_buf_px + 1)
        cmin = max(0, c - base_buf_px)
        cmax = min(chm.shape[1], c + base_buf_px + 1)
        window = chm[rmin:rmax, cmin:cmax]
        finite_mask = np.isfinite(window)
        if not np.any(finite_mask):
            skipped += 1
            continue
        local = np.where(finite_mask, window, -np.inf)
        max_idx = np.argmax(local)
        max_local_idx = np.unravel_index(max_idx, local.shape)
        max_r = rmin + max_local_idx[0]
        max_c = cmin + max_local_idx[1]
        refined_rows.append(max_r)
        refined_cols.append(max_c)
        refined_indices.append(idx)
    if len(refined_rows) == 0:
        logging.error("No valid refined tree tops found.")
        return None, None, None, None 
    markers = np.zeros(chm.shape, dtype=np.int32)
    for i, (rr, cc) in enumerate(zip(refined_rows, refined_cols)):
        markers[rr, cc] = i + 1
    refined_gdf = gdf.iloc[refined_indices].copy()
    refined_gdf = refined_gdf.reset_index(drop=True)
    refined_gdf['geometry'] = [
        Point(xy(transform, int(r), int(c), offset="center")) for r, c in zip(refined_rows, refined_cols)
    ]
    refined_gdf['tree_id'] = np.arange(1, len(refined_gdf) + 1)
    refined_gdf['height'] = [chm[r, c] if np.isfinite(chm[r, c]) else np.nan for r, c in zip(refined_rows, refined_cols)]
    logging.info(f"Loaded {len(refined_gdf)} tree tops (skipped {skipped})")
    return np.array(refined_rows), np.array(refined_cols), markers, refined_gdf

def adaptive_watershed(
    chm, markers, res_m_per_px, profile, min_height=0.1, base_height_factor=0.8, height_factor_scale=0.2,
    penalty_strength=0.1, boundary_penalty_weight=1.0, gradient_weight=0.4, surface_smooth_sigma=0.5,
    compactness=0.0001, extent_gdf=None
):
    """
    Perform adaptive marker-controlled watershed segmentation on the CHM.

    Args:
        chm (np.ndarray): CHM raster array.
        markers (np.ndarray): Marker array for segmentation.
        res_m_per_px (float): Raster resolution in meters per pixel.
        profile (dict): Rasterio profile.
        min_height (float, optional): Minimum CHM height to consider.
        base_height_factor, height_factor_scale, penalty_strength, boundary_penalty_weight,
        gradient_weight, surface_smooth_sigma, compactness: Watershed parameters.
        extent_gdf (GeoDataFrame, optional): Extent geometry for masking.

    Returns:
        np.ndarray or None: Segmented label array, or None if failed.
    """
    if markers is None or np.max(markers) == 0:
        logging.error("No markers available for watershed.")
        return None

    mask = np.isfinite(chm) & (chm > (min_height if min_height is not None else 0))
    if extent_gdf is not None:
        mask = mask_segments_within_extent(mask.astype(np.uint8), profile, extent_gdf).astype(bool)
    if not np.any(mask):
        logging.error("No valid CHM pixels to segment (check minimum height threshold and extent).")
        return None
    marker_positions = []
    marker_heights = []
    marker_ids = np.unique(markers)
    marker_ids = marker_ids[marker_ids > 0]
    for marker_id in marker_ids:
        pos = np.where(markers == marker_id)
        if len(pos[0]) > 0:
            r, c = pos[0][0], pos[1][0]
            marker_positions.append((r, c))
            marker_heights.append(chm[r, c])
    marker_positions = np.array(marker_positions)
    marker_heights = np.array(marker_heights)
    smoothed_chm = gaussian(chm, sigma=surface_smooth_sigma, preserve_range=True)
    grad_y, grad_x = np.gradient(smoothed_chm)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mag = np.where(np.isfinite(gradient_mag), gradient_mag, 0)
    height_range = np.nanmax(smoothed_chm) - np.nanmin(smoothed_chm[np.isfinite(smoothed_chm)])
    proximity_penalty = np.full_like(smoothed_chm, np.inf)
    def path_crosses_low_height(start_r, start_c, end_r, end_c):
        from skimage.draw import line
        line_r, line_c = line(start_r, start_c, end_r, end_c)
        for lr, lc in zip(line_r, line_c):
            if (0 <= lr < chm.shape[0] and 0 <= lc < chm.shape[1]):
                val = chm[lr, lc]
                if np.isfinite(val) and min_height is not None and val < min_height:
                    return True
        return False
    for i, marker_id in enumerate(marker_ids):
        marker_pos = marker_positions[i]
        marker_height = marker_heights[i]
        max_search_distance = 25.0
        distances_to_others = []
        neighbor_info = []
        for j, other_pos in enumerate(marker_positions):
            if i != j:
                dist_px = np.sqrt((marker_pos[0] - other_pos[0])**2 + (marker_pos[1] - other_pos[1])**2)
                dist_m = dist_px * res_m_per_px
                if (dist_m <= max_search_distance and not path_crosses_low_height(marker_pos[0], marker_pos[1], other_pos[0], other_pos[1])):
                    distances_to_others.append(dist_m)
                    neighbor_info.append((j, dist_m, marker_heights[j]))
        if len(distances_to_others) > 0:
            nearest_idx = np.argmin(distances_to_others)
            nearest_distance = distances_to_others[nearest_idx]
            nearest_neighbor_info = neighbor_info[nearest_idx]
            nearest_neighbor_height = nearest_neighbor_info[2]
            height_ratio = marker_height / nearest_neighbor_height
            height_factor = base_height_factor + height_factor_scale * np.clip(height_ratio, 0.5, 1.5)
            local_characteristic_distance = (nearest_distance / 2.0) * height_factor
            logging.info(f"Tree {marker_id}: height={marker_height:.1f}m, neighbor_dist={nearest_distance:.1f}m, "
                         f"neighbor_height={nearest_neighbor_height:.1f}m, height_factor={height_factor:.2f}, "
                         f"boundary_dist={local_characteristic_distance:.1f}m")
            marker_mask = (markers == marker_id)
            distance_from_this_marker = ndimage.distance_transform_edt(~marker_mask)
            distance_m = distance_from_this_marker * res_m_per_px
            local_penalty = penalty_strength * height_range * (1 - np.exp(-distance_m / local_characteristic_distance))
            proximity_penalty = np.minimum(proximity_penalty, local_penalty)
        else:
            logging.info(f"Tree {marker_id}: isolated, using natural watershed boundaries")
            continue

    proximity_penalty = np.where(np.isinf(proximity_penalty), penalty_strength * height_range, proximity_penalty)
    inv_height = np.where(np.isfinite(smoothed_chm), -smoothed_chm, 0.0)
    surface = inv_height + (boundary_penalty_weight * proximity_penalty) + (gradient_weight * gradient_mag)
    segments = watershed(
        surface,
        markers,
        connectivity=2,
        compactness=compactness,
        mask=mask
    )
    logging.info(f"Adaptive watershed produced {len(np.unique(segments)) - 1} segments")
    return segments

def save_refined_tree_tops(refined_gdf, profile, output_dir):
    """
    Save the refined tree/bush top points as a shapefile.

    Args:
        refined_gdf (GeoDataFrame): Refined points with attributes.
        profile (dict): Rasterio profile (for CRS).
        output_dir (str): Output directory.

    Returns:
        None
    """
    prefix = prefix_from_chm(profile["chm_path"])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_treetops.shp")
    remove_shapefile_if_exists(out_path)
    refined_gdf.to_file(out_path)
    logging.info(f"Saved refined treetops: {out_path}")

def save_segments(segments, chm, profile, output_dir, res_m_per_px=1.0, extent_gdf=None,
                  min_hole_area=8, min_object_size=8, refined_gdf=None):
    """
    Convert segment labels to polygons, merge with attributes, and save as a shapefile.

    Args:
        segments (np.ndarray): Segmented label array.
        chm (np.ndarray): CHM raster array.
        profile (dict): Rasterio profile (for CRS).
        output_dir (str): Output directory.
        res_m_per_px (float): Raster resolution in meters per pixel.
        extent_gdf (GeoDataFrame, optional): Extent geometry for masking.
        min_hole_area (int): Minimum hole area to fill in polygons.
        min_object_size (int): Minimum object size to keep in polygons.
        refined_gdf (GeoDataFrame, optional): Refined points with attributes for joining.

    Returns:
        None
    """
    prefix = prefix_from_chm(profile["chm_path"])
    os.makedirs(output_dir, exist_ok=True)
    polygons = []
    labels = []
    transform = profile["transform"]
    for val in np.unique(segments):
        if val == 0:
            continue
        mask = (segments == val)
        mask_clean = remove_small_objects(mask, min_size=min_object_size)
        mask_clean = remove_small_holes(mask_clean, area_threshold=min_hole_area)
        for geom, _ in shapes(mask_clean.astype(np.uint8), mask=mask_clean, transform=transform):
            poly = shape(geom)
            if extent_gdf is not None and not poly.within(extent_gdf.unary_union):
                continue
            polygons.append(poly)
            labels.append(int(val))
            break
    if len(polygons) == 0:
        logging.warning("No polygons generated from segments.")
        return
    pix_area = res_m_per_px ** 2
    stats = []
    for lbl in labels:
        mask = segments == lbl
        area_m2 = np.sum(mask) * pix_area
        heights = chm[mask]
        max_h = np.nanmax(heights) if np.any(np.isfinite(heights)) else np.nan
        mean_h = np.nanmean(heights) if np.any(np.isfinite(heights)) else np.nan
        stats.append((area_m2, max_h, mean_h))
    if refined_gdf is not None and 'tree_id' in refined_gdf.columns:
        attr_gdf = refined_gdf.set_index('tree_id')
        data = []
        for i, lbl in enumerate(labels):
            if lbl in attr_gdf.index:
                attrs = attr_gdf.loc[lbl]
                if hasattr(attrs, "to_dict"):
                    attrs = attrs.to_dict()
            else:
                attrs = {}
            row = {
                "tree_id": lbl,
                "geometry": polygons[i],
                "area_m2": stats[i][0],
                "max_h": stats[i][1],
                "mean_h": stats[i][2],
            }
            if isinstance(attrs, dict):
                # Only update non-geometry attributes to avoid overwriting the polygon geometry
                attrs_no_geom = {k: v for k, v in attrs.items() if k != "geometry"}
                row.update(attrs_no_geom)
            data.append(row)
        gdf = gpd.GeoDataFrame(data, crs=profile["crs"])
    else:
        gdf = gpd.GeoDataFrame({
            "tree_id": labels,
            "geometry": polygons,
            "area_m2": [s[0] for s in stats],
            "max_h": [s[1] for s in stats],
            "mean_h": [s[2] for s in stats]
        }, crs=profile["crs"])
    out_path = os.path.join(output_dir, f"{prefix}_segments.shp")
    remove_shapefile_if_exists(out_path)
    gdf.to_file(out_path)
    logging.info(f"Saved canopy polygons: {out_path} ({len(gdf)} features)")

def remove_shapefile_if_exists(path_shp):
    """
    Remove all files associated with a shapefile (by basename).

    Args:
        path_shp (str): Path to the .shp file.

    Returns:
        None
    """
    base, _ = os.path.splitext(path_shp)
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"]
    for e in exts:
        p = base + e
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def prefix_from_chm(chm_path):
    """
    Get a filename prefix from a CHM raster path.

    Args:
        chm_path (str): Path to the CHM raster.

    Returns:
        str: Prefix for output files.
    """
    name = os.path.basename(chm_path)
    if name.lower().endswith("_chm.tif"):
        return name[:-8]
    elif name.lower().endswith(".tif"):
        return name[:-4]
    else:
        return os.path.splitext(name)[0]

def segment_canopies(
    chm_path,
    tree_tops_shp,
    output_dir=None,
    extent_shapefile=None,
    buffer_meters=1.75,
    base_height_factor=0.8,
    height_factor_scale=0.2,
    penalty_strength=0.1,
    boundary_penalty_weight=1.0,
    gradient_weight=0.4,
    surface_smooth_sigma=0.5,
    compactness=0.0001,
    min_height=0.1
):
    """
    Full pipeline: Load CHM, refine tree tops, segment canopies, and save outputs.

    Args:
        chm_path (str): Path to CHM raster.
        tree_tops_shp (str): Path to input marker shapefile.
        output_dir (str, optional): Output directory.
        extent_shapefile (str, optional): Path to extent shapefile for cropping/masking.
        buffer_meters (float): Buffer for local maxima search (meters).
        base_height_factor (float): Base scaling factor for adaptive boundary distance.
        height_factor_scale (float): Scaling factor for height ratio adjustment.
        penalty_strength (float): Strength of proximity penalty in segmentation.
        boundary_penalty_weight (float): Weight for proximity penalty in surface.
        gradient_weight (float): Weight for CHM gradient in surface.
        surface_smooth_sigma (float): Gaussian smoothing sigma for CHM.
        compactness (float): Compactness parameter for watershed.
        min_height (float): Minimum CHM height to consider (meters).

    Returns:
        dict: {
            "segments": segments,
            "markers": markers,
            "chm": chm,
            "profile": profile,
            "output_dir": output_dir
        }
    """
    setup_logging()
    chm, profile, res_m_per_px, extent_gdf = load_chm(chm_path, extent_shapefile)
    _, _, markers, refined_gdf = refine_tree_tops(
        chm, profile, tree_tops_shp, buffer_meters=buffer_meters, extent_gdf=extent_gdf
    )
    if extent_gdf is not None:
        markers = mask_markers_within_extent(markers, profile, extent_gdf)
    if markers is None:
        logging.error("No valid refined tree tops found; exiting.")
        return None
    segments = adaptive_watershed(
        chm, markers, res_m_per_px, profile,
        min_height=min_height,
        base_height_factor=base_height_factor,
        height_factor_scale=height_factor_scale,
        penalty_strength=penalty_strength,
        boundary_penalty_weight=boundary_penalty_weight,
        gradient_weight=gradient_weight,
        surface_smooth_sigma=surface_smooth_sigma,
        compactness=compactness,
        extent_gdf=extent_gdf
    )
    if segments is None:
        logging.error("Segmentation failed.")
        return None
    if output_dir is None:
        output_dir = os.path.dirname(chm_path) or "."
    save_segments(
        segments, chm, profile, output_dir, res_m_per_px, extent_gdf=extent_gdf, refined_gdf=refined_gdf
    )
    save_refined_tree_tops(refined_gdf, profile, output_dir)
    return {
        "segments": segments,
        "markers": markers,
        "chm": chm,
        "profile": profile,
        "output_dir": output_dir
    }