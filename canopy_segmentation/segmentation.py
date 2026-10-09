"""
Proximity-based segmentation of tree canopies from a CHM raster for
hazelnut biomass estimation.

Steps:
1. Load CHM raster (optionally crop to extent)
2. Refine bush/tree top points to local maxima within a buffer
3. Adaptive watershed segmentation using tree tops
4. Save results as shapefiles (canopy polygons and refined tree tops)
"""

import logging
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import mask as rio_mask
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine, rowcol, xy
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, shape
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed


def setup_logging():
    """
    Set up logging for the module.
    No inputs or outputs.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )


def load_chm(chm_path, extent_shapefile=None):
    """
    Load a Canopy Height Model (CHM) raster, optionally cropping to a
    shapefile extent.

    Args:
        chm_path (str): Path to the CHM raster (.tif).
        extent_shapefile (str, optional): Path to a shapefile for cropping
            extent.

    Returns:
        chm (np.ndarray): 2D array of CHM values (float64, np.nan for nodata).
        profile (dict): Rasterio profile dictionary (metadata).
        res_m_per_px (float): Pixel resolution in meters.
        extent_gdf (GeoDataFrame or None): Extent geometry if cropped,
            else None.
    """
    with rasterio.open(chm_path) as src:
        if extent_shapefile:
            gdf = gpd.read_file(extent_shapefile)
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            geoms = [geom for geom in gdf.geometry]
            arr, out_transform = rio_mask.mask(
                src, geoms, crop=True, nodata=np.nan
            )
            profile = src.profile.copy()
            profile.update(
                {
                    "height": arr.shape[1],
                    "width": arr.shape[2],
                    "transform": out_transform,
                    "nodata": np.nan,
                }
            )
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
        logging.info(
            f"Loaded CHM: shape={chm.shape}, "
            f"resolution={res_m_per_px * 100:.4f} cm/px"
        )
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
        dtype=np.uint8,
    )
    return np.where(extent_mask == 1, segments, 0)


# Largest snap radius used when none is given: the fixed radius this
# pipeline used before the radius was derived from seed spacing.
MAX_SNAP_RADIUS_M = 1.75


def snap_radius_px_from_spacing(gdf, res_m_per_px):
    """
    Derive the snap radius in pixels from the spacing of the seed points.

    The radius is half the smallest distance between any two seeds, less
    one pixel, rounded down, and never more than MAX_SNAP_RADIUS_M. Two
    seeds are then always more than two radii apart on the pixel grid, so
    their search disks cannot share a pixel. A single seed gets
    MAX_SNAP_RADIUS_M.

    Args:
        gdf (GeoDataFrame): Seed points in the CHM CRS.
        res_m_per_px (float): Raster resolution in meters per pixel.

    Raises:
        ValueError: If the two closest seeds are too close to leave a
            radius of at least one pixel. Fix the seeds, or pass
            buffer_meters to refine_tree_tops or segment_canopies.

    Returns:
        int: Snap radius in pixels.
    """
    max_radius_px = max(1, int(MAX_SNAP_RADIUS_M / res_m_per_px))
    if len(gdf) < 2:
        return max_radius_px
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    dist, neighbor = cKDTree(coords).query(coords, k=2)
    first = int(np.argmin(dist[:, 1]))
    second = int(neighbor[first, 1])
    spacing = float(dist[first, 1])
    radius_px = int(np.floor(spacing / (2 * res_m_per_px) - 1))
    if radius_px < 1:
        raise ValueError(
            f"Seeds {gdf.index[first]} and {gdf.index[second]} are "
            f"{spacing:.3f} m apart, too close to derive a snap radius at "
            f"{res_m_per_px:.3f} m per pixel; fix the seeds or pass "
            "buffer_meters."
        )
    radius_px = min(radius_px, max_radius_px)
    logging.info(
        f"Snap radius from seed spacing: {radius_px * res_m_per_px:.3f} m "
        f"(closest seeds {spacing:.3f} m apart)"
    )
    return radius_px


def refine_tree_tops(
    chm, profile, shapefile_path, buffer_meters=None, extent_gdf=None
):
    """
    Refine tree/bush top points to local maxima within a buffer on the CHM.

    Args:
        chm (np.ndarray): CHM raster array.
        profile (dict): Rasterio profile.
        shapefile_path (str): Path to input marker shapefile.
        buffer_meters (float, optional): Radius in meters of the disk
            searched for the local maximum. If None, derived from the seed
            spacing; see snap_radius_px_from_spacing.
        extent_gdf (GeoDataFrame, optional): Extent geometry for filtering.

    Raises:
        ValueError: If two seeds snap to the same pixel, which can only
            happen when buffer_meters exceeds half their spacing.

    Returns:
        tuple: (refined_rows, refined_cols, markers, refined_gdf)
            - refined_rows (np.ndarray): Row indices of refined points.
            - refined_cols (np.ndarray): Column indices of refined points.
            - markers (np.ndarray): Marker array for segmentation.
            - refined_gdf (GeoDataFrame): Refined points with attributes and
                new geometry.
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])
    if extent_gdf is not None:
        gdf = gdf[gdf.geometry.within(extent_gdf.geometry.union_all())]
    transform = profile["transform"]
    res_m_per_px = float((abs(transform.a) + abs(transform.e)) / 2.0)
    skipped = 0
    if len(gdf) == 0:
        logging.error("No valid refined tree tops found.")
        return None, None, None, None
    if buffer_meters is None:
        base_buf_px = snap_radius_px_from_spacing(gdf, res_m_per_px)
    else:
        base_buf_px = max(1, meters_to_pixels(buffer_meters, res_m_per_px))
    offsets = np.arange(-base_buf_px, base_buf_px + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= base_buf_px**2
    seed_at_pixel = {}
    refined_rows = []
    refined_cols = []
    refined_indices = []
    for idx, row in gdf.iterrows():
        pt = row.geometry
        r, c = rowcol(transform, pt.x, pt.y)
        r = int(r)
        c = int(c)
        if not (0 <= r < chm.shape[0] and 0 <= c < chm.shape[1]):
            skipped += 1
            continue
        rmin = max(0, r - base_buf_px)
        rmax = min(chm.shape[0], r + base_buf_px + 1)
        cmin = max(0, c - base_buf_px)
        cmax = min(chm.shape[1], c + base_buf_px + 1)
        window = chm[rmin:rmax, cmin:cmax]
        window_disk = disk[
            rmin - r + base_buf_px : rmax - r + base_buf_px,
            cmin - c + base_buf_px : cmax - c + base_buf_px,
        ]
        finite_mask = np.isfinite(window) & window_disk
        if not np.any(finite_mask):
            skipped += 1
            continue
        local = np.where(finite_mask, window, -np.inf)
        max_idx = np.argmax(local)
        max_local_idx = np.unravel_index(max_idx, local.shape)
        max_r = rmin + max_local_idx[0]
        max_c = cmin + max_local_idx[1]
        if (max_r, max_c) in seed_at_pixel:
            raise ValueError(
                f"Seeds {seed_at_pixel[(max_r, max_c)]} and {idx} both snap "
                f"to pixel (row {max_r}, col {max_c}); buffer_meters is "
                "larger than half their spacing."
            )
        seed_at_pixel[(max_r, max_c)] = idx
        refined_rows.append(max_r)
        refined_cols.append(max_c)
        refined_indices.append(idx)
    if len(refined_rows) == 0:
        logging.error("No valid refined tree tops found.")
        return None, None, None, None
    markers = np.zeros(chm.shape, dtype=np.int32)
    for i, (rr, cc) in enumerate(zip(refined_rows, refined_cols)):
        markers[rr, cc] = i + 1
    refined_gdf = gdf.loc[refined_indices].copy()
    refined_gdf = refined_gdf.reset_index(drop=True)
    refined_gdf['geometry'] = [
        Point(xy(transform, int(r), int(c), offset="center"))
        for r, c in zip(refined_rows, refined_cols)
    ]
    refined_gdf['tree_id'] = np.arange(1, len(refined_gdf) + 1)
    refined_gdf['height'] = [
        chm[r, c] if np.isfinite(chm[r, c]) else np.nan
        for r, c in zip(refined_rows, refined_cols)
    ]
    logging.info(f"Loaded {len(refined_gdf)} tree tops (skipped {skipped})")
    return np.array(refined_rows), np.array(refined_cols), markers, refined_gdf


def marker_watershed(
    chm,
    markers,
    profile,
    min_height=0.1,
    surface_smooth_sigma=0.5,
    extent_gdf=None,
):
    """
    Perform marker-controlled watershed segmentation on the CHM using only
    inverted height.

    Args:
        chm (np.ndarray): CHM raster array.
        markers (np.ndarray): Marker array for segmentation.
        profile (dict): Rasterio profile.
        min_height (float): Minimum CHM height to consider.
        surface_smooth_sigma (float): Gaussian smoothing sigma for CHM.
        extent_gdf (GeoDataFrame, optional): Extent geometry for masking.

    Returns:
        np.ndarray: Segmented label array.
    """
    if markers is None or np.max(markers) == 0:
        logging.error("No markers available for watershed.")
        return None

    threshold = min_height if min_height is not None else 0
    mask = np.isfinite(chm) & (chm > threshold)
    if extent_gdf is not None:
        mask = mask_segments_within_extent(
            mask.astype(np.uint8), profile, extent_gdf
        ).astype(bool)
    if not np.any(mask):
        logging.error(
            "No valid CHM pixels to segment "
            "(check minimum height threshold and extent)."
        )
        return None

    marker_ids = np.unique(markers)
    marker_ids = marker_ids[marker_ids > 0]
    marker_positions = [
        np.where(markers == marker_id) for marker_id in marker_ids
    ]
    marker_positions = [
        (pos[0][0], pos[1][0]) for pos in marker_positions if len(pos[0]) > 0
    ]

    for r, c in marker_positions:
        mask[
            max(0, r - 1) : min(mask.shape[0], r + 2),
            max(0, c - 1) : min(mask.shape[1], c + 2),
        ] = True

    smoothed_chm = gaussian(
        chm, sigma=surface_smooth_sigma, preserve_range=True
    )
    inv_height = np.where(np.isfinite(smoothed_chm), -smoothed_chm, 0.0)

    segments = watershed(inv_height, markers, connectivity=2, mask=mask)
    logging.info(f"Watershed produced {len(np.unique(segments)) - 1} segments")
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


def save_segments(
    segments,
    chm,
    profile,
    output_dir,
    res_m_per_px=1.0,
    extent_gdf=None,
    min_hole_area=8,
    min_object_size=8,
    refined_gdf=None,
):
    """
    Convert segment labels to polygons, merge with attributes, and save as a
    shapefile.

    Args:
        segments (np.ndarray): Segmented label array.
        chm (np.ndarray): CHM raster array.
        profile (dict): Rasterio profile (for CRS).
        output_dir (str): Output directory.
        res_m_per_px (float): Raster resolution in meters per pixel.
        extent_gdf (GeoDataFrame, optional): Extent geometry for masking.
        min_hole_area (int): Minimum hole area to fill in polygons.
        min_object_size (int): Minimum object size to keep in polygons.
        refined_gdf (GeoDataFrame, optional): Refined points with attributes
            for joining.

    Returns:
        None
    """
    prefix = prefix_from_chm(profile["chm_path"])
    os.makedirs(output_dir, exist_ok=True)
    polygons = []
    labels = []
    stats = []
    crossing_extent = set()
    transform = profile["transform"]
    pix_area = res_m_per_px**2
    extent_union = (
        extent_gdf.geometry.union_all() if extent_gdf is not None else None
    )
    for lbl, window in enumerate(ndimage.find_objects(segments), start=1):
        if window is None:
            continue
        mask = segments[window] == lbl
        # One pixel of padding keeps background at the window edge joined
        # to the outside, so it is not filled as if it were a hole.
        padded = np.pad(mask, 1)
        mask_clean = remove_small_objects(padded, max_size=min_object_size - 1)
        mask_clean = remove_small_holes(mask_clean, max_size=min_hole_area - 1)
        window_transform = transform @ Affine.translation(
            window[1].start - 1, window[0].start - 1
        )
        parts = []
        for geom, _ in shapes(
            mask_clean.astype(np.uint8),
            mask=mask_clean,
            transform=window_transform,
        ):
            poly = shape(geom)
            if extent_union is not None and not poly.within(extent_union):
                crossing_extent.add(lbl)
                continue
            parts.append(poly)
        if not parts:
            continue
        polygons.append(MultiPolygon(parts) if len(parts) > 1 else parts[0])
        labels.append(lbl)
        area_m2 = np.sum(mask) * pix_area
        heights = chm[window][mask]
        max_h = np.nanmax(heights) if np.any(np.isfinite(heights)) else np.nan
        mean_h = (
            np.nanmean(heights) if np.any(np.isfinite(heights)) else np.nan
        )
        stats.append((area_m2, max_h, mean_h))
    if crossing_extent:
        logging.warning(
            f"{len(crossing_extent)} segments reach the extent boundary and "
            "had the parts not fully inside it left out; tree_id "
            f"{sorted(crossing_extent)}, of which "
            f"{sorted(crossing_extent - set(labels))} were dropped entirely"
        )
    if len(polygons) == 0:
        logging.warning("No polygons generated from segments.")
        return
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
                # Only update non-geometry attributes to avoid overwriting
                # the polygon geometry
                attrs_no_geom = {
                    k: v for k, v in attrs.items() if k != "geometry"
                }
                row.update(attrs_no_geom)
            data.append(row)
        gdf = gpd.GeoDataFrame(data, crs=profile["crs"])
    else:
        gdf = gpd.GeoDataFrame(
            {
                "tree_id": labels,
                "geometry": polygons,
                "area_m2": [s[0] for s in stats],
                "max_h": [s[1] for s in stats],
                "mean_h": [s[2] for s in stats],
            },
            crs=profile["crs"],
        )
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
    buffer_meters=None,
    surface_smooth_sigma=0.5,
    min_height=0.1,
):
    """
    Full pipeline: Load CHM, refine tree tops, segment canopies, and save
    outputs.

    Args:
        chm_path (str): Path to CHM raster.
        tree_tops_shp (str): Path to input marker shapefile.
        output_dir (str, optional): Output directory.
        extent_shapefile (str, optional): Path to extent shapefile for
            cropping/masking.
        buffer_meters (float, optional): Radius for local maxima search
            (meters). If None, derived from the seed spacing; see
            refine_tree_tops.
        surface_smooth_sigma (float): Gaussian smoothing sigma for CHM.
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
    chm, profile, res_m_per_px, extent_gdf = load_chm(
        chm_path, extent_shapefile
    )
    _, _, markers, refined_gdf = refine_tree_tops(
        chm,
        profile,
        tree_tops_shp,
        buffer_meters=buffer_meters,
        extent_gdf=extent_gdf,
    )
    if markers is None:
        logging.error("No valid refined tree tops found; exiting.")
        return None
    if extent_gdf is not None:
        markers = mask_markers_within_extent(markers, profile, extent_gdf)
    segments = marker_watershed(
        chm,
        markers,
        profile,
        min_height=min_height,
        surface_smooth_sigma=surface_smooth_sigma,
        extent_gdf=extent_gdf,
    )
    if segments is None:
        logging.error("Segmentation failed.")
        return None
    if output_dir is None:
        output_dir = os.path.dirname(chm_path) or "."
    save_segments(
        segments,
        chm,
        profile,
        output_dir,
        res_m_per_px,
        extent_gdf=extent_gdf,
        refined_gdf=refined_gdf,
    )
    save_refined_tree_tops(refined_gdf, profile, output_dir)
    return {
        "segments": segments,
        "markers": markers,
        "chm": chm,
        "profile": profile,
        "output_dir": output_dir,
    }
