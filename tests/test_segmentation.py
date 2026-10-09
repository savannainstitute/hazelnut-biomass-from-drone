"""
Tests for canopy_segmentation.segmentation on small planted rasters whose
correct answer is known by construction.

Run from the repository root with: python -m pytest tests
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from biomass_estimation import biomass
from canopy_segmentation import segmentation

RES = 0.1
CRS = "EPSG:32616"
TRANSFORM = from_origin(500000.0, 4000000.0, RES, RES)


def write_chm(path, chm):
    """Write a planted CHM array as a GeoTIFF and return its path."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=chm.shape[0],
        width=chm.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
    ) as dst:
        dst.write(chm.astype("float32"), 1)
    return str(path)


def write_seeds(path, rowcols):
    """Write one seed point at the center of each (row, col) pixel."""
    points = [
        Point(*rasterio.transform.xy(TRANSFORM, row, col))
        for row, col in rowcols
    ]
    gpd.GeoDataFrame(
        {"plant": [f"plant_{i}" for i in range(len(points))]},
        geometry=points,
        crs=CRS,
    ).to_file(path)
    return str(path)


def add_cone(chm, row, col, radius_px, height):
    """Add a crown that falls linearly from height at its center to 0.2 m."""
    rows, cols = np.ogrid[: chm.shape[0], : chm.shape[1]]
    dist = np.hypot(rows - row, cols - col)
    inside = dist <= radius_px
    crown = height - (height - 0.2) * dist / radius_px
    chm[inside] = np.maximum(chm[inside], crown[inside])


def seeds_gdf(coords):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=CRS)


def test_snap_radius_is_half_the_smallest_spacing_less_one_pixel():
    seeds = seeds_gdf([(0.0, 0.0), (3.05, 0.0), (10.0, 5.0)])
    # 3.05 m / (2 * 0.1 m) - 1 = 14.25 pixels, rounded down
    assert segmentation.snap_radius_px_from_spacing(seeds, RES) == 14


def test_snap_radius_is_capped_for_widely_spaced_seeds():
    seeds = seeds_gdf([(0.0, 0.0), (30.0, 0.0)])
    # MAX_SNAP_RADIUS_M of 1.75 m at 0.1 m per pixel
    assert segmentation.snap_radius_px_from_spacing(seeds, RES) == 17


def test_single_seed_gets_the_capped_radius():
    seeds = seeds_gdf([(0.0, 0.0)])
    assert segmentation.snap_radius_px_from_spacing(seeds, RES) == 17


def test_snap_radius_names_seeds_that_are_too_close():
    seeds = seeds_gdf([(0.0, 0.0), (0.15, 0.0), (10.0, 5.0)])
    with pytest.raises(ValueError, match="Seeds 0 and 1"):
        segmentation.snap_radius_px_from_spacing(seeds, RES)


def test_derived_radius_keeps_each_seed_on_its_own_bush(tmp_path):
    chm = np.zeros((100, 100))
    add_cone(chm, 50, 40, 6, 2.0)
    add_cone(chm, 50, 55, 5, 0.8)
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    seeds = write_seeds(tmp_path / "seeds.shp", [(50, 40), (50, 55)])
    chm, profile, _, _ = segmentation.load_chm(chm_path)
    _, _, markers, refined = segmentation.refine_tree_tops(chm, profile, seeds)
    assert refined.height.tolist() == pytest.approx([2.0, 0.8])
    assert markers[50, 40] == 1
    assert markers[50, 55] == 2


def test_two_seeds_snapping_to_one_pixel_raise_and_are_named(tmp_path):
    chm = np.zeros((100, 100))
    add_cone(chm, 50, 40, 6, 2.0)
    add_cone(chm, 50, 55, 5, 0.8)
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    seeds = write_seeds(tmp_path / "seeds.shp", [(50, 40), (50, 55)])
    chm, profile, _, _ = segmentation.load_chm(chm_path)
    with pytest.raises(ValueError, match="Seeds 0 and 1 both snap"):
        segmentation.refine_tree_tops(chm, profile, seeds, buffer_meters=1.75)


def test_snap_search_is_a_disk_not_a_square(tmp_path):
    chm = np.zeros((80, 80))
    add_cone(chm, 40, 40, 3, 0.4)
    # 16 pixels away on both axes: inside a 1.75 m square, 2.26 m away
    add_cone(chm, 56, 56, 3, 1.6)
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    seeds = write_seeds(tmp_path / "seeds.shp", [(40, 40)])
    chm, profile, _, _ = segmentation.load_chm(chm_path)
    _, _, _, refined = segmentation.refine_tree_tops(
        chm, profile, seeds, buffer_meters=1.75
    )
    assert refined.height.iloc[0] == pytest.approx(0.4)


def save_and_read_segments(tmp_path, segments, chm):
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    profile = {
        "transform": TRANSFORM,
        "crs": rasterio.crs.CRS.from_string(CRS),
        "chm_path": chm_path,
    }
    segmentation.save_segments(
        segments, chm, profile, str(tmp_path), res_m_per_px=RES
    )
    return gpd.read_file(tmp_path / "planted_segments.shp"), chm_path


def test_segment_in_two_pieces_keeps_both(tmp_path):
    segments = np.zeros((40, 40), dtype=np.int32)
    segments[5:15, 5:15] = 1
    segments[20:30, 22:32] = 1
    chm = np.where(segments == 1, 1.0, 0.0)
    written, chm_path = save_and_read_segments(tmp_path, segments, chm)
    assert len(written) == 1
    assert written.geometry.iloc[0].geom_type == "MultiPolygon"
    assert written.geometry.area.iloc[0] == pytest.approx(200 * RES**2)
    volumes = biomass.calculate_polygon_volumes(written, chm_path)
    assert volumes.volume_m3.iloc[0] == pytest.approx(200 * RES**2 * 1.0)


def test_background_at_the_window_edge_is_not_filled(tmp_path):
    segments = np.zeros((30, 30), dtype=np.int32)
    segments[10:20, 10:20] = 1
    segments[10:12, 10:12] = 0  # 4 pixel notch in the corner
    chm = np.where(segments == 1, 1.0, 0.0)
    written, _ = save_and_read_segments(tmp_path, segments, chm)
    assert written.geometry.area.iloc[0] == pytest.approx(96 * RES**2)


def test_small_hole_is_filled_in_the_polygon_but_not_in_area_m2(tmp_path):
    segments = np.zeros((30, 30), dtype=np.int32)
    segments[10:20, 10:20] = 1
    segments[14:16, 14:16] = 0  # 4 pixel hole inside the crown
    chm = np.where(segments == 1, 1.0, 0.0)
    written, _ = save_and_read_segments(tmp_path, segments, chm)
    assert written.geometry.area.iloc[0] == pytest.approx(100 * RES**2)
    assert written.area_m2.iloc[0] == pytest.approx(96 * RES**2)
    assert written.mean_h.iloc[0] == pytest.approx(1.0)


def test_extent_keeps_only_the_seeds_inside_it(tmp_path):
    chm = np.zeros((100, 100))
    for col in (20, 50, 80):
        add_cone(chm, 50, col, 6, 1.0 + col / 100)
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    seeds = write_seeds(tmp_path / "seeds.shp", [(50, 20), (50, 50), (50, 80)])
    # covers columns 35 to 100, so the first seed is outside
    extent = str(tmp_path / "extent.shp")
    gpd.GeoDataFrame(
        geometry=[box(500003.5, 3999990.0, 500010.0, 4000000.0)], crs=CRS
    ).to_file(extent)
    segmentation.segment_canopies(
        chm_path, seeds, str(tmp_path), extent_shapefile=extent
    )
    written = gpd.read_file(tmp_path / "planted_segments.shp")
    assert sorted(written.plant) == ["plant_1", "plant_2"]
    assert sorted(written.max_h) == pytest.approx([1.5, 1.8])


def test_extent_with_no_seed_inside_returns_none(tmp_path):
    chm = np.zeros((80, 80))
    add_cone(chm, 40, 40, 6, 1.0)
    chm_path = write_chm(tmp_path / "planted_chm.tif", chm)
    extent = str(tmp_path / "extent.shp")
    gpd.GeoDataFrame(
        geometry=[box(500002.0, 3999994.0, 500006.0, 3999998.0)], crs=CRS
    ).to_file(extent)
    seeds = str(tmp_path / "seeds.shp")
    gpd.GeoDataFrame(
        {"plant": ["outside"]}, geometry=[Point(500050.0, 3999950.0)], crs=CRS
    ).to_file(seeds)
    result = segmentation.segment_canopies(
        chm_path, seeds, str(tmp_path), extent_shapefile=extent
    )
    assert result is None
