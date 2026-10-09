"""
Planted cases for defects found in a review of the three stages. Each test
builds a small CHM whose correct answer is known, runs the module function,
and asserts the value it produces.

Run with: python -m pytest tests
"""

import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomass_estimation import biomass as bio  # noqa: E402
from canopy_segmentation import segmentation as seg  # noqa: E402
from lidar_preprocessing import preprocessing as pre  # noqa: E402

RES = 0.1
CRS = "EPSG:32616"


def write_tif(path, arr, nodata=None):
    tr = from_origin(1000.0, 2000.0, RES, RES)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=tr,
        nodata=nodata,
    ) as d:
        d.write(arr.astype("float32"), 1)
    return tr


def disk(arr, r, c, radius_px, h):
    rr, cc = np.ogrid[: arr.shape[0], : arr.shape[1]]
    m = (rr - r) ** 2 + (cc - c) ** 2 <= radius_px**2
    arr[m] = np.maximum(arr[m], h)


def seeds_file(path, tr, rowcols):
    pts = [Point(*rasterio.transform.xy(tr, r, c)) for r, c in rowcols]
    gpd.GeoDataFrame(
        {"plot": [f"P{i}" for i in range(len(pts))]}, geometry=pts, crs=CRS
    ).to_file(path)
    return path


def test_chm_hole_is_nodata_not_ten_thousand_meters(tmp_path):
    dsm = np.full((60, 60), 185.0)
    dtm = np.full((60, 60), 184.5)
    dsm[25:35, 25:35] = 186.2
    dtm[28:32, 28:32] = -9999.0  # IDW found no ground under the crown
    p = {k: str(tmp_path / f"{k}.tif") for k in ("dsm", "dtm", "chm")}
    write_tif(p["dsm"], dsm, -9999)
    write_tif(p["dtm"], dtm, -9999)
    pre.create_chm(p["dsm"], p["dtm"], p["chm"])
    with rasterio.open(p["chm"]) as s:
        a = s.read(1)
        assert np.isnan(s.nodata)
    assert np.nanmax(a) == pytest.approx(1.7, abs=1e-3)
    assert np.isnan(a[28:32, 28:32]).all()


def test_two_part_segment_keeps_both_parts(tmp_path):
    chm = np.zeros((80, 80))
    disk(chm, 40, 30, 8, 1.2)
    disk(chm, 40, 52, 8, 1.0)
    chm_path = str(tmp_path / "chm.tif")
    tr = write_tif(chm_path, chm)
    segments = np.where(chm > 0, 1, 0).astype(np.int32)
    prof = dict(
        transform=tr, crs=rasterio.crs.CRS.from_string(CRS), chm_path=chm_path
    )
    seg.save_segments(segments, chm, prof, str(tmp_path), res_m_per_px=RES)
    g = gpd.read_file(tmp_path / "chm_segments.shp")
    labeled = float((segments == 1).sum() * RES**2)
    assert g.geometry.area.sum() == pytest.approx(labeled, rel=1e-6)
    assert float(g.area_m2.iloc[0]) == pytest.approx(labeled, rel=1e-6)
    vol = bio.calculate_polygon_volumes(g, chm_path)
    assert float(vol.volume_m3.iloc[0]) == pytest.approx(
        float(chm[segments == 1].sum() * RES**2), rel=1e-6
    )


def test_colliding_seeds_each_keep_a_marker(tmp_path):
    chm = np.zeros((80, 80))
    disk(chm, 40, 40, 6, 1.5)
    chm_path = str(tmp_path / "chm.tif")
    tr = write_tif(chm_path, chm)
    seeds = seeds_file(str(tmp_path / "s.shp"), tr, [(40, 34), (40, 46)])
    chm_l, prof, _, _ = seg.load_chm(chm_path)
    _, _, markers, refined = seg.refine_tree_tops(chm_l, prof, seeds, 1.75)
    labels = np.unique(markers[markers > 0])
    assert len(refined) == 2
    assert len(labels) == 2
    assert sorted(refined.tree_id) == sorted(labels.tolist())


def test_snap_window_is_a_disk(tmp_path):
    chm = np.zeros((80, 80))
    disk(chm, 40, 40, 3, 0.4)  # the seeded bush
    disk(chm, 56, 56, 3, 1.6)  # taller neighbor 2.26 m away on the diagonal
    chm_path = str(tmp_path / "chm.tif")
    tr = write_tif(chm_path, chm)
    seeds = seeds_file(str(tmp_path / "s.shp"), tr, [(40, 40)])
    chm_l, prof, _, _ = seg.load_chm(chm_path)
    _, _, _, refined = seg.refine_tree_tops(chm_l, prof, seeds, 1.75)
    x0, y0 = rasterio.transform.xy(tr, 40, 40)
    moved = refined.geometry.iloc[0].distance(Point(x0, y0))
    assert moved <= 1.75 + RES
    assert refined.height.iloc[0] == pytest.approx(0.4)


def test_max_radius_stops_a_segment_absorbing_a_neighbor(tmp_path):
    chm = np.zeros((80, 80))
    disk(chm, 40, 30, 7, 1.0)
    disk(chm, 40, 47, 7, 1.0)
    chm[40, 37:41] = 0.3  # grass bridge above the 0.1 m mask threshold
    chm_path = str(tmp_path / "chm.tif")
    tr = write_tif(chm_path, chm)
    seeds = seeds_file(str(tmp_path / "s.shp"), tr, [(40, 30)])
    own = np.pi * (7 * RES) ** 2
    seg.segment_canopies(chm_path, seeds, output_dir=str(tmp_path / "a"))
    free = gpd.read_file(tmp_path / "a" / "chm_segments.shp")
    assert float(free.area_m2.iloc[0]) > 1.8 * own  # the defect, unchanged
    seg.segment_canopies(
        chm_path, seeds, output_dir=str(tmp_path / "b"), max_radius=1.0
    )
    capped = gpd.read_file(tmp_path / "b" / "chm_segments.shp")
    assert float(capped.area_m2.iloc[0]) == pytest.approx(own, rel=0.15)


def test_extent_with_no_seed_inside_returns_none(tmp_path):
    chm = np.zeros((80, 80))
    disk(chm, 40, 40, 7, 1.0)
    chm_path = str(tmp_path / "chm.tif")
    write_tif(chm_path, chm)
    ext = str(tmp_path / "ext.shp")
    gpd.GeoDataFrame(
        geometry=[Point(1003, 1996).buffer(2.0)], crs=CRS
    ).to_file(ext)
    far = str(tmp_path / "far.shp")
    gpd.GeoDataFrame(
        {"plot": ["A"]}, geometry=[Point(5000, 5000)], crs=CRS
    ).to_file(far)
    assert (
        seg.segment_canopies(
            chm_path, far, str(tmp_path), extent_shapefile=ext
        )
        is None
    )
