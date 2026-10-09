"""
Tests for lidar_preprocessing.preprocessing on small planted rasters and
point clouds whose correct answer is known by construction. The cropping
tests run PDAL, so the suite needs the hazelnut-biomass conda env.

Run from the repository root with: python -m pytest tests
"""

import geopandas as gpd
import laspy
import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely import wkt
from shapely.geometry import Polygon, box

from lidar_preprocessing import preprocessing

RES = 0.1
CRS = "EPSG:32616"
NODATA = -9999.0  # the value PDAL's writers.gdal tags DTM and DSM holes with
X0, Y0 = 500000.0, 4000000.0


def write_raster(path, array, top=Y0):
    """Write a planted DSM or DTM whose top edge is at northing `top`."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(X0, top, RES, RES),
        nodata=NODATA,
    ) as dst:
        dst.write(array.astype("float32"), 1)
    return str(path)


def read_chm(path):
    with rasterio.open(path) as src:
        return src.read(1), src.nodata


def test_chm_is_nodata_where_dtm_or_dsm_has_none(tmp_path):
    dsm = np.full((40, 40), 186.5)
    dtm = np.full((40, 40), 185.0)
    dtm[10:14, 10:14] = NODATA
    dsm[30:33, 30:33] = NODATA
    chm_path = str(tmp_path / "chm.tif")
    preprocessing.create_chm(
        write_raster(tmp_path / "dsm.tif", dsm),
        write_raster(tmp_path / "dtm.tif", dtm),
        chm_path,
    )
    chm, nodata = read_chm(chm_path)
    assert np.isnan(nodata)
    assert np.isnan(chm[10:14, 10:14]).all()
    assert np.isnan(chm[30:33, 30:33]).all()
    assert int(np.isnan(chm).sum()) == 16 + 9
    assert chm[np.isfinite(chm)] == pytest.approx(1.5)


def test_chm_keeps_nodata_when_dtm_is_aligned_to_dsm(tmp_path):
    dsm = np.full((40, 40), 186.5)
    # DTM grid is two rows taller than the DSM, starting two rows north
    dtm = np.full((42, 40), 185.0)
    dtm[12:16, 10:14] = NODATA  # rows 10 to 13 on the DSM grid
    chm_path = str(tmp_path / "chm.tif")
    preprocessing.create_chm(
        write_raster(tmp_path / "dsm.tif", dsm),
        write_raster(tmp_path / "dtm.tif", dtm, top=Y0 + 2 * RES),
        chm_path,
    )
    chm, _ = read_chm(chm_path)
    assert chm.shape == (40, 40)
    assert np.isnan(chm[10:14, 10:14]).all()
    assert np.nanmax(chm) == pytest.approx(1.5)
    assert np.nanmin(chm) == pytest.approx(1.5)


def write_las(path, crs=CRS):
    """Write flat ground on a 0.5 m grid over a 20 m square."""
    grid = np.arange(0.25, 20.0, 0.5)
    xs, ys = np.meshgrid(X0 + grid, Y0 + grid)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([X0, Y0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs is not None:
        header.add_crs(pyproj.CRS.from_user_input(crs))
    las = laspy.LasData(header)
    las.x = xs.ravel()
    las.y = ys.ravel()
    las.z = np.full(xs.size, 185.0)
    las.return_number = np.ones(xs.size, dtype=np.uint8)
    las.number_of_returns = np.ones(xs.size, dtype=np.uint8)
    las.write(path)
    return str(path)


# The hypotenuse x + y = 20.2 passes through no point of the 0.5 m grid.
TRIANGLE = Polygon(
    [(X0 + 2, Y0 + 2), (X0 + 18.2, Y0 + 2), (X0 + 2, Y0 + 18.2)]
)


def write_extent(path, geometries, crs=CRS):
    """Write geometries given in the test CRS, stored in `crs` (or none)."""
    if crs is None:
        gdf = gpd.GeoDataFrame(geometry=geometries)
    else:
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=CRS).to_crs(crs)
    gdf.to_file(path)
    return str(path)


def test_cloud_is_cropped_to_the_polygon_not_its_bounding_box(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    extent = write_extent(tmp_path / "extent.shp", [TRIANGLE])
    out = str(tmp_path / "classified.las")
    preprocessing.classify_ground(
        las_path,
        out,
        crop_polygon=preprocessing.get_crop_polygon_wkt(extent, las_path),
    )
    las = laspy.read(out)
    x, y = np.asarray(las.x), np.asarray(las.y)
    grid = np.arange(0.25, 20.0, 0.5)
    xs, ys = np.meshgrid(X0 + grid, Y0 + grid)
    in_triangle = (
        (xs > X0 + 2) & (ys > Y0 + 2) & ((xs - X0) + (ys - Y0) < 20.2)
    )
    assert len(x) == int(in_triangle.sum())
    assert ((x - X0) + (y - Y0) < 20.2).all()


def test_extent_in_another_crs_is_reprojected_to_the_las_crs(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    extent = write_extent(tmp_path / "extent.shp", [TRIANGLE], crs="EPSG:4326")
    polygon = wkt.loads(preprocessing.get_crop_polygon_wkt(extent, las_path))
    assert polygon.symmetric_difference(TRIANGLE).area < 1e-3


def test_crop_polygon_grows_by_the_margin(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    extent = write_extent(
        tmp_path / "extent.shp", [box(X0 + 5, Y0 + 5, X0 + 9, Y0 + 9)]
    )
    polygon = wkt.loads(
        preprocessing.get_crop_polygon_wkt(extent, las_path, margin_m=2.5)
    )
    assert polygon.bounds == pytest.approx(
        (X0 + 2.5, Y0 + 2.5, X0 + 11.5, Y0 + 11.5)
    )


def test_extent_with_several_features_crops_to_all_of_them(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    squares = [
        box(X0 + 1, Y0 + 1, X0 + 5, Y0 + 5),
        box(X0 + 12, Y0 + 12, X0 + 16, Y0 + 16),
    ]
    extent = write_extent(tmp_path / "extent.shp", squares)
    out = str(tmp_path / "classified.las")
    preprocessing.classify_ground(
        las_path,
        out,
        crop_polygon=preprocessing.get_crop_polygon_wkt(extent, las_path),
    )
    # each 4 m square holds 8 by 8 points of the 0.5 m grid
    assert len(laspy.read(out).x) == 2 * 64


def test_cropping_does_not_change_the_estimated_resolution(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    extent = write_extent(tmp_path / "extent.shp", [TRIANGLE])
    full = preprocessing.preprocess_lidar(las_path, str(tmp_path / "full"))
    cropped = preprocessing.preprocess_lidar(
        las_path, str(tmp_path / "cropped"), extent_shapefile=extent
    )
    with rasterio.open(full["chm"]) as a, rasterio.open(cropped["chm"]) as b:
        assert b.res == pytest.approx(a.res, rel=1e-3)


def test_extent_without_a_crs_raises(tmp_path):
    las_path = write_las(tmp_path / "cloud.las")
    extent = write_extent(tmp_path / "extent.shp", [TRIANGLE], crs=None)
    with pytest.raises(ValueError, match="Extent shapefile has no CRS"):
        preprocessing.get_crop_polygon_wkt(extent, las_path)


def test_las_without_a_crs_raises(tmp_path):
    las_path = write_las(tmp_path / "cloud.las", crs=None)
    extent = write_extent(tmp_path / "extent.shp", [TRIANGLE])
    with pytest.raises(ValueError, match="LAS file has no CRS"):
        preprocessing.get_crop_polygon_wkt(extent, las_path)
