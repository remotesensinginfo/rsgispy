"""Tests for rsgispy.tools.checkdatasets."""

import importlib
import json
import sys
from unittest.mock import patch

import geopandas
import numpy
import rasterio
from rasterio.transform import from_origin
import shapely.geometry

import rsgispy.tools as rtools
from rsgispy.tools import checkdatasets

try:
    import h5py  # pylint: disable=import-error
except ImportError:
    h5py = None


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _write_test_tif(path, width=16, height=16, count=2, epsg=32630, origin=(500000.0, 5800000.0)):
    crs = f"EPSG:{epsg}" if epsg is not None else None
    data = numpy.arange(count * height * width, dtype="uint8").reshape(
        count, height, width
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="uint8",
        crs=crs,
        transform=from_origin(origin[0], origin[1], 10.0, 10.0),
    ) as dst:
        dst.write(data)


def test_check_gdal_image_file_valid_and_invalid(tmp_path):
    """Test valid, mismatched, empty, and corrupted KEA raster checks."""
    tif = tmp_path / "valid.tif"
    _write_test_tif(tif, count=2, epsg=32630)

    ok, err = checkdatasets.check_gdal_image_file(
        str(tif),
        check_bands=True,
        n_bands=2,
        chk_proj=True,
        epsg_code=32630,
        read_img=True,
        smpl_n_pxls=5,
        calc_chk_sum=True,
    )
    assert ok is True
    assert err == ""

    # Wrong band count
    ok, err = checkdatasets.check_gdal_image_file(str(tif), n_bands=3)
    assert ok is False
    assert "3 image bands" in err

    # Wrong EPSG
    ok, err = checkdatasets.check_gdal_image_file(
        str(tif), chk_proj=True, epsg_code=4326
    )
    assert ok is False
    assert "does not match" in err

    # Unprojected raster
    unproj = tmp_path / "unproj.tif"
    _write_test_tif(unproj, epsg=None)
    ok, err = checkdatasets.check_gdal_image_file(str(unproj), chk_proj=True)
    assert ok is False
    assert "None" in err

    # Zero-byte file caught even when max_file_size is None
    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"")
    ok, err = checkdatasets.check_gdal_image_file(str(empty))
    assert ok is False
    assert "zero" in err.lower()

    # Corrupted .kea file (tests bugfix unpacking (bool, str) from check_hdf5_file)
    bad_kea = tmp_path / "bad.kea"
    bad_kea.write_bytes(b"not-an-hdf5-file")
    ok, err = checkdatasets.check_gdal_image_file(str(bad_kea))
    assert ok is False
    assert err != ""


def test_check_gdal_vector_file(tmp_path):
    """Test valid, mismatched EPSG, and corrupted vector checks."""
    gpkg = tmp_path / "vec.gpkg"
    gdf = geopandas.GeoDataFrame(
        {"id": [1, 2], "geometry": [shapely.geometry.Point(0, 0), shapely.geometry.Point(1, 1)]},
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg, layer="pts", driver="GPKG")

    ok, err = checkdatasets.check_gdal_vector_file(
        str(gpkg), chk_proj=True, epsg_code=4326
    )
    assert ok is True
    assert err == ""

    ok, err = checkdatasets.check_gdal_vector_file(
        str(gpkg), chk_proj=True, epsg_code=32630
    )
    assert ok is False
    assert "does not match" in err

    corrupt = tmp_path / "corrupt.gpkg"
    corrupt.write_bytes(b"garbage")
    ok, _ = checkdatasets.check_gdal_vector_file(str(corrupt))
    assert ok is False


def test_check_hdf5_file(tmp_path):
    """Test valid and corrupted HDF5 validation."""
    if not checkdatasets.HAVE_H5PY:
        return

    h5_path = tmp_path / "valid.h5"
    with h5py.File(h5_path, "w") as hf:
        grp = hf.create_group("grp")
        grp.create_dataset("arr", data=numpy.zeros((4, 4), dtype="float32"))

    ok, err = checkdatasets.check_hdf5_file(str(h5_path))
    assert ok is True
    assert err == ""

    bad_h5 = tmp_path / "bad.h5"
    bad_h5.write_bytes(b"not-hdf5")
    ok, err = checkdatasets.check_hdf5_file(str(bad_h5))
    assert ok is False
    assert err != ""


def test_multiprocessing_runners_and_rm_err(tmp_path):
    """Test subprocess pool validation runners and error file removal."""
    good_tif = tmp_path / "good.tif"
    bad_tif = tmp_path / "bad.tif"
    _write_test_tif(good_tif, epsg=4326)
    bad_tif.write_bytes(b"bad-tif-content")

    assert checkdatasets.run_check_gdal_image_file(
        str(good_tif), chk_proj=True, epsg_code=4326, timeout=10
    )
    assert not checkdatasets.run_check_gdal_image_files(
        [str(good_tif), str(bad_tif)],
        chk_proj=True,
        epsg_code=4326,
        rm_err=True,
        print_file_names=True,
        timeout=10,
    )
    assert good_tif.exists()
    assert not bad_tif.exists()

    good_gpkg = tmp_path / "good.gpkg"
    geopandas.GeoDataFrame(
        {"id": [1], "geometry": [shapely.geometry.Point(0, 0)]},
        crs="EPSG:4326",
    ).to_file(good_gpkg, layer="p", driver="GPKG")
    assert checkdatasets.run_check_gdal_vector_file(
        str(good_gpkg), chk_proj=True, epsg_code=4326, timeout=10
    )
    assert checkdatasets.run_check_gdal_vector_files(
        [str(good_gpkg)], chk_proj=True, epsg_code=4326, timeout=10
    )

    if checkdatasets.HAVE_H5PY:
        good_h5 = tmp_path / "good.h5"
        with h5py.File(good_h5, "w") as hf:
            hf.create_dataset("d", data=numpy.ones(3))
        assert checkdatasets.run_check_hdf5_file(str(good_h5), timeout=10)
        assert checkdatasets.run_check_hdf5_files([str(good_h5)], timeout=10)


def test_cmp_to_ref_imgs(tmp_path):
    """Test comparing output rasters against reference rasters."""
    ref_dir = tmp_path / "ref"
    chk_dir = tmp_path / "chk"
    ref_dir.mkdir()
    chk_dir.mkdir()

    ref1 = ref_dir / "tile01.tif"
    ref2 = ref_dir / "tile02.tif"
    _write_test_tif(ref1, width=10, height=10, epsg=32630)
    _write_test_tif(ref2, width=10, height=10, epsg=32630)

    chk_ok = chk_dir / "tile01_ndvi.tif"
    chk_bad_size = chk_dir / "tile01_badsize.tif"
    chk_orphan = chk_dir / "other_scene.tif"
    _write_test_tif(chk_ok, width=10, height=10, epsg=32630)
    _write_test_tif(chk_bad_size, width=12, height=10, epsg=32630)
    _write_test_tif(chk_orphan, width=10, height=10, epsg=32630)

    report_json = tmp_path / "report.json"
    report = checkdatasets.cmp_to_ref_imgs(
        [str(ref1), str(ref2)],
        str(chk_dir),
        "tif",
        rm_errs=True,
        output_file=str(report_json),
    )

    assert str(chk_ok) in report["checked"]
    assert str(chk_bad_size) in report["errors"]
    assert not chk_bad_size.exists()
    assert chk_ok.exists()
    assert report["not_checked"] == [str(chk_orphan)]
    assert report["ref_no_match"] == [str(ref2)]
    assert json.loads(report_json.read_text())["errors"] == [str(chk_bad_size)]


def test_optional_h5py_missing(tmp_path):
    """Verify HDF5 functions are omitted when h5py is unavailable."""
    h5_only = {"check_hdf5_file", "run_check_hdf5_file", "run_check_hdf5_files"}
    always_avail = {
        "check_gdal_image_file",
        "run_check_gdal_image_file",
        "run_check_gdal_image_files",
        "check_gdal_vector_file",
        "run_check_gdal_vector_file",
        "run_check_gdal_vector_files",
        "cmp_to_ref_imgs",
    }

    tif = tmp_path / "valid.tif"
    _write_test_tif(tif, count=2, epsg=32630)

    orig_mod = checkdatasets
    try:
        with patch.dict(sys.modules, {"h5py": None}):
            sys.modules.pop("rsgispy.tools.checkdatasets", None)
            mod = importlib.import_module("rsgispy.tools.checkdatasets")

        assert not mod.HAVE_H5PY
        for fn_name in h5_only:
            assert not hasattr(mod, fn_name)
            assert fn_name not in mod.__all__
        for fn_name in always_avail:
            assert hasattr(mod, fn_name)
            assert fn_name in mod.__all__

        ok, err = mod.check_gdal_image_file(str(tif), chk_proj=True, epsg_code=32630)
        assert ok is True
        assert err == ""
    finally:
        sys.modules["rsgispy.tools.checkdatasets"] = orig_mod
        rtools.checkdatasets = orig_mod
