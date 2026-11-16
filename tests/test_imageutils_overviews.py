"""
Tests for tiled multi-core raster overview pyramid generation in
``rsgispy.imageutils.stats.build_img_pyramids_multi_core``.
"""

import os

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, from_origin

import rsgispy
import rsgispy.imageutils.stats as iu_stats


def _write_geotiff(
    path,
    data,
    *,
    crs="EPSG:32630",
    transform=None,
    nodata=None,
    colormap=None,
    tags=None,
    band_tags=None,
    descriptions=None,
):
    """Write a test GeoTIFF with optional georeferencing, colormap, and metadata."""
    # pylint: disable=too-many-arguments
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "tiled": True,
        "blockxsize": 32,
        "blockysize": 32,
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)
        if colormap is not None:
            dst.write_colormap(1, colormap)
        if tags:
            dst.update_tags(**tags)
        if band_tags:
            for b_idx, b_tags in enumerate(band_tags, start=1):
                if b_tags:
                    dst.update_tags(b_idx, **b_tags)
        if descriptions:
            for b_idx, desc in enumerate(descriptions, start=1):
                if desc is not None:
                    dst.set_band_description(b_idx, desc)


def test_build_img_pyramids_multi_core_external_matches_gdal(tmp_path):
    """External .ovr generated across multiple tiles/cores matches GDAL overviews."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 100.0, size=(3, 160, 128)).astype(np.float32)
    arr[:, :16, :16] = -999.0

    src_tif = tmp_path / "src.tif"
    ref_tif = tmp_path / "ref.tif"
    xfm = from_origin(500000.0, 5200000.0, 10.0, 10.0)
    _write_geotiff(src_tif, arr, transform=xfm, nodata=-999.0)
    _write_geotiff(ref_tif, arr, transform=xfm, nodata=-999.0)

    with rasterio.open(str(ref_tif), "r+") as dst:
        dst.build_overviews([2, 4, 8], Resampling.average)

    orig_bytes = src_tif.read_bytes()
    scratch_dir = tmp_path / "ovr_scratch"

    iu_stats.build_img_pyramids_multi_core(
        src_tif,
        resampling_method=Resampling.average,
        levels=[2, 4, 8],
        tile_size=19,
        n_cores=2,
        external=True,
        tmp_dir=scratch_dir,
        compress="LZW",
        ovr_block_size=64,
        cascade=True,
        use_pbar=False,
    )

    ovr_path = tmp_path / "src.tif.ovr"
    assert ovr_path.exists()
    assert src_tif.read_bytes() == orig_bytes
    assert list(scratch_dir.iterdir()) == []

    with rasterio.open(str(src_tif), "r") as dst, rasterio.open(
        str(ref_tif), "r"
    ) as rdst:
        assert dst.overviews(1) == [2, 4, 8]
        assert dst.overviews(2) == [2, 4, 8]
        assert dst.overviews(3) == [2, 4, 8]
        assert rdst.overviews(1) == [2, 4, 8]

    for ovr_idx in range(3):
        with rasterio.open(str(src_tif), "r", OVERVIEW_LEVEL=ovr_idx) as ovr_ds:
            with rasterio.open(
                str(ref_tif), "r", OVERVIEW_LEVEL=ovr_idx
            ) as ref_ovr:
                assert ovr_ds.shape == ref_ovr.shape
                assert ovr_ds.nodata == -999.0
                np.testing.assert_allclose(
                    ovr_ds.read(), ref_ovr.read(), rtol=1e-5, atol=1e-5
                )

    # Also verify non-divisible odd image dimensions (127x129) produce valid shapes.
    odd_arr = rng.uniform(0.0, 100.0, size=(2, 129, 127)).astype(np.float32)
    odd_tif = tmp_path / "odd.tif"
    odd_ref = tmp_path / "odd_ref.tif"
    _write_geotiff(odd_tif, odd_arr, transform=xfm, nodata=-999.0)
    _write_geotiff(odd_ref, odd_arr, transform=xfm, nodata=-999.0)
    with rasterio.open(str(odd_ref), "r+") as dst:
        dst.build_overviews([2, 4, 8], Resampling.average)

    iu_stats.build_img_pyramids_multi_core(
        odd_tif,
        resampling_method=Resampling.average,
        levels=[2, 4, 8],
        tile_size=19,
        n_cores=2,
        external=True,
        ovr_block_size=64,
        cascade=True,
        use_pbar=False,
    )
    expected_shapes = [(65, 64), (33, 32), (17, 16)]
    for ovr_idx, exp_shape in enumerate(expected_shapes):
        with rasterio.open(str(odd_tif), "r", OVERVIEW_LEVEL=ovr_idx) as ovr_ds:
            with rasterio.open(
                str(odd_ref), "r", OVERVIEW_LEVEL=ovr_idx
            ) as ref_ovr:
                assert ovr_ds.shape == exp_shape
                assert ref_ovr.shape == exp_shape
                got_odd = ovr_ds.read()
                exp_odd = ref_ovr.read()
                np.testing.assert_allclose(
                    got_odd[:, 1:-1, :],
                    exp_odd[:, 1:-1, :],
                    rtol=1e-5,
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    got_odd, exp_odd, rtol=1e-2, atol=0.35
                )


def test_build_img_pyramids_multi_core_nan_nodata_and_non_cascade(tmp_path):
    """Non-cascading overviews handle NaN nodata and un-georeferenced rasters."""
    yy, xx = np.mgrid[0:101, 0:97]
    arr = np.stack(
        [
            (yy * 0.5 + xx * 0.25).astype(np.float32),
            (yy * 1.2 - xx * 0.1).astype(np.float32),
        ],
        axis=0,
    )
    arr[:, 10:25, 20:35] = np.nan

    src_tif = tmp_path / "ungeo_nan.tif"
    ref_tifs = {}
    _write_geotiff(
        src_tif,
        arr,
        crs=None,
        transform=Affine.identity(),
        nodata=float("nan"),
    )
    for lvl in (4, 6):
        ref_p = tmp_path / f"ungeo_ref_{lvl}.tif"
        _write_geotiff(
            ref_p,
            arr,
            crs=None,
            transform=Affine.identity(),
            nodata=float("nan"),
        )
        with rasterio.open(str(ref_p), "r+") as dst:
            dst.build_overviews([lvl], Resampling.bilinear)
        ref_tifs[lvl] = ref_p

    iu_stats.build_img_pyramids_multi_core(
        src_tif,
        resampling_method=Resampling.bilinear,
        levels=[4, 6],
        tile_size=16,
        n_cores=2,
        external=True,
        ovr_block_size=64,
        cascade=False,
        use_pbar=False,
    )

    assert (tmp_path / "ungeo_nan.tif.ovr").exists()
    for ovr_idx, lvl in enumerate((4, 6)):
        with rasterio.open(str(src_tif), "r", OVERVIEW_LEVEL=ovr_idx) as ovr_ds:
            with rasterio.open(
                str(ref_tifs[lvl]), "r", OVERVIEW_LEVEL=0
            ) as ref_ovr:
                got = ovr_ds.read()
                exp = ref_ovr.read()
                assert np.array_equal(np.isnan(got), np.isnan(exp))
                valid = ~np.isnan(exp)
                np.testing.assert_allclose(
                    got[valid], exp[valid], rtol=1e-5, atol=1e-5
                )


def test_build_img_pyramids_multi_core_aliases_auto_levels_and_stale_ovr(tmp_path):
    """Test string/rsgispy resampling, auto levels, custom path, and stale .ovr isolation."""
    arr_200 = np.arange(200 * 200, dtype=np.float32).reshape(1, 200, 200)
    img_200 = tmp_path / "auto_200.tif"
    _write_geotiff(img_200, arr_200, transform=from_origin(0.0, 2000.0, 10.0, 10.0))

    iu_stats.build_img_pyramids_multi_core(
        img_200,
        resampling_method="median",
        levels=None,
        tile_size=25,
        n_cores=2,
        external=True,
        ovr_block_size=64,
        use_pbar=False,
    )
    with rasterio.open(str(img_200), "r") as dst:
        assert dst.overviews(1) == [4]

    img_small = tmp_path / "small_64.tif"
    _write_geotiff(
        img_small,
        np.ones((1, 64, 64), dtype=np.float32),
        transform=from_origin(0.0, 640.0, 10.0, 10.0),
    )
    iu_stats.build_img_pyramids_multi_core(
        img_small, levels=None, n_cores=2, use_pbar=False
    )
    assert not (tmp_path / "small_64.tif.ovr").exists()

    # Stale .ovr filled with 999.0 must NOT contaminate rebuilt overviews.
    arr_const = np.full((1, 128, 128), 10.0, dtype=np.float32)
    stale_src = tmp_path / "stale.tif"
    _write_geotiff(
        stale_src, arr_const, transform=from_origin(0.0, 1280.0, 10.0, 10.0)
    )
    stale_ovr = tmp_path / "stale.tif.ovr"
    _write_geotiff(
        stale_ovr,
        np.full((1, 32, 32), 999.0, dtype=np.float32),
        transform=from_origin(0.0, 1280.0, 40.0, 40.0),
    )

    iu_stats.build_img_pyramids_multi_core(
        stale_src,
        resampling_method=rsgispy.INTERP_AVERAGE,
        levels=[4, 8],
        tile_size=16,
        n_cores=2,
        external=True,
        ovr_block_size=64,
        use_pbar=False,
    )
    with rasterio.open(str(stale_src), "r", OVERVIEW_LEVEL=0) as ovr_ds:
        np.testing.assert_allclose(ovr_ds.read(), 10.0)

    # Custom external output path.
    custom_ovr = tmp_path / "sub" / "custom_pyramids.ovr"
    iu_stats.build_img_pyramids_multi_core(
        stale_src,
        resampling_method="nearest_neighbour",
        levels=[4],
        tile_size=16,
        n_cores=1,
        external=True,
        output_img=custom_ovr,
        ovr_block_size=64,
        use_pbar=False,
    )
    assert custom_ovr.exists()
    with rasterio.open(str(custom_ovr), "r") as c_ds:
        assert c_ds.shape == (32, 32)
        np.testing.assert_allclose(c_ds.read(), 10.0)


def test_build_img_pyramids_multi_core_internal_overviews_and_metadata(tmp_path):
    """Internal overviews preserve colormaps, tags, descriptions, and remove stale .ovr."""
    arr = np.zeros((1, 96, 96), dtype=np.uint8)
    arr[0, :48, :48] = 1
    arr[0, 48:, 48:] = 2

    cmap = {
        0: (0, 0, 0, 255),
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
    }
    in_tif = tmp_path / "thematic.tif"
    _write_geotiff(
        in_tif,
        arr,
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        nodata=0,
        colormap=cmap,
        tags={"SENSOR": "TestSensor"},
        band_tags=[{"CLASS_SCHEME": "Basic"}],
        descriptions=["LandCover"],
    )

    out_tif = tmp_path / "thematic_internal.tif"
    iu_stats.build_img_pyramids_multi_core(
        in_tif,
        resampling_method="mode",
        levels=[2, 4],
        tile_size=24,
        n_cores=2,
        external=False,
        output_img=out_tif,
        ovr_block_size=64,
        use_pbar=False,
    )
    assert out_tif.exists()
    assert not (tmp_path / "thematic_internal.tif.ovr").exists()
    with rasterio.open(str(out_tif), "r") as dst:
        assert dst.overviews(1) == [2, 4]
        assert dst.colorinterp == (ColorInterp.palette,)
        out_cmap = dst.colormap(1)
        assert out_cmap[1] == (255, 0, 0, 255)
        assert out_cmap[2] == (0, 255, 0, 255)
        assert dst.tags()["SENSOR"] == "TestSensor"
        assert dst.tags(1)["CLASS_SCHEME"] == "Basic"
        assert dst.descriptions == ("LandCover",)
        assert dst.nodata == 0
        np.testing.assert_array_equal(dst.read(), arr)

    # Atomic in-place update should also clean up any stale .ovr sidecar.
    stale_ovr = tmp_path / "thematic.tif.ovr"
    _write_geotiff(
        stale_ovr,
        np.full((1, 48, 48), 9, dtype=np.uint8),
        transform=from_origin(500000.0, 5200000.0, 20.0, 20.0),
    )
    assert stale_ovr.exists()

    iu_stats.build_img_pyramids_multi_core(
        in_tif,
        resampling_method=Resampling.mode,
        levels=[2, 4],
        tile_size=24,
        n_cores=2,
        external=False,
        output_img=None,
        ovr_block_size=64,
        use_pbar=False,
    )
    assert not os.path.exists(stale_ovr)
    with rasterio.open(str(in_tif), "r") as dst:
        assert dst.overviews(1) == [2, 4]
        assert dst.tags()["SENSOR"] == "TestSensor"
        assert dst.descriptions == ("LandCover",)
        np.testing.assert_array_equal(dst.read(), arr)


def test_build_img_pyramids_multi_core_validation_errors(tmp_path):
    """Invalid arguments raise RSGISPyException."""
    missing = tmp_path / "does_not_exist.tif"
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(missing, use_pbar=False)

    valid_tif = tmp_path / "valid.tif"
    _write_geotiff(valid_tif, np.zeros((1, 64, 64), dtype=np.float32))

    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, n_cores=0, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, tile_size=0, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, ovr_block_size=16, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, ovr_block_size=80, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, ovr_block_size=8192, use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, levels=[], use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, levels=[1], use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, levels="4", use_pbar=False
        )
    with pytest.raises(rsgispy.RSGISPyException):
        iu_stats.build_img_pyramids_multi_core(
            valid_tif, resampling_method="not_a_method", use_pbar=False
        )


def test_pop_stats_multi_core_pyramids(tmp_path):
    """Test pop_thmt_img_stats and pop_img_stats with use_multi_core_pyramids=True."""
    xfm = from_origin(500000.0, 5200000.0, 10.0, 10.0)

    # 1. Thematic image with multi-core nearest-neighbour overviews.
    thmt_arr = np.zeros((1, 200, 200), dtype=np.uint8)
    thmt_arr[0, 10:100, 10:100] = 1
    thmt_arr[0, 100:190, 100:190] = 2
    thmt_tif = tmp_path / "thmt_mc.tif"
    _write_geotiff(thmt_tif, thmt_arr, transform=xfm)

    iu_stats.pop_thmt_img_stats(
        str(thmt_tif),
        add_clr_tab=True,
        calc_pyramids=True,
        ignore_zero=True,
        use_pbar=False,
        use_multi_core_pyramids=True,
        n_cores=2,
        tile_size=16,
    )
    assert (tmp_path / "thmt_mc.tif.ovr").exists()
    with rasterio.open(str(thmt_tif), "r") as dst:
        tags = dst.tags(1)
        assert tags["LAYER_TYPE"] == "thematic"
        assert tags["STATISTICS_MINIMUM"] == "1"
        assert tags["STATISTICS_MAXIMUM"] == "2"
        assert dst.overviews(1) == [4]
    with rasterio.open(str(thmt_tif), "r", OVERVIEW_LEVEL=0) as ovr_ds:
        assert ovr_ds.shape == (50, 50)
        assert set(np.unique(ovr_ds.read())).issubset({0, 1, 2})

    # 2. Continuous image with multi-core average overviews vs single-core.
    rng = np.random.default_rng(7)
    cont_arr = rng.uniform(0.0, 50.0, size=(2, 200, 200)).astype(np.float32)
    cont_arr[:, :12, :12] = -999.0

    cont_mc = tmp_path / "cont_mc.tif"
    cont_ref = tmp_path / "cont_ref.tif"
    _write_geotiff(cont_mc, cont_arr, transform=xfm)
    _write_geotiff(cont_ref, cont_arr, transform=xfm)

    iu_stats.pop_img_stats(
        str(cont_ref),
        no_data_val=-999.0,
        calc_pyramids=True,
        use_pbar=False,
    )
    iu_stats.pop_img_stats(
        str(cont_mc),
        no_data_val=-999.0,
        calc_pyramids=True,
        use_pbar=False,
        use_multi_core_pyramids=True,
        n_cores=2,
        tile_size=16,
    )
    assert (tmp_path / "cont_mc.tif.ovr").exists()
    with rasterio.open(str(cont_mc), "r") as dst, rasterio.open(
        str(cont_ref), "r"
    ) as rdst:
        assert dst.overviews(1) == [4]
        assert dst.tags(1)["STATISTICS_MINIMUM"] == rdst.tags(1)["STATISTICS_MINIMUM"]
        assert dst.tags(1)["STATISTICS_MAXIMUM"] == rdst.tags(1)["STATISTICS_MAXIMUM"]
    with rasterio.open(str(cont_mc), "r", OVERVIEW_LEVEL=0) as ovr_mc, rasterio.open(
        str(cont_ref), "r", OVERVIEW_LEVEL=0
    ) as ovr_ref:
        np.testing.assert_allclose(
            ovr_mc.read(), ovr_ref.read(), rtol=1e-5, atol=1e-5
        )

    # 3. calc_pyramids=False skips overview generation even if use_multi_core_pyramids=True.
    no_pyr = tmp_path / "no_pyr.tif"
    _write_geotiff(no_pyr, cont_arr, transform=xfm)
    iu_stats.pop_img_stats(
        str(no_pyr),
        no_data_val=-999.0,
        calc_pyramids=False,
        use_pbar=False,
        use_multi_core_pyramids=True,
        n_cores=2,
        tile_size=16,
    )
    assert not (tmp_path / "no_pyr.tif.ovr").exists()
    with rasterio.open(str(no_pyr), "r") as dst:
        assert dst.overviews(1) == []
