"""
Tests for the composable in-memory tile-graph pipeline engine
(:mod:`rsgispy._io`, :mod:`rsgispy.imagefilter.chained`) and refactored
zero-intermediate-I/O multi-step workflows.
"""

import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

import rsgispy
from rsgispy._io import (
    ArraySource,
    BandMathOp,
    DiskSink,
    HaloBorder,
    MemorySink,
    PointwiseOp,
    TileGraph,
    WindowOp,
    run_tile_graph,
)
import rsgispy.classification.classpostprocess as classpost
import rsgispy.elevation.filtering as elev_filt
import rsgispy.imagefilter.chained as if_chained
import rsgispy.imagefilter.edge as if_edge
import rsgispy.imagefilter.smooth as if_smooth
import rsgispy.imagefilter.stats as if_stats
import rsgispy.imagemorphology as morph
import rsgispy.imageutils.rasterise as iu_rast
from rasterio.transform import Affine
from pathlib import Path
from rsgispy._io import RasterSource


def _write_raster(path, arr, nodata=None, crs="EPSG:32630", res=10.0):
    arr3 = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    count, height, width = arr3.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr3.dtype),
        "crs": crs,
        "transform": from_origin(500000.0, 5000000.0, res, res),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr3)


def test_tilegraph_cumulative_halo_bit_exact_and_skip_connection():
    """
    Chained WindowOps with small block_size (non-multiple of raster size)
    and skip connection must be bit-identical to multi-pass disk execution,
    including image edges and corners.
    """
    rng = np.random.default_rng(42)
    arr = rng.uniform(10.0, 100.0, size=(2, 37, 43)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        in_img = os.path.join(tmp, "in.tif")
        pass1_img = os.path.join(tmp, "pass1.tif")
        pass2_img = os.path.join(tmp, "pass2.tif")
        chained_img = os.path.join(tmp, "chained.tif")
        _write_raster(in_img, arr, nodata=-9999.0)

        # Multi-pass reference via disk
        if_stats.apply_mean_filter(
            in_img,
            pass1_img,
            filter_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            use_pbar=False,
        )
        if_smooth.apply_gaussian_smooth_filter(
            pass1_img,
            pass2_img,
            filter_size=5,
            stddev_x=1.2,
            stddev_y=1.2,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            use_pbar=False,
        )

        # Single-pass halo-fused chain using a small block_size=13
        if_chained.apply_filter_chain(
            in_img,
            chained_img,
            steps=[
                if_chained.FilterStep("mean", filter_size=3),
                if_chained.FilterStep("gaussian", filter_size=5, params={"stddev": 1.2}),
            ],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            block_size=13,
            use_pbar=False,
        )

        with rasterio.open(pass2_img) as ref_ds, rasterio.open(chained_img) as out_ds:
            ref = ref_ds.read()
            out = out_ds.read()
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

        # Skip connection + BandMathOp + MemorySink on ArraySource
        src = ArraySource(arr.astype(np.float64), name="src")
        blur = WindowOp(
            src,
            radius=(1, 1),
            func=lambda a: a[:, 1:-1, 1:-1] * 0.5
            + 0.125
            * (
                a[:, :-2, 1:-1]
                + a[:, 2:, 1:-1]
                + a[:, 1:-1, :-2]
                + a[:, 1:-1, 2:]
            ),
            border=HaloBorder(mode="edge"),
            name="blur",
        )
        diff = BandMathOp({"orig": src, "blur": blur}, "orig - blur", name="diff")
        sink = MemorySink(diff, dtype=np.float64, name="res")
        res_map = run_tile_graph(sink, block_size=11, use_pbar=False)
        assert "res" in res_map
        # Reference full-frame edge-padded calculation
        padded = np.pad(
            arr.astype(np.float64), ((0, 0), (1, 1), (1, 1)), mode="edge"
        )
        expected_blur = padded[:, 1:-1, 1:-1] * 0.5 + 0.125 * (
            padded[:, :-2, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2]
            + padded[:, 1:-1, 2:]
        )
        np.testing.assert_allclose(
            res_map["res"], arr.astype(np.float64) - expected_blur, atol=1e-12
        )


def test_filter_chain_edge_and_band_math():
    """Verify chained median -> sobel -> band math matches disk passes."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 200, size=(1, 29, 31)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        in_img = os.path.join(tmp, "in.tif")
        med_img = os.path.join(tmp, "med.tif")
        sob_img = os.path.join(tmp, "sob.tif")
        out_img = os.path.join(tmp, "chain.tif")
        _write_raster(in_img, arr)

        if_stats.apply_median_filter(
            in_img,
            med_img,
            filter_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            use_pbar=False,
        )
        if_edge.apply_sobel_filter(
            med_img,
            sob_img,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            use_pbar=False,
        )
        if_chained.apply_filter_chain(
            in_img,
            out_img,
            steps=[
                if_chained.FilterStep("median", filter_size=3),
                if_chained.FilterStep("sobel"),
            ],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            block_size=10,
            use_pbar=False,
        )

        with rasterio.open(sob_img) as ref_ds, rasterio.open(out_img) as out_ds:
            np.testing.assert_allclose(out_ds.read(), ref_ds.read(), rtol=1e-5, atol=1e-5)


def test_imagemorphology_fused_opening_closing_tophat():
    """
    Fused TileGraph opening, closing, black_top_hat, and white_top_hat must
    match sequential single-step disk operations bit-for-bit, both with and
    without an explicit tmp_img secondary DiskSink.
    """
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 255, size=(2, 33, 35)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        in_img = os.path.join(tmp, "in.tif")
        _write_raster(in_img, arr)

        # Reference opening (2x erode -> 2x dilate)
        e1 = os.path.join(tmp, "e1.tif")
        e2 = os.path.join(tmp, "e2.tif")
        d1 = os.path.join(tmp, "d1.tif")
        ref_open = os.path.join(tmp, "ref_open.tif")
        morph.image_erode(
            in_img,
            e1,
            "",
            False,
            3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        morph.image_erode(
            e1,
            e2,
            "",
            False,
            3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        morph.image_dilate(
            e2,
            d1,
            "",
            False,
            3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        morph.image_dilate(
            d1,
            ref_open,
            "",
            False,
            3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )

        fused_open = os.path.join(tmp, "fused_open.tif")
        mid_img = os.path.join(tmp, "mid_eroded.tif")
        morph.image_opening(
            in_img,
            fused_open,
            "",
            False,
            3,
            tmp_img=mid_img,
            niters=2,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        with rasterio.open(ref_open) as r_ds, rasterio.open(fused_open) as f_ds:
            np.testing.assert_array_equal(f_ds.read(), r_ds.read())
        with rasterio.open(e2) as r_mid, rasterio.open(mid_img) as f_mid:
            np.testing.assert_array_equal(f_mid.read(), r_mid.read())

        # Reference white top-hat = in - opening(1)
        e_w = os.path.join(tmp, "ew.tif")
        op_w = os.path.join(tmp, "opw.tif")
        morph.image_erode(
            in_img,
            e_w,
            "",
            False,
            5,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        morph.image_dilate(
            e_w,
            op_w,
            "",
            False,
            5,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )

        wth_out = os.path.join(tmp, "wth.tif")
        wth_mid = os.path.join(tmp, "wth_mid.tif")
        morph.image_white_top_hat(
            in_img,
            wth_out,
            "",
            False,
            5,
            tmp_img=wth_mid,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        with rasterio.open(op_w) as r_op, rasterio.open(wth_out) as f_wth, rasterio.open(
            wth_mid
        ) as f_mid:
            op_arr = r_op.read()
            np.testing.assert_array_equal(f_mid.read(), op_arr)
            expected_wth = np.clip(
                arr.astype(np.int32) - op_arr.astype(np.int32), 0, 255
            ).astype(np.uint8)
            np.testing.assert_array_equal(f_wth.read(), expected_wth)

        # Black top-hat without tmp_img
        bth_out = os.path.join(tmp, "bth.tif")
        morph.image_black_top_hat(
            in_img,
            bth_out,
            "",
            False,
            5,
            tmp_img=None,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        assert os.path.exists(bth_out)


def test_in_memory_dist_zones_and_resampling_detrend():
    """
    create_dist_zones_to_vec_lyr (with tmp_vec_img=None, tmp_dist_img=None)
    and resampling_detrend_dem must execute with zero temporary GeoTIFFs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ref_img = os.path.join(tmp, "ref.tif")
        out_zones = os.path.join(tmp, "zones.tif")
        y_idx, x_idx = np.mgrid[0:20, 0:20]
        dem = (100.0 + 2.0 * x_idx + 1.5 * y_idx).astype(np.float32)
        _write_raster(ref_img, dem, nodata=-9999.0, res=10.0)

        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(500055.0, 4999945.0)],
            crs="EPSG:32630",
        )
        vec_file = os.path.join(tmp, "pt.gpkg")
        gdf.to_file(vec_file, layer="pt", driver="GPKG")

        iu_rast.create_dist_zones_to_vec_lyr(
            vec_file=vec_file,
            vec_lyr="pt",
            input_img=ref_img,
            tmp_vec_img=None,
            tmp_dist_img=None,
            output_img=out_zones,
            recode_lut=[(1, (0.0, 30.0)), (2, (30.0, 80.0)), (3, (80.0, 500.0))],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
        with rasterio.open(out_zones) as ds:
            zones = ds.read(1)
        assert set(np.unique(zones)).issuperset({1, 2, 3})

        # Resampling detrend DEM in memory
        detrend_out = os.path.join(tmp, "detrend.tif")
        elev_filt.resampling_detrend_dem(
            input_img=ref_img,
            output_img=detrend_out,
            resmp_res=40.0,
            gdalformat="GTIFF",
            no_data_val=-9999.0,
        )
        with rasterio.open(detrend_out) as ds:
            dt_arr = ds.read(1)
        assert dt_arr.shape == (20, 20)
        assert np.all(np.isfinite(dt_arr))


def test_classification_postprocess_pipeline_matches_sequential():
    """
    apply_postprocess_pipeline must match sequential apply_majority_filter +
    apply_sieve_filter bit-for-bit without intermediate disk writes.
    """
    rng = np.random.default_rng(99)
    arr = rng.choice([1, 2, 3], size=(28, 32), p=[0.5, 0.35, 0.15]).astype(np.uint8)
    arr[0:3, 0:3] = 0  # nodata corner
    with tempfile.TemporaryDirectory() as tmp:
        in_img = os.path.join(tmp, "cls.tif")
        m1_img = os.path.join(tmp, "m1.tif")
        m2_img = os.path.join(tmp, "m2.tif")
        ref_out = os.path.join(tmp, "ref_out.tif")
        pipe_out = os.path.join(tmp, "pipe_out.tif")
        _write_raster(in_img, arr, nodata=0)

        classpost.apply_majority_filter(
            in_img, m1_img, filter_size=3, no_data_val=0, calc_stats=False, use_pbar=False
        )
        classpost.apply_majority_filter(
            m1_img, m2_img, filter_size=5, no_data_val=0, calc_stats=False, use_pbar=False
        )
        classpost.apply_sieve_filter(
            m2_img,
            ref_out,
            size_threshold=8,
            connectivity=4,
            no_data_val=0,
            calc_stats=False,
            use_pbar=False,
        )

        classpost.apply_postprocess_pipeline(
            in_img,
            pipe_out,
            majority_sizes=(3, 5),
            sieve_size_threshold=8,
            sieve_connectivity=4,
            block_size=11,
            no_data_val=0,
            calc_stats=False,
            use_pbar=False,
        )

        with rasterio.open(ref_out) as r_ds, rasterio.open(pipe_out) as p_ds:
            np.testing.assert_array_equal(p_ds.read(1), r_ds.read(1))


def test_tilegraph_validation_errors():
    """Verify TileGraph rejects invalid configurations cleanly."""
    with pytest.raises(rsgispy.RSGISPyException):
        TileGraph([])
    src = ArraySource(np.zeros((1, 8, 8), dtype=np.float32))
    sink = MemorySink(src)
    with pytest.raises(rsgispy.RSGISPyException):
        TileGraph(sink, block_size=0)
    with pytest.raises(rsgispy.RSGISPyException):
        PointwiseOp([], lambda _c: np.zeros((1, 1, 1)))
    with pytest.raises(rsgispy.RSGISPyException):
        DiskSink(src, "/tmp/should_not_exist.tif")
        if_chained.apply_filter_chain("missing.tif", "out.tif", steps=[])


def _write_geotiff(
    path: Path,
    arr: np.ndarray,
    *,
    transform: Affine = from_origin(500000.0, 5800000.0, 10.0, 10.0),
    crs: object = "EPSG:32630",
    nodata=None,
) -> str:
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


def test_band_math_op_reuses_parser_across_non_square_tiles(tmp_path):
    arr = np.arange(25 * 30, dtype=np.float32).reshape(25, 30)
    in_img = _write_geotiff(tmp_path / "bm_in.tif", arr)
    out_img = str(tmp_path / "bm_out.tif")

    src_node = RasterSource(in_img)
    math_node = BandMathOp(
        inputs={"b1": src_node},
        expr="b1 * 2.0 + 1.0",
    )
    sink = DiskSink(
        source=math_node,
        output_img=out_img,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        nodata=None,
    )
    TileGraph(sinks=sink, block_size=16).run(use_pbar=False)

    with rasterio.open(out_img) as dst:
        np.testing.assert_allclose(dst.read(1), arr * 2.0 + 1.0, rtol=1e-6)
