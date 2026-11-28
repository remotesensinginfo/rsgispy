"""
Tests for :mod:`rsgispy.imagespectral` submodules
(``classification``, ``endmembers``, ``indices``, ``preprocessing``,
``transforms``, ``unmixing``).
"""

import importlib.util
import inspect
import json
import math
import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy
import rsgispy.imageprocess as img_proc
import rsgispy.imagespectral.classification as sc
import rsgispy.imagespectral.endmembers as em
import rsgispy.imagespectral.indices as si
import rsgispy.imagespectral.preprocessing as sp
import rsgispy.imagespectral.transforms as st
import rsgispy.imagespectral.unmixing as su
import rsgispy.imagespectral.endmembers as endm
import rsgispy.imageprocess as imgproc
from rasterio.transform import Affine
from pathlib import Path
import rsgispy.imagespectral.transforms as spec_tr
import rsgispy.imagespectral.unmixing as unmix

matplotlib.use("Agg")

HAVE_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAVE_OPENPYXL = importlib.util.find_spec("openpyxl") is not None
HAVE_TABLES = importlib.util.find_spec("tables") is not None


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    res=10.0,
    nodata=None,
    descriptions=None,
):
    """Write a 2D or 3D GeoTIFF raster."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(arr.dtype),
        crs="EPSG:32630",
        transform=from_origin(origin[0], origin[1], res, res),
        nodata=nodata,
    ) as dst:
        dst.write(arr)
        if descriptions is not None:
            for b_idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(b_idx, desc)


def _write_legacy_mtxt(path, em_arr):
    """Write a legacy RSGISLib ``.mtxt`` matrix file for conversion tests."""
    m, n = em_arr.shape
    vals = []
    for n_idx in range(n):
        for m_idx in range(m):
            vals.append(repr(float(em_arr[m_idx, n_idx])))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"m={m}\n")
        handle.write(f"n={n}\n")
        handle.write(",".join(vals) + "\n")


def test_exports_and_default_gdalformat():
    """Verify imagespectral exports across submodules and GTIFF defaults."""
    assert hasattr(em, "read_endmembers_df")
    assert hasattr(em, "write_endmembers_df")
    assert hasattr(em, "convert_mtxt_to_df")
    assert not hasattr(em, "read_endmembers_mtxt")
    assert not hasattr(em, "write_endmembers_mtxt")

    module_funcs = [
        (su, su.spec_unmix_ucls),
        (su, su.spec_unmix_nnls),
        (su, su.spec_unmix_fcls),
        (su, su.spec_unmix_spts_ucls),
        (su, su.spec_unmix_spts_nnls),
        (su, su.spec_unmix_spts_fcls),
        (su, su.spec_unmix_pymcr_nnls),
        (su, su.spec_unmix_pymcr_fcls),
        (su, su.exhcon_linear_spec_unmix),
        (su, su.rescale_unmixing_results),
        (su, su.predict_refl_linear_unmixing),
        (su, su.calc_unmixing_rmse_residual_err),
        (su, su.summarise_multi_endmember_linear_unmixing),
        (em, em.calc_ppi),
        (sc, sc.spectral_angle_mapper),
        (sc, sc.spectral_correlation_mapper),
        (sp, sp.spectral_smoothing),
        (sp, sp.mask_outliers_data_values),
        (sp, sp.polyfill_nan_data_values),
        (st, st.whiten_image),
        (st, st.calc_pca),
        (st, st.calc_mnf),
        (st, st.calc_tasseled_cap),
        (st, st.histogram_matching),
    ]
    for mod, fn in module_funcs:
        assert fn.__name__ in mod.__all__
        sig = inspect.signature(fn)
        assert sig.parameters["gdalformat"].default == "GTIFF"


def test_df_roundtrip_formats_plot_and_equality(tmp_path):
    """Verify DataFrame write/read across user-chosen formats, plot, and equality."""
    em_df = pd.DataFrame(
        {"Soil": [0.2, 0.4, 0.8], "Veg": [0.6, 0.2, 0.1]},
        index=pd.Index(["blue", "green", "nir"], name="band"),
    )

    formats = [
        ("csv", "em.csv", None),
        ("tsv", "em.tsv", None),
        ("json", "em.json", None),
        ("pickle", "em.pkl", None),
        ("csv_override", "em_custom.dat", "csv"),
    ]
    if HAVE_PYARROW:
        formats.extend(
            [
                ("parquet", "em.parquet", None),
                ("feather", "em.feather", None),
            ]
        )
    if HAVE_OPENPYXL:
        formats.append(("excel", "em.xlsx", None))
    if HAVE_TABLES:
        formats.append(("hdf", "em.h5", None))

    for _, fname, fmt_override in formats:
        out_path = tmp_path / fname
        em.write_endmembers_df(em_df, out_path, file_format=fmt_override)
        assert out_path.exists()

        read_back = em.read_endmembers_df(out_path, file_format=fmt_override)
        assert list(read_back.columns) == ["Soil", "Veg"]
        assert list(read_back.index) == ["blue", "green", "nir"]
        np.testing.assert_allclose(read_back.to_numpy(), em_df.to_numpy(), atol=1e-9)
        assert em.are_endmembers_equal(em_df, read_back, flt_dif=1e-7)

    plain_csv = tmp_path / "plain.csv"
    plain_csv.write_text("Soil,Veg\n0.2,0.6\n0.4,0.2\n0.8,0.1\n", encoding="utf-8")
    plain_df = em.read_endmembers_df(plain_csv, gain=2.0)
    assert list(plain_df.columns) == ["Soil", "Veg"]
    assert list(plain_df.index) == ["band_1", "band_2", "band_3"]
    np.testing.assert_allclose(plain_df.to_numpy(), em_df.to_numpy() / 2.0)

    em_df_close = em_df + 1e-5
    em_df_renamed = em_df.rename(columns={"Veg": "Forest"})
    assert em.are_endmembers_equal(em_df, em_df_close, flt_dif=1e-4)
    assert not em.are_endmembers_equal(em_df, em_df_close, flt_dif=1e-7)
    assert not em.are_endmembers_equal(em_df, em_df_renamed, check_names=True)
    assert em.are_endmembers_equal(em_df, em_df_renamed, check_names=False)

    plot_from_df = tmp_path / "em_df.png"
    plot_from_csv = tmp_path / "em_csv.png"
    fig, ax = em.plot_endmembers(
        em_df,
        plot_title="Test Endmembers (DF)",
        wavelengths=[480, 560, 660],
        out_plot_file=str(plot_from_df),
    )
    assert len(ax.get_lines()) == em_df.shape[1]
    plt.close(fig)
    fig, _ = em.plot_endmembers(str(tmp_path / "em.csv"), out_plot_file=str(plot_from_csv))
    plt.close(fig)
    assert plot_from_df.exists() and plot_from_df.stat().st_size > 0
    assert plot_from_csv.exists() and plot_from_csv.stat().st_size > 0


def test_convert_mtxt_to_df(tmp_path):
    """Verify legacy .mtxt conversion returns DataFrame and optionally writes to disk."""
    mtxt_path = tmp_path / "legacy.mtxt"
    out_csv = tmp_path / "converted.csv"
    em_arr = np.array([[0.2, 0.4, 0.8], [0.6, 0.2, 0.1]], dtype=np.float64)
    _write_legacy_mtxt(mtxt_path, em_arr)

    default_df = em.convert_mtxt_to_df(mtxt_path)
    assert default_df.shape == (3, 2)
    assert list(default_df.columns) == ["endmember_1", "endmember_2"]
    assert list(default_df.index) == ["band_1", "band_2", "band_3"]
    np.testing.assert_allclose(default_df.to_numpy().T, em_arr)

    named_df = em.convert_mtxt_to_df(
        mtxt_path,
        endmember_names=["Soil", "Veg"],
        band_names=["B2", "B3", "B8"],
        gain=2.0,
        out_file=out_csv,
    )
    assert list(named_df.columns) == ["Soil", "Veg"]
    assert list(named_df.index) == ["B2", "B3", "B8"]
    np.testing.assert_allclose(named_df.to_numpy().T, em_arr / 2.0)

    reread_df = em.read_endmembers_df(out_csv)
    assert em.are_endmembers_equal(named_df, reread_df, flt_dif=1e-9)

    with pytest.raises(rsgispy.RSGISPyException):
        em.convert_mtxt_to_df(mtxt_path, endmember_names=["OnlyOne"])
    with pytest.raises(rsgispy.RSGISPyException):
        em.convert_mtxt_to_df(mtxt_path, band_names=["B1", "B2"])


def test_extract_avg_endmembers_and_zonalstats_constants(tmp_path):
    """Verify extract_avg_endmembers returns a DataFrame with class columns."""
    img_path = tmp_path / "img.tif"
    vec_path = tmp_path / "rois.gpkg"
    out_csv = tmp_path / "extracted_em.csv"

    arr = np.zeros((3, 4, 4), dtype=np.float32)
    arr[:, :2, :2] = np.array([0.2, 0.4, 0.8])[:, None, None]
    arr[:, :2, 2:] = np.array([0.4, 0.6, 0.4])[:, None, None]
    arr[:, 2:, 2:] = np.array([0.6, 0.2, 0.1])[:, None, None]
    _write_raster(
        img_path,
        arr,
        origin=(0.0, 40.0),
        res=10.0,
        descriptions=["Blue", "Green", "NIR"],
    )

    gdf = gpd.GeoDataFrame(
        {"landcover": ["Soil", "Soil", "Veg"]},
        geometry=[
            box(0.0, 20.0, 20.0, 40.0),
            box(20.0, 20.0, 40.0, 40.0),
            box(20.0, 0.0, 40.0, 20.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(vec_path, driver="GPKG", layer="rois")

    em_df = em.extract_avg_endmembers(
        str(img_path),
        str(vec_path),
        "rois",
        out_file=str(out_csv),
        class_col="landcover",
        pxl_in_poly_method=1,
    )

    assert isinstance(em_df, pd.DataFrame)
    assert list(em_df.columns) == ["Soil", "Veg"]
    assert list(em_df.index) == ["Blue", "Green", "NIR"]
    np.testing.assert_allclose(em_df["Soil"].to_numpy(), [0.3, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(em_df["Veg"].to_numpy(), [0.6, 0.2, 0.1], atol=1e-6)
    assert out_csv.exists()
    assert em.are_endmembers_equal(em_df, em.read_endmembers_df(out_csv))

    em_df_gdf = em.extract_avg_endmembers(
        str(img_path),
        gdf,
        None,
        class_col="landcover",
        pxl_in_poly_method=1,
    )
    assert em.are_endmembers_equal(em_df, em_df_gdf)


def test_exhcon_unmixing_and_downstream_metrics(tmp_path):
    """Verify exhaustive constrained unmixing, rescaling, prediction, error, and PPI."""
    img_path = tmp_path / "refl.tif"
    em_path = tmp_path / "em.csv"
    exh_path = tmp_path / "exh.tif"
    resc_path = tmp_path / "resc.tif"
    pred_path = tmp_path / "pred.tif"
    err_path = tmp_path / "err.tif"
    ppi_path = tmp_path / "ppi.tif"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pd.DataFrame(
        {"Soil": em1, "Veg": em2},
        index=pd.Index(["Blue", "Green", "NIR"], name="band"),
    )
    em.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = em1
    arr[:, 0, 1] = em2
    arr[:, 1, 0] = 0.5 * em1 + 0.5 * em2
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    su.exhcon_linear_spec_unmix(
        str(img_path),
        str(exh_path),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        endmembers=str(em_path),
        step_res=0.1,
        use_pbar=False,
    )
    with rasterio.open(exh_path) as src:
        out_exh = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "RMSE"]
    assert out_exh.shape == (3, 2, 2)
    np.testing.assert_allclose(out_exh[:, 0, 0], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_exh[:, 0, 1], [0.0, 1.0, 0.0], atol=1e-5)
    # The all-zero (zero-norm) pixel is not unmixed: no data.
    np.testing.assert_allclose(out_exh[:, 1, 1], [-9999.0] * 3)
    assert abs((out_exh[0, 1, 0] + out_exh[1, 1, 0]) - 1.0) < 1e-5

    raw_coef_path = tmp_path / "raw_coef.tif"
    raw_coef = np.array(
        [[[0.6, -0.2], [0.3, 0.0]], [[0.6, 0.8], [0.7, 0.0]]],
        dtype=np.float32,
    )
    _write_raster(raw_coef_path, raw_coef, origin=(0.0, 20.0), res=10.0)
    su.rescale_unmixing_results(
        str(raw_coef_path), str(resc_path), calc_stats=False, use_pbar=False
    )
    with rasterio.open(resc_path) as src:
        resc = src.read()
    np.testing.assert_allclose(resc[:, 0, 0], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(resc[:, 0, 1], [0.0, 1.0], atol=1e-6)
    # No positive fraction: cannot be normalised, so no data.
    np.testing.assert_allclose(resc[:, 1, 1], [-9999.0, -9999.0])

    exact_coef = np.array(
        [[[1.0, 0.0], [0.5, 0.0]], [[0.0, 1.0], [0.5, 0.0]]],
        dtype=np.float32,
    )
    exact_coef_path = tmp_path / "exact_coef.tif"
    _write_raster(exact_coef_path, exact_coef, origin=(0.0, 20.0), res=10.0)

    su.predict_refl_linear_unmixing(
        str(exact_coef_path),
        str(pred_path),
        em_df,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(pred_path) as src:
        pred = src.read()
        assert list(src.descriptions) == ["Blue", "Green", "NIR"]
    np.testing.assert_allclose(pred, arr, atol=1e-6)

    su.calc_unmixing_rmse_residual_err(
        str(img_path),
        str(exact_coef_path),
        str(err_path),
        str(em_path),
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(err_path) as src:
        err = src.read()
        assert list(src.descriptions) == ["RMSE", "RMSPE", "Residual"]
        assert src.nodata == -9999.0
    np.testing.assert_allclose(err[:, 0, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(err[:, 1, 0], 0.0, atol=1e-5)
    # The all-zero input pixel has no error estimate.
    np.testing.assert_allclose(err[:, 1, 1], [-9999.0] * 3)

    em.calc_ppi(
        str(img_path),
        str(ppi_path),
        niters=50,
        rnd_seed=42,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ppi_path) as src:
        ppi = src.read(1)
        assert src.dtypes[0] == "uint16"
    assert int(np.sum(ppi)) > 0


def test_summarise_multi_endmember_linear_unmixing(tmp_path):
    """Verify multi-model summary picks lower-RMSPE model using DataFrame/file columns."""
    refl_path = tmp_path / "refl.tif"
    em_a_path = tmp_path / "em_a.csv"
    unmix_a_path = tmp_path / "unmix_a.tif"
    unmix_b_path = tmp_path / "unmix_b.tif"
    out_unmix = tmp_path / "best_unmix.tif"
    out_ref = tmp_path / "best_ref.tif"

    em_veg = np.array([0.1, 0.3, 0.8], dtype=np.float64)
    em_soil = np.array([0.5, 0.4, 0.2], dtype=np.float64)
    em_water = np.array([0.2, 0.1, 0.05], dtype=np.float64)

    em_a_df = pd.DataFrame({"Veg": em_veg, "Soil": em_soil})
    em_b_df = pd.DataFrame({"Veg": em_veg, "Water": em_water})
    em.write_endmembers_df(em_a_df, str(em_a_path))

    refl = np.zeros((3, 2, 2), dtype=np.float32)
    refl[:, :, 0] = em_soil[:, None]
    refl[:, :, 1] = em_water[:, None]
    _write_raster(refl_path, refl, origin=(0.0, 20.0), res=10.0)

    coef_a = np.zeros((2, 2, 2), dtype=np.float32)
    coef_a[1, :, 0] = 1.0
    _write_raster(unmix_a_path, coef_a, origin=(0.0, 20.0), res=10.0)

    coef_b = np.zeros((2, 2, 2), dtype=np.float32)
    coef_b[1, :, 1] = 1.0
    _write_raster(unmix_b_path, coef_b, origin=(0.0, 20.0), res=10.0)

    info_a = em.ImageEndmemberInfo(str(unmix_a_path), str(em_a_path))
    info_b = em.ImageEndmemberInfo(str(unmix_b_path), em_b_df)
    assert info_a == em.ImageEndmemberInfo(str(unmix_a_path), str(em_a_path))
    assert info_b == em.ImageEndmemberInfo(str(unmix_b_path), em_b_df.copy())
    assert info_a != info_b

    su.summarise_multi_endmember_linear_unmixing(
        str(refl_path),
        str(out_unmix),
        str(out_ref),
        [info_a, info_b],
        tmp_dir=str(tmp_path / "tmp"),
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_ref) as src:
        ref = src.read(1)
    assert np.all(ref[:, 0] == 1)
    assert np.all(ref[:, 1] == 2)

    with rasterio.open(out_unmix) as src:
        comb = src.read()
        assert list(src.descriptions) == ["Veg", "Soil", "Water"]
    np.testing.assert_allclose(comb[:, 0, 0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(comb[:, 0, 1], [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.skipif(not su.HAVE_PYSPTOOLS, reason="pysptools not installed")
def test_pysptools_unmixers(tmp_path):
    """Verify PySptools UCLS, NNLS, and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pd.DataFrame({"Soil": em1, "Veg": em2})
    em.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_ucls = tmp_path / "ucls.tif"
    su.spec_unmix_spts_ucls(
        str(img_path),
        str(out_ucls),
        str(em_path),
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_ucls) as src:
        assert list(src.descriptions) == ["Soil", "Veg"]
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-3)

    out_nnls = tmp_path / "nnls.tif"
    su.spec_unmix_spts_nnls(
        str(img_path),
        str(out_nnls),
        em_df,
        str(msk_path),
        1,
        weight=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    if su.HAVE_CVXOPT:
        out_fcls = tmp_path / "fcls.tif"
        su.spec_unmix_spts_fcls(
            str(img_path),
            str(out_fcls),
            str(em_path),
            str(msk_path),
            1,
            calc_stats=False,
            use_pbar=False,
        )
        with rasterio.open(out_fcls) as src:
            np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


@pytest.mark.skipif(not su.HAVE_PYMCR, reason="pymcr not installed")
def test_pymcr_unmixers(tmp_path):
    """Verify PyMCR NNLS and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.json"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pd.DataFrame({"Soil": em1, "Veg": em2})
    em.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "pymcr_nnls.tif"
    su.spec_unmix_pymcr_nnls(
        str(img_path),
        str(out_nnls),
        str(em_path),
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        assert list(src.descriptions) == ["Soil", "Veg"]
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    out_fcls = tmp_path / "pymcr_fcls.tif"
    su.spec_unmix_pymcr_fcls(
        str(img_path),
        str(out_fcls),
        em_df,
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


def test_recommended_numpy_ucls_unmixer(tmp_path):
    """Verify recommended pure-NumPy UCLS unmixing with DataFrame and CSV files."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em3 = np.array([0.3, 0.7, 0.2], dtype=np.float64)
    em_df = pd.DataFrame({"Soil": em1, "Veg": em2, "Water": em3})
    em.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.2 * em1 + 0.5 * em2 + 0.3 * em3
    arr[:, 0, 1] = 0.8 * em1 + 0.2 * em2
    arr[:, 1, 0] = em3
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.array([[[1, 1], [0, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_masked = tmp_path / "numpy_ucls.tif"
    su.spec_unmix_ucls(
        str(img_path),
        str(out_masked),
        str(em_path),
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_masked) as src:
        res = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Water"]
    np.testing.assert_allclose(res[:, 0, 0], [0.2, 0.5, 0.3], atol=1e-5)
    np.testing.assert_allclose(res[:, 0, 1], [0.8, 0.2, 0.0], atol=1e-5)
    # Masked out and all-zero pixels are not unmixed: no data.
    np.testing.assert_allclose(res[:, 1, 0], [-9999.0] * 3)
    np.testing.assert_allclose(res[:, 1, 1], [-9999.0] * 3)

    out_nomask = tmp_path / "numpy_ucls_nomask.tif"
    su.spec_unmix_ucls(
        str(img_path),
        str(out_nomask),
        em_df,
        None,
        1,
        weight=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nomask) as src:
        res_nm = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Water"]
    np.testing.assert_allclose(res_nm[:, 1, 0], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(res_nm[:, 1, 1], [-9999.0] * 3)


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_recommended_scipy_nnls_fcls_unmixers(tmp_path):
    """Verify recommended SciPy NNLS and FCLS unmixers using DataFrame and files."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.1, 0.2, 0.7, 0.3], dtype=np.float64)
    em2 = np.array([0.6, 0.3, 0.1, 0.2], dtype=np.float64)
    em3 = np.array([0.2, 0.6, 0.3, 0.7], dtype=np.float64)
    em_df = pd.DataFrame({"Soil": em1, "Veg": em2, "Shadow": em3})
    em.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((4, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.25 * em1 + 0.45 * em2 + 0.30 * em3
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = 1.4 * em2 + 0.2 * em3
    arr[:, 1, 1] = np.clip(1.2 * em1 - 0.15 * em2 + 0.05 * em3, 0.01, None)
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "scipy_nnls.tif"
    su.spec_unmix_nnls(
        str(img_path),
        str(out_nnls),
        str(em_path),
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        nnls_arr = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Shadow"]
    assert np.all(nnls_arr >= -1e-7)
    np.testing.assert_allclose(nnls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(nnls_arr[:, 1, 0], [0.0, 1.4, 0.2], atol=1e-5)

    out_fcls = tmp_path / "scipy_fcls.tif"
    su.spec_unmix_fcls(
        str(img_path),
        str(out_fcls),
        em_df,
        str(msk_path),
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        fcls_arr = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Shadow"]
    assert np.all(fcls_arr >= -1e-7)
    np.testing.assert_allclose(np.sum(fcls_arr, axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 1], [1.0, 0.0, 0.0], atol=1e-5)

    scaled_img = tmp_path / "refl_scaled.tif"
    ext = ".parquet" if HAVE_PYARROW else ".json"
    scaled_em = tmp_path / f"em_scaled{ext}"
    _write_raster(
        scaled_img, (arr * 10000.0).astype(np.float32), origin=(0.0, 20.0), res=10.0
    )
    em.write_endmembers_df(em_df * 10000.0, str(scaled_em))
    out_fcls_scaled = tmp_path / "scipy_fcls_scaled.tif"
    su.spec_unmix_fcls(
        str(scaled_img),
        str(out_fcls_scaled),
        str(scaled_em),
        None,
        1,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls_scaled) as src:
        fcls_scaled = src.read()
    np.testing.assert_allclose(fcls_scaled, fcls_arr, atol=1e-5)


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_unmixing_no_data_outputs(tmp_path):
    """Unmixing outputs use a distinct no data value; fractions of 0 are valid."""
    # pylint: disable=too-many-locals,too-many-statements
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pd.DataFrame({"Soil": em1, "Veg": em2})

    in_nd = 7.0
    arr = np.zeros((3, 3, 3), dtype=np.float32)
    arr[:, 0, 0] = em1  # pure: fractions [1, 0]
    arr[:, 0, 1] = em2  # pure: fractions [0, 1]
    arr[:, 0, 2] = 0.5 * em1 + 0.5 * em2
    arr[:, 1, 0] = [0.3, 0.3, 0.0]  # a band <= 0: skipped
    arr[:, 1, 1] = em1  # outside the valid mask: skipped
    arr[:, 1, 2] = in_nd  # input no data: skipped
    arr[:, 2, 0] = [np.nan, 0.3, 0.3]  # non-finite: skipped
    arr[:, 2, 1] = [0.3, -0.1, 0.3]  # negative: skipped
    arr[:, 2, 2] = em1
    _write_raster(img_path, arr, origin=(0.0, 30.0), res=10.0, nodata=in_nd)
    msk = np.ones((1, 3, 3), dtype=np.uint8)
    msk[0, 1, 1] = 0
    _write_raster(msk_path, msk, origin=(0.0, 30.0), res=10.0)
    skipped = np.zeros((3, 3), dtype=bool)
    skipped[1, :] = True
    skipped[2, :2] = True

    fcls_path = tmp_path / "fcls.tif"
    su.spec_unmix_fcls(
        str(img_path), str(fcls_path), em_df, str(msk_path), 1,
        calc_stats=True, use_pbar=False,
    )
    with rasterio.open(fcls_path) as src:
        assert src.nodatavals == (-9999.0, -9999.0)
        fcls = src.read()
        veg_min = float(src.tags(2)["STATISTICS_MINIMUM"])
        soil_min = float(src.tags(1)["STATISTICS_MINIMUM"])
    np.testing.assert_allclose(fcls[:, 0, 0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(fcls[:, 0, 1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(fcls[:, 0, 2], [0.5, 0.5], atol=1e-6)
    assert np.all(fcls[:, skipped] == -9999.0)
    # Fractions of exactly 0 are valid data and included in the statistics.
    assert veg_min == pytest.approx(0.0, abs=1e-6)
    assert soil_min == pytest.approx(0.0, abs=1e-6)

    ucls_path = tmp_path / "ucls.tif"
    su.spec_unmix_ucls(
        str(img_path), str(ucls_path), em_df, out_no_data_val=-1.0,
        calc_stats=False, use_pbar=False,
    )
    with rasterio.open(ucls_path) as src:
        assert src.nodatavals == (-1.0, -1.0)
        ucls = src.read()
    no_msk_skip = skipped.copy()
    no_msk_skip[1, 1] = False
    assert np.all(ucls[:, no_msk_skip] == -1.0)
    np.testing.assert_allclose(ucls[:, 1, 1], [1.0, 0.0], atol=1e-6)

    # Rescaling: input no data and pixels without a positive fraction are no
    # data; a clamped fraction of 0 stays valid.
    raw_path = tmp_path / "raw.tif"
    raw = np.array(
        [[[0.6, 0.0, -9999.0, 0.3]], [[-0.2, 0.0, -9999.0, 0.9]]],
        dtype=np.float32,
    )
    _write_raster(raw_path, raw, origin=(0.0, 10.0), res=10.0, nodata=-9999.0)
    resc_path = tmp_path / "resc.tif"
    su.rescale_unmixing_results(
        str(raw_path), str(resc_path), calc_stats=True, use_pbar=False
    )
    with rasterio.open(resc_path) as src:
        assert src.nodata == -9999.0
        resc = src.read()
        assert float(src.tags(2)["STATISTICS_MINIMUM"]) == pytest.approx(0.0)
    np.testing.assert_allclose(
        resc[:, 0, :],
        [[1.0, -9999.0, -9999.0, 0.25], [0.0, -9999.0, -9999.0, 0.75]],
        atol=1e-6,
    )

    # Predicted reflectance and errors are no data where unmixing was skipped.
    pred_path = tmp_path / "pred.tif"
    su.predict_refl_linear_unmixing(
        str(fcls_path), str(pred_path), em_df, calc_stats=True, use_pbar=False
    )
    with rasterio.open(pred_path) as src:
        assert src.nodata == -9999.0
        pred = src.read()
    np.testing.assert_allclose(pred[:, 0, 0], em1, atol=1e-6)
    assert np.all(pred[:, skipped] == -9999.0)

    err_path = tmp_path / "err.tif"
    su.calc_unmixing_rmse_residual_err(
        str(img_path), str(fcls_path), str(err_path), em_df,
        calc_stats=True, use_pbar=False,
    )
    with rasterio.open(err_path) as src:
        assert src.nodata == -9999.0
        err = src.read()
    np.testing.assert_allclose(err[:, 0, 0], 0.0, atol=1e-5)
    assert np.all(err[:, skipped] == -9999.0)

    # Multi-model summary: a model's no data pixels are never selected.
    alt_path = tmp_path / "alt.tif"
    alt = fcls.copy()
    alt[:, 0, 0] = -9999.0
    _write_raster(alt_path, alt, origin=(0.0, 30.0), res=10.0, nodata=-9999.0)
    best_path = tmp_path / "best.tif"
    ref_path = tmp_path / "best_ref.tif"
    su.summarise_multi_endmember_linear_unmixing(
        str(img_path), str(best_path), str(ref_path),
        [em.ImageEndmemberInfo(str(alt_path), em_df),
         em.ImageEndmemberInfo(str(fcls_path), em_df)],
        out_no_data_val=-9999.0, calc_stats=True, use_pbar=False,
    )
    with rasterio.open(best_path) as src:
        assert src.nodata == -9999.0
        best = src.read()
    with rasterio.open(ref_path) as src:
        ref = src.read(1)
    assert ref[0, 0] == 2 and ref[0, 1] == 1
    np.testing.assert_allclose(best[:, 0, 0], [1.0, 0.0], atol=1e-6)
    assert np.all(ref[skipped] == 0)
    assert np.all(best[:, skipped] == -9999.0)

    # Exhaustive search: no data where not unmixed, sized to the data type.
    exh_path = tmp_path / "exh.tif"
    su.exhcon_linear_spec_unmix(
        str(img_path), str(exh_path), em_df, step_res=0.1, use_pbar=False
    )
    with rasterio.open(exh_path) as src:
        assert src.nodata == -9999.0
        exh = src.read()
    np.testing.assert_allclose(exh[:2, 0, 0], [1.0, 0.0], atol=1e-5)
    assert np.all(exh[:, 1, 2] == -9999.0)  # input no data
    assert np.all(exh[:, 2, 0] == -9999.0)  # non-finite
    exh_u8 = tmp_path / "exh_u8.tif"
    su.exhcon_linear_spec_unmix(
        str(img_path), str(exh_u8), em_df, step_res=0.1, gain=100.0,
        datatype=rsgispy.TYPE_8UINT, use_pbar=False,
    )
    with rasterio.open(exh_u8) as src:
        assert src.nodata == 255.0
        assert np.all(src.read()[:, 1, 2] == 255)
    with pytest.raises(rsgispy.RSGISPyException):
        su.exhcon_linear_spec_unmix(
            str(img_path), str(exh_u8), em_df, datatype=rsgispy.TYPE_8UINT,
            out_no_data_val=-9999.0, use_pbar=False,
        )


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_batched_nnls_fcls_equivalence_and_edge_cases():
    """Verify batched active-set solvers match SciPy reference across m=1, m=5, and m=22."""
    import scipy.optimize

    rng = np.random.default_rng(123)

    em1 = rng.uniform(0.1, 0.9, size=(1, 4))
    px1 = rng.uniform(0.1, 0.9, size=(10, 4))
    out_fcls1 = su._scipy_fcls_solver(px1, em1)  # pylint: disable=protected-access
    np.testing.assert_allclose(out_fcls1, np.ones((10, 1)), atol=1e-12)
    out_nnls1 = su._scipy_nnls_solver(px1, em1)  # pylint: disable=protected-access
    ref_nnls1 = np.array([scipy.optimize.nnls(em1.T, p)[0] for p in px1])
    np.testing.assert_allclose(out_nnls1, ref_nnls1, atol=1e-10)

    em5 = rng.uniform(0.05, 0.85, size=(5, 8))
    w_true = rng.dirichlet(np.ones(5) * 0.4, size=200)
    px5 = w_true @ em5 + rng.normal(0.0, 0.05, size=(200, 8))
    px5[:20] = -0.3 * em5[0] + 1.5 * em5[2] + rng.normal(0.0, 0.02, size=(20, 8))

    out_nnls5 = su._scipy_nnls_solver(px5, em5)  # pylint: disable=protected-access
    ref_nnls5 = np.array([scipy.optimize.nnls(em5.T, p)[0] for p in px5])
    np.testing.assert_allclose(out_nnls5, ref_nnls5, atol=1e-8)

    out_fcls5 = su._scipy_fcls_solver(px5, em5)  # pylint: disable=protected-access
    ref_fcls5 = su._scipy_fcls_heinz_chang_loop(px5, em5)  # pylint: disable=protected-access
    assert np.all(out_fcls5 >= -1e-12)
    np.testing.assert_allclose(np.sum(out_fcls5, axis=1), 1.0, atol=1e-10)
    res_batched = np.sum((out_fcls5 @ em5 - px5) ** 2, axis=1)
    res_ref = np.sum((ref_fcls5 @ em5 - px5) ** 2, axis=1)
    assert np.all(res_batched <= res_ref + 1e-7)

    em22 = rng.uniform(0.1, 0.9, size=(22, 25))
    px22 = rng.uniform(0.1, 0.9, size=(5, 25))
    out_nnls22 = su._scipy_nnls_solver(px22, em22)  # pylint: disable=protected-access
    assert out_nnls22.shape == (5, 22)
    assert np.all(out_nnls22 >= -1e-12)
    out_fcls22 = su._scipy_fcls_solver(px22, em22)  # pylint: disable=protected-access
    assert out_fcls22.shape == (5, 22)
    np.testing.assert_allclose(np.sum(out_fcls22, axis=1), 1.0, atol=1e-6)


def test_optional_dependency_friendly_errors(monkeypatch, tmp_path):
    """Verify optional unmixers raise RSGISPyException with pip install hints when missing."""
    em_df = pd.DataFrame({"Soil": [0.2, 0.4], "Veg": [0.6, 0.2]})
    dummy_in = str(tmp_path / "in.tif")
    dummy_out = str(tmp_path / "out.tif")

    checks = [
        (su.spec_unmix_nnls, {"HAVE_SCIPY": False}, "scipy"),
        (su.spec_unmix_fcls, {"HAVE_SCIPY": False}, "scipy"),
        (su.spec_unmix_spts_ucls, {"HAVE_PYSPTOOLS": False}, "pysptools"),
        (su.spec_unmix_spts_nnls, {"HAVE_PYSPTOOLS": False}, "pysptools"),
        (su.spec_unmix_spts_fcls, {"HAVE_PYSPTOOLS": False}, "pysptools"),
        (
            su.spec_unmix_spts_fcls,
            {"HAVE_PYSPTOOLS": True, "HAVE_CVXOPT": False},
            "cvxopt",
        ),
        (su.spec_unmix_pymcr_nnls, {"HAVE_PYMCR": False}, "pyMCR"),
        (su.spec_unmix_pymcr_fcls, {"HAVE_PYMCR": False}, "pyMCR"),
    ]
    for fn, flags, pkg in checks:
        with monkeypatch.context() as mp:
            for flag, val in flags.items():
                mp.setattr(su, flag, val)
            with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
                fn(dummy_in, dummy_out, em_df)
            with pytest.raises(rsgispy.RSGISPyException, match=pkg):
                fn(dummy_in, dummy_out, em_df)


def test_spectral_angle_mapper(tmp_path):
    """Test spectral_angle_mapper accuracy, scale invariance, thresholding, and nodata."""
    em_veg = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    em_soil = np.array([3.0, 1.0, 1.0], dtype=np.float64)
    em_df = pd.DataFrame({"Veg": em_veg, "Soil": em_soil})
    em_csv = tmp_path / "sam_em.csv"
    em.write_endmembers_df(em_df, str(em_csv))

    img_path = tmp_path / "sam_in.tif"
    msk_path = tmp_path / "sam_msk.tif"
    out_ang = tmp_path / "sam_ang.tif"
    out_cls = tmp_path / "sam_cls.tif"
    out_min = tmp_path / "sam_min.tif"

    cube = np.zeros((3, 2, 3), dtype=np.float32)
    cube[:, 0, 0] = (3.0 * em_veg).astype(np.float32)
    cube[:, 0, 1] = (0.5 * em_soil).astype(np.float32)
    cube[:, 0, 2] = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    cube[:, 1, 0] = em_veg.astype(np.float32)
    cube[:, 1, 1] = -9999.0
    cube[:, 1, 2] = 0.0
    _write_raster(img_path, cube, origin=(0.0, 20.0), res=10.0, nodata=-9999.0)

    msk = np.array([[[1, 1, 1], [0, 1, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    sc.spectral_angle_mapper(
        str(img_path),
        str(out_ang),
        str(em_csv),
        out_class_img=str(out_cls),
        out_min_angle_img=str(out_min),
        degrees=True,
        max_angle=15.0,
        valid_msk_img=str(msk_path),
        valid_msk_val=1,
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_ang) as src:
        ang = src.read()
        assert list(src.descriptions) == ["Veg", "Soil"]
    assert ang[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
    assert ang[1, 0, 1] == pytest.approx(0.0, abs=1e-3)
    expected_deg = math.degrees(math.acos(2.0 / math.sqrt(21.0)))
    assert ang[0, 0, 2] == pytest.approx(expected_deg, abs=1e-3)
    assert ang[0, 1, 0] == pytest.approx(-9999.0)
    assert ang[0, 1, 1] == pytest.approx(-9999.0)
    assert ang[0, 1, 2] == pytest.approx(-9999.0)

    with rasterio.open(out_cls) as src:
        cls = src.read(1)
    assert cls[0, 0] == 1
    assert cls[0, 1] == 2
    assert cls[0, 2] == 0
    assert cls[1, 0] == 0
    assert cls[1, 1] == 0
    assert cls[1, 2] == 0

    with rasterio.open(out_min) as src:
        m_ang = src.read(1)
    assert m_ang[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert m_ang[0, 1] == pytest.approx(0.0, abs=1e-3)

    with pytest.raises(rsgispy.RSGISPyException):
        sc.spectral_angle_mapper(
            str(img_path),
            str(tmp_path / "bad_bands.tif"),
            pd.DataFrame({"A": [1.0, 2.0]}),
        )
    with pytest.raises(rsgispy.RSGISPyException):
        sc.spectral_angle_mapper(
            str(img_path),
            str(tmp_path / "bad_zero.tif"),
            pd.DataFrame({"Zero": [0.0, 0.0, 0.0]}),
        )


def test_spectral_correlation_mapper(tmp_path):
    """Test spectral_correlation_mapper affine invariance, thresholds, and nodata."""
    em_veg = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float64)
    em_soil = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float64)
    em_df = pd.DataFrame({"Veg": em_veg, "Soil": em_soil})

    img_path = tmp_path / "scm_in.tif"
    msk_path = tmp_path / "scm_msk.tif"
    out_corr = tmp_path / "scm_corr.tif"
    out_cls = tmp_path / "scm_cls.tif"
    out_max = tmp_path / "scm_max.tif"

    cube = np.zeros((4, 2, 3), dtype=np.float32)
    cube[:, 0, 0] = (2.5 * em_veg + 10.0).astype(np.float32)
    cube[:, 0, 1] = (0.5 * em_soil - 1.0).astype(np.float32)
    cube[:, 0, 2] = (-em_veg + 5.0).astype(np.float32)
    cube[:, 1, 0] = em_veg.astype(np.float32)
    cube[:, 1, 1] = -9999.0
    cube[:, 1, 2] = 5.0
    _write_raster(img_path, cube, origin=(0.0, 20.0), res=10.0, nodata=-9999.0)

    msk = np.array([[[1, 1, 1], [0, 1, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    sc.spectral_correlation_mapper(
        str(img_path),
        str(out_corr),
        em_df,
        out_class_img=str(out_cls),
        out_max_corr_img=str(out_max),
        min_corr=0.8,
        valid_msk_img=str(msk_path),
        valid_msk_val=1,
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_corr) as src:
        corr = src.read()
        assert list(src.descriptions) == ["Veg", "Soil"]
    assert corr[0, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert corr[1, 0, 1] == pytest.approx(1.0, abs=1e-5)
    assert corr[0, 0, 2] == pytest.approx(-1.0, abs=1e-5)
    assert corr[0, 1, 0] == pytest.approx(-9999.0)
    assert corr[0, 1, 1] == pytest.approx(-9999.0)
    assert corr[0, 1, 2] == pytest.approx(-9999.0)

    with rasterio.open(out_cls) as src:
        cls = src.read(1)
    assert cls[0, 0] == 1
    assert cls[0, 1] == 2
    assert cls[0, 2] == 0
    assert cls[1, 0] == 0
    assert cls[1, 1] == 0
    assert cls[1, 2] == 0

    with rasterio.open(out_max) as src:
        max_c = src.read(1)
        assert list(src.descriptions) == ["max_corr"]
    assert max_c[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert max_c[0, 1] == pytest.approx(1.0, abs=1e-5)

    with pytest.raises(rsgispy.RSGISPyException):
        sc.spectral_correlation_mapper(
            str(img_path),
            str(tmp_path / "bad_const.tif"),
            pd.DataFrame({"Const": [2.0, 2.0, 2.0, 2.0]}),
        )
    with pytest.raises(rsgispy.RSGISPyException):
        sc.spectral_correlation_mapper(
            str(img_path),
            str(tmp_path / "bad_thresh.tif"),
            em_df,
            out_class_img=str(tmp_path / "bad_thresh_cls.tif"),
            min_corr=1.5,
        )
    single_img = tmp_path / "single_band.tif"
    _write_raster(single_img, np.ones((1, 2, 2), dtype=np.float32))
    with pytest.raises(rsgispy.RSGISPyException):
        sc.spectral_correlation_mapper(
            str(single_img),
            str(tmp_path / "bad_1band.tif"),
            pd.DataFrame({"A": [1.0]}),
        )


def test_spectral_smoothing_and_outlier_mask(tmp_path):
    """Test spectral_smoothing (with calc_stats) and mask_outliers_data_values."""
    n_bands = 7
    wavelengths = np.arange(n_bands, dtype=np.float64)
    clean_spec = 2.0 * wavelengths + 1.0
    cube = (
        np.broadcast_to(clean_spec[:, None, None], (n_bands, 6, 6))
        .astype(np.float32)
        .copy()
    )

    in_img = tmp_path / "spec.tif"
    msk_img = tmp_path / "spec_msk.tif"
    _write_raster(in_img, cube, nodata=0.0)
    msk = np.zeros((6, 6), dtype=np.uint8)
    msk[1:5, 1:5] = 1
    _write_raster(msk_img, msk)

    smooth_out = tmp_path / "smooth.tif"
    sp.spectral_smoothing(
        str(in_img),
        str(smooth_out),
        str(msk_img),
        1,
        win_len=5,
        polyorder=2,
        gdalformat="GTIFF",
        calc_stats=True,
    )
    with rasterio.open(smooth_out) as src:
        sm = src.read()
        assert np.allclose(sm[:, 2, 2], clean_spec, atol=1e-4)
        assert np.allclose(sm[:, 0, 0], 0.0)
        assert "STATISTICS_MEAN" in src.tags(1)

    outl_arr = cube.copy()
    outl_arr[3, 2, 2] = 500.0
    outl_in = tmp_path / "outl_in.tif"
    outl_out = tmp_path / "outl_out.tif"
    _write_raster(outl_in, outl_arr)
    sp.mask_outliers_data_values(
        str(outl_in),
        str(outl_out),
        str(msk_img),
        1,
        lower=5,
        upper=85,
        gdalformat="GTIFF",
        calc_stats=False,
    )
    with rasterio.open(outl_out) as src:
        assert np.isnan(src.read()[3, 2, 2])


def test_polyfill_nan_data_values(tmp_path):
    """Test polyfill_nan_data_values reconstructs NaN spectral bands."""
    wavelengths = np.arange(7, dtype=np.float64)
    clean_spec = 2.0 * wavelengths + 1.0
    nan_arr = (
        np.broadcast_to(clean_spec[:, None, None], (7, 6, 6))
        .astype(np.float32)
        .copy()
    )
    nan_arr[3, 2, 2] = np.nan

    nan_in = tmp_path / "nan_in.tif"
    msk_img = tmp_path / "poly_msk.tif"
    poly_out = tmp_path / "poly_out.tif"
    _write_raster(nan_in, nan_arr)
    _write_raster(msk_img, np.ones((6, 6), dtype=np.uint8))

    sp.polyfill_nan_data_values(
        str(nan_in),
        str(poly_out),
        wavelengths.tolist(),
        str(msk_img),
        1,
        polyorder=2,
        gdalformat="GTIFF",
        calc_stats=False,
    )
    with rasterio.open(poly_out) as src:
        assert src.read()[3, 2, 2] == pytest.approx(clean_spec[3], abs=1e-3)


def test_whiten_image(tmp_path):
    """Test whiten_image produces uncorrelated unit-variance valid spectra."""
    rng = np.random.default_rng(99)
    w_in_arr = rng.normal(0.0, 1.0, (3, 12, 12)).astype(np.float32)
    w_in_arr[1] = (
        w_in_arr[0] * 2.0 + rng.normal(0.0, 0.5, (12, 12)).astype(np.float32)
    )
    w_in = tmp_path / "w_in.tif"
    w_msk = tmp_path / "w_msk.tif"
    w_out = tmp_path / "w_out.tif"
    _write_raster(w_in, w_in_arr)
    _write_raster(w_msk, np.ones((12, 12), dtype=np.uint8))

    st.whiten_image(str(w_in), str(w_out), str(w_msk), 1, gdalformat="GTIFF")
    with rasterio.open(w_out) as src:
        cov = np.cov(src.read().reshape(3, -1))
        assert np.allclose(cov, np.eye(3), atol=1e-2)


def test_spectral_pca_and_mnf_windowed(tmp_path):
    """Test streaming windowed PCA and MNF against full-matrix linear algebra."""
    rng = np.random.default_rng(12345)
    height, width, n_bands = 20, 20, 4
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    s1 = np.sin(xx / 4.0) + np.cos(yy / 5.0)
    s2 = 0.5 * xx / width - 0.3 * yy / height
    clean = np.stack(
        [
            10.0 + 3.0 * s1 + 1.0 * s2,
            20.0 + 6.0 * s1 - 2.0 * s2,
            5.0 - 1.5 * s1 + 4.0 * s2,
            12.0 + 0.5 * s1 + 0.5 * s2,
        ],
        axis=0,
    )
    noise = rng.normal(0.0, 0.05, size=(n_bands, height, width))
    noise[3] = rng.normal(0.0, 2.0, size=(height, width))
    cube = (clean + noise).astype(np.float32)
    cube[:, 0, 0] = -9999.0

    in_img = str(tmp_path / "hyp_in.tif")
    _write_raster(in_img, cube, nodata=-9999.0)
    msk = np.ones((height, width), dtype=np.uint8)
    msk[0, 1] = 0
    msk_img = str(tmp_path / "hyp_msk.tif")
    _write_raster(msk_img, msk)

    pca_single = str(tmp_path / "pca_single.tif")
    pca_multi = str(tmp_path / "pca_multi.tif")
    pca_json = str(tmp_path / "pca_stats.json")

    info_single = st.calc_pca(
        in_img,
        pca_single,
        n_components=2,
        valid_msk_img=msk_img,
        valid_msk_val=1,
        standardise=True,
        out_stats_json=pca_json,
        calc_stats=True,
        use_pbar=False,
    )
    assert info_single["n_valid_pixels"] == height * width - 2
    assert info_single["n_components"] == 2
    assert os.path.exists(pca_json)
    with open(pca_json, "r", encoding="utf-8") as handle:
        loaded_json = json.load(handle)
    assert loaded_json["eigenvalues"] == pytest.approx(info_single["eigenvalues"])

    img_proc.set_default_block_size(7)
    try:
        info_multi = st.calc_pca(
            in_img,
            pca_multi,
            n_components=2,
            valid_msk_img=msk_img,
            valid_msk_val=1,
            standardise=True,
            calc_stats=False,
            use_pbar=False,
        )
    finally:
        img_proc.reset_default_block_size()

    assert info_multi["mean"] == pytest.approx(info_single["mean"], rel=1e-10)
    assert info_multi["eigenvalues"] == pytest.approx(
        info_single["eigenvalues"], rel=1e-10
    )
    with rasterio.open(pca_single) as s_src, rasterio.open(pca_multi) as m_src:
        assert list(s_src.descriptions) == ["PC_1", "PC_2"]
        np.testing.assert_allclose(s_src.read(), m_src.read(), rtol=1e-5, atol=1e-5)
        assert s_src.read(1)[0, 0] == pytest.approx(-9999.0)
        assert s_src.read(1)[0, 1] == pytest.approx(-9999.0)

    mnf_single = str(tmp_path / "mnf_single.tif")
    mnf_multi = str(tmp_path / "mnf_multi.tif")
    mnf_json = str(tmp_path / "mnf_stats.json")

    mnf_s_info = st.calc_mnf(
        in_img,
        mnf_single,
        n_components=3,
        valid_msk_img=msk_img,
        valid_msk_val=1,
        noise_direction="both",
        out_stats_json=mnf_json,
        calc_stats=True,
        use_pbar=False,
    )
    img_proc.set_default_block_size(7)
    try:
        mnf_m_info = st.calc_mnf(
            in_img,
            mnf_multi,
            n_components=3,
            valid_msk_img=msk_img,
            valid_msk_val=1,
            noise_direction="both",
            calc_stats=False,
            use_pbar=False,
        )
    finally:
        img_proc.reset_default_block_size()

    assert mnf_s_info["n_noise_pairs"] == mnf_m_info["n_noise_pairs"]
    assert mnf_s_info["mnf_eigenvalues"] == pytest.approx(
        mnf_m_info["mnf_eigenvalues"], rel=1e-10
    )
    vals = mnf_s_info["mnf_eigenvalues"]
    assert vals[0] > vals[1] > vals[2]
    assert os.path.exists(mnf_json)

    with rasterio.open(mnf_single) as s_src, rasterio.open(mnf_multi) as m_src:
        assert list(s_src.descriptions) == ["MNF_1", "MNF_2", "MNF_3"]
        np.testing.assert_allclose(s_src.read(), m_src.read(), rtol=1e-5, atol=1e-5)

    with pytest.raises(rsgispy.RSGISPyException):
        st.calc_mnf(in_img, str(tmp_path / "bad.tif"), noise_direction="diag")
    with pytest.raises(rsgispy.RSGISPyException):
        st.calc_pca(in_img, str(tmp_path / "bad_pca.tif"), n_components=9)


def test_spectral_tasseled_cap_and_histogram_matching(tmp_path, monkeypatch):
    """Test Tasseled Cap presets/custom coeffs, histogram matching, and SciPy guard."""
    sensors = st.get_tasseled_cap_sensors()
    assert {
        "landsat5_tm",
        "landsat7_etm",
        "landsat8_oli",
        "landsat9_oli",
        "modis_nbar",
        "sentinel2_msi",
    }.issubset(set(sensors))

    in_6b = np.full((6, 4, 4), 1000.0, dtype=np.float32)
    in_6b[:, 0, 0] = 0.0
    tct_in = str(tmp_path / "s2_in.tif")
    _write_raster(tct_in, in_6b, nodata=0.0)

    tct_out = str(tmp_path / "s2_tct.tif")
    st.calc_tasseled_cap(
        tct_in,
        tct_out,
        sensor="sentinel2_msi",
        gain=10000.0,
        out_no_data_val=-9999.0,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(tct_out) as src:
        arr = src.read()
        assert list(src.descriptions) == ["Brightness", "Greenness", "Wetness"]
        expected_bright = 0.1 * (
            0.3510 + 0.3813 + 0.3437 + 0.7196 + 0.2396 + 0.1949
        )
        assert arr[0, 1, 1] == pytest.approx(expected_bright, abs=1e-5)
        assert arr[0, 0, 0] == pytest.approx(-9999.0)

    custom_df = pd.DataFrame({"AxisA": [1.0, 0.0], "AxisB": [0.0, 2.0]})
    custom_out = str(tmp_path / "custom_tct.tif")
    st.calc_tasseled_cap(
        tct_in,
        custom_out,
        bands=[1, 4],
        coeffs=custom_df,
        gain=1000.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(custom_out) as src:
        c_arr = src.read()
        assert list(src.descriptions) == ["AxisA", "AxisB"]
        assert c_arr[0, 1, 1] == pytest.approx(1.0)
        assert c_arr[1, 1, 1] == pytest.approx(2.0)

    with pytest.raises(rsgispy.RSGISPyException):
        st.calc_tasseled_cap(tct_in, str(tmp_path / "bad_s.tif"), sensor="unknown")

    rng = np.random.default_rng(42)
    src_data = rng.uniform(0.0, 50.0, size=(2, 24, 24)).astype(np.float32)
    src_data[:, 0, 0] = -9999.0
    ref_data = rng.normal(200.0, 20.0, size=(2, 30, 30)).astype(np.float32)

    hm_in = str(tmp_path / "hm_in.tif")
    hm_ref = str(tmp_path / "hm_ref.tif")
    hm_out = str(tmp_path / "hm_out.tif")
    _write_raster(hm_in, src_data, nodata=-9999.0)
    _write_raster(hm_ref, ref_data, nodata=-9999.0)

    img_proc.set_default_block_size(10)
    try:
        st.histogram_matching(
            hm_in,
            hm_ref,
            hm_out,
            n_bins=1024,
            calc_stats=True,
            use_pbar=False,
        )
    finally:
        img_proc.reset_default_block_size()

    with rasterio.open(hm_out) as src:
        matched = src.read()
        assert matched[0, 0, 0] == pytest.approx(-9999.0)
        valid_b1 = matched[0].ravel()
        valid_b1 = valid_b1[valid_b1 != -9999.0]
        assert float(np.mean(valid_b1)) == pytest.approx(
            float(np.mean(ref_data[0])), abs=2.5
        )
        assert float(np.std(valid_b1)) == pytest.approx(
            float(np.std(ref_data[0])), abs=2.5
        )

    monkeypatch.setattr(sp, "HAVE_SCIPY", False)
    with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
        sp.spectral_smoothing(tct_in, str(tmp_path / "no_scipy.tif"), tct_in, 1)


def test_calc_area_under_curve(tmp_path):
    """Test rsgispy.imagespectral.indices.calc_area_under_curve."""
    assert inspect.signature(si.calc_area_under_curve).parameters["gdalformat"].default == "GTIFF"
    in_p = tmp_path / "auc_in.tif"
    out_p = tmp_path / "auc_out.tif"
    arr = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    _write_raster(in_p, arr)

    # The old rectangle sum, sum(band_i * band_widths[i]), as in RSGISLib.
    si.calc_area_under_curve(
        str(in_p), str(out_p), [0.5, 2.0], method="rectangle", use_pbar=False
    )
    with rasterio.open(out_p) as src:
        np.testing.assert_allclose(
            src.read(1), np.array([[4.5, 5.0], [5.5, 6.0]], dtype=np.float32)
        )

    with pytest.raises(rsgispy.RSGISPyException):
        si.calc_area_under_curve(str(in_p), str(out_p), None, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        si.calc_area_under_curve(str(in_p), str(out_p), [1.0], use_pbar=False)


def test_calc_area_under_curve_trapezoid_and_nodata(tmp_path):
    """The default trapezoid rule between band centres, with nodata handling."""
    in_p = tmp_path / "auc_tz.tif"
    out_p = tmp_path / "auc_tz_out.tif"
    arr = np.array(
        [
            [[1.0, 2.0, -99.0]],
            [[3.0, 2.0, 5.0]],
            [[5.0, 2.0, 7.0]],
        ],
        dtype=np.float32,
    )
    _write_raster(in_p, arr, nodata=-99.0)
    # Widths 10, 20, 10: band centres at 5, 20, 35.
    si.calc_area_under_curve(str(in_p), str(out_p), [10.0, 20.0, 10.0],
                             use_pbar=False)
    centres = [5.0, 20.0, 35.0]
    with rasterio.open(out_p) as src:
        assert src.nodata == -99.0
        out = src.read(1)
    np.testing.assert_allclose(out[0, 0], np.trapezoid([1.0, 3.0, 5.0], centres))
    np.testing.assert_allclose(out[0, 1], np.trapezoid([2.0, 2.0, 2.0], centres))
    assert out[0, 2] == -99.0  # one band is nodata

    # Explicit band centres (e.g. wavelengths) and an explicit no_data_val.
    out_c = tmp_path / "auc_centres.tif"
    si.calc_area_under_curve(str(in_p), str(out_c), None,
                             band_centres=[450.0, 550.0, 650.0],
                             no_data_val=-1.0, use_pbar=False)
    with rasterio.open(out_c) as src:
        assert src.nodata == -1.0
        np.testing.assert_allclose(
            src.read(1)[0, 0], np.trapezoid([1.0, 3.0, 5.0], [450.0, 550.0, 650.0])
        )
    with pytest.raises(rsgispy.RSGISPyException):
        si.calc_area_under_curve(str(in_p), str(out_c), None, use_pbar=False)
    with pytest.raises(rsgispy.RSGISPyException):
        si.calc_area_under_curve(str(in_p), str(out_c), [1.0, 1.0, 1.0],
                                 method="simpson", use_pbar=False)


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


def test_calc_ppi_tiled_streaming_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(7)
    cube = rng.uniform(0.2, 0.8, size=(3, 16, 16)).astype(np.float32)
    # Plant extreme pure endmembers at opposite corners
    cube[:, 0, 0] = [1.0, 0.0, 0.0]
    cube[:, 15, 15] = [0.0, 1.0, 0.0]
    in_img = _write_geotiff(tmp_path / "ppi_in.tif", cube)

    out1 = str(tmp_path / "ppi1.tif")
    out2 = str(tmp_path / "ppi2.tif")
    endm.calc_ppi(
        in_img, out1, niters=200, rnd_seed=99, calc_stats=False, use_pbar=False
    )
    endm.calc_ppi(
        in_img, out2, niters=200, rnd_seed=99, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out1) as d1, rasterio.open(out2) as d2:
        a1 = d1.read(1)
        a2 = d2.read(1)
        assert a1.dtype == np.uint16
        np.testing.assert_array_equal(a1, a2)
        assert a1[0, 0] > 0
        assert a1[15, 15] > 0


def test_whiten_image_filters_nan_and_nodata(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(11)
    n = 16 * 16
    base = rng.normal(size=(n, 3))
    mix = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]])
    cube = (base @ mix + np.array([10.0, 20.0, 30.0])).T.reshape(3, 16, 16)
    cube = cube.astype(np.float32)
    cube[:, 0, 0] = np.nan
    cube[:, 0, 1] = -9999.0

    msk = np.ones((16, 16), dtype=np.uint8)
    msk[15, 15] = 0

    in_img = _write_geotiff(tmp_path / "w_in.tif", cube, nodata=-9999.0)
    msk_img = _write_geotiff(tmp_path / "w_msk.tif", msk)
    out_img = str(tmp_path / "w_out.tif")

    spec_tr.whiten_image(in_img, out_img, msk_img, 1, use_pbar=False)
    with rasterio.open(out_img) as dst:
        out = dst.read()
        assert np.all(out[:, 0, 0] == 0.0)
        assert np.all(out[:, 0, 1] == 0.0)
        assert np.all(out[:, 15, 15] == 0.0)
        valid = np.ones((16, 16), dtype=bool)
        valid[0, 0] = False
        valid[0, 1] = False
        valid[15, 15] = False
        vecs = out[:, valid]
        cov = np.cov(vecs)
        np.testing.assert_allclose(cov, np.eye(3), atol=5e-2)


def test_summarise_multi_endmember_unmixing_nodata_alignment(tmp_path):
    em_mat = np.array([[0.8, 0.2, 0.1], [0.1, 0.7, 0.9]], dtype=np.float32)
    em_df = pd.DataFrame(em_mat.T, columns=["EM_A", "EM_B"])

    # Pixel (0,0) is all-zero (invalid); remaining pixels are valid combinations
    orig = np.zeros((3, 4, 4), dtype=np.float32)
    for b in range(3):
        orig[b, :, :] = 0.6 * em_mat[0, b] + 0.4 * em_mat[1, b]
    orig[:, 0, 0] = 0.0

    abund = np.zeros((2, 4, 4), dtype=np.float32)
    abund[0, :, :] = 0.6
    abund[1, :, :] = 0.4

    orig_img = _write_geotiff(tmp_path / "orig.tif", orig)
    abund_img = _write_geotiff(tmp_path / "abund1.tif", abund)
    out_unmix = str(tmp_path / "sum_unmix.tif")
    out_ref = str(tmp_path / "sum_ref.tif")

    ds_info = endm.ImageEndmemberInfo(
        in_unmix_img=abund_img,
        endmembers=em_df,
        endmember_names=["EM_A", "EM_B"],
    )
    unmix.summarise_multi_endmember_linear_unmixing(
        input_img=orig_img,
        out_unmix_img=out_unmix,
        out_ref_img=out_ref,
        in_unmixed_datasets=[ds_info],
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_unmix) as dst_u, rasterio.open(out_ref) as dst_r:
        assert dst_u.nodata == 999.0
        assert dst_r.nodata == 0.0
        u_arr = dst_u.read()
        r_arr = dst_r.read(1)
        assert np.all(u_arr[:, 0, 0] == 999.0)
        assert r_arr[0, 0] == 0
        assert np.allclose(u_arr[:, 1, 1], [0.6, 0.4], atol=1e-4)
        assert r_arr[1, 1] == 1
