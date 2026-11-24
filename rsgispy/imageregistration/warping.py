"""
GCP embedding, GCP/polynomial/shift image warping, and end-to-end tie-point
co-registration for :mod:`rsgispy.imageregistration`.
"""

import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.control import GroundControlPoint

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageregistration import (
    OUTLIER_METHOD_RANSAC,
    TIE_METHOD_PHASE_CORR,
    TRANSFORM_AFFINE,
    TRANSFORM_POLY1,
    TRANSFORM_POLY2,
    TRANSFORM_POLY3,
    TRANSFORM_SHIFT,
    TRANSFORM_SIMILARITY,
    TRANSFORM_TPS,
)
from rsgispy.imageregistration._helpers import (
    copy_raster_with_updates,
    df_to_gcps,
    load_tie_points_table,
    write_tie_points_file,
)
from rsgispy.imageregistration.shift import apply_offset_to_image
from rsgispy.imageregistration.tiepoints import (
    evaluate_geometric_model,
    filter_tie_points_outliers,
    fit_geometric_model,
    generate_tie_points,
)
from rsgispy.imageutils.stats import pop_img_stats
import rsgispy.imageutils.warp
from rsgispy.vectorprocess import VectorInput, read_vec_gdf

__all__ = [
    "gcp_to_gdal",
    "add_vec_pts_as_gcps_to_img",
    "warp_with_gcps",
    "warp_with_tie_points",
    "coregister_images",
]


def _copy_raster_with_gcps(
    in_path: str,
    out_path: str,
    gcps: Any,
    gcp_crs: Any,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
) -> Optional[float]:
    """Copy ``in_path`` to ``out_path`` attaching ``gcps`` and ``gcp_crs``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return copy_raster_with_updates(
        in_path,
        out_path,
        gdalformat=gdalformat,
        datatype=datatype,
        gcps=(gcps, gcp_crs),
        no_data_val=no_data_val,
    )


def gcp_to_gdal(
    input_img: str,
    output_img: str,
    in_gcp_file: Union[str, os.PathLike, pd.DataFrame],
    gcp_epsg: Optional[int] = None,
    *,
    gcp_crs: Optional[Any] = None,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Attach tie points from a GCP file or :class:`pandas.DataFrame` as raster
    Ground Control Points (GCPs) in ``output_img``.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output image containing embedded GCPs.
    :param in_gcp_file: Tie-point file path (RSGIS/ENVI GCP text, CSV, Parquet,
                        or vector file) or a :class:`pandas.DataFrame`.
    :param gcp_epsg: Optional EPSG code for GCP coordinates (defaults to
                     ``input_img`` CRS).
    :param gcp_crs: Optional full CRS (``rasterio.crs.CRS``, ``pyproj.CRS``,
                    WKT, or PROJ string) for GCP coordinates; takes precedence
                    over ``gcp_epsg`` when provided.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param no_data_val: Optional output no-data value (``None`` preserves input
                        no-data value).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar when computing statistics
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageregistration.warping as imreg_warp

        imreg_warp.gcp_to_gdal(
            input_img="float.tif",
            output_img="float_gcps.tif",
            in_gcp_file="tie_points.txt",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input image")

    tie_df = load_tie_points_table(in_gcp_file)
    gcps = df_to_gcps(tie_df)

    with rasterio.open(in_path, "r") as src:
        if gcp_crs is not None:
            eff_gcp_crs = rasterio.crs.CRS.from_user_input(gcp_crs)
        elif gcp_epsg is not None:
            eff_gcp_crs = rasterio.crs.CRS.from_epsg(gcp_epsg)
        else:
            eff_gcp_crs = src.crs

    nd_val = _copy_raster_with_gcps(
        in_path,
        out_path,
        gcps=gcps,
        gcp_crs=eff_gcp_crs,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
    )

    if calc_stats:
        pop_img_stats(
            out_path,
            no_data_val=None if nd_val is None else float(nd_val),
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def add_vec_pts_as_gcps_to_img(
    input_img: str,
    output_img: str,
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    gcp_x_col: str = "x_match",
    gcp_y_col: str = "y_match",
    gcp_z_col: Optional[str] = None,
    gcp_epsg: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Embed Ground Control Points (GCPs) into ``output_img`` from a point vector
    layer whose attributes define target map coordinates.

    If ``input_img`` and ``output_img`` point to the same file, GCPs are
    written in-place.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output raster image.
    :param vec_file: Point vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Optional layer name when ``vec_file`` is a file path.
    :param gcp_x_col: Attribute column containing target GCP X coordinates
                      (Default = ``'x_match'``).
    :param gcp_y_col: Attribute column containing target GCP Y coordinates
                      (Default = ``'y_match'``).
    :param gcp_z_col: Optional attribute column for GCP Z elevation.
    :param gcp_epsg: Optional EPSG code for the target GCP coordinates
                     (defaults to ``input_img`` CRS).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param no_data_val: Optional output no-data value (``None`` preserves input
                        no-data value).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar when computing statistics
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageregistration.warping as imreg_warp

        imreg_warp.add_vec_pts_as_gcps_to_img(
            input_img="float.tif",
            output_img="float_gcps.tif",
            vec_file="gcps.gpkg",
            vec_lyr="gcps",
            gcp_x_col="x_match",
            gcp_y_col="y_match",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input image")

    vec_gdf = read_vec_gdf(vec_file, vec_lyr)
    geom_types = set(vec_gdf.geom_type.dropna().unique())
    if not geom_types or not geom_types.issubset({"Point"}):
        raise rsgispy.RSGISPyException(
            f"Vector layer must contain Point geometries: got {sorted(geom_types)}."
        )
    for col in (gcp_x_col, gcp_y_col):
        if col not in vec_gdf.columns:
            raise rsgispy.RSGISPyException(
                f"GCP coordinate column '{col}' not found in vector layer."
            )
    if gcp_z_col is not None and gcp_z_col not in vec_gdf.columns:
        raise rsgispy.RSGISPyException(
            f"GCP Z coordinate column '{gcp_z_col}' not found in vector layer."
        )

    with rasterio.open(in_path, "r") as src:
        img_transform = src.transform
        img_crs = src.crs
        gcp_crs = (
            rasterio.crs.CRS.from_epsg(gcp_epsg)
            if gcp_epsg is not None
            else img_crs
        )

    pts_series = vec_gdf.geometry
    if vec_gdf.crs is not None and img_crs is not None:
        if pyproj.CRS.from_user_input(vec_gdf.crs) != pyproj.CRS.from_user_input(
            img_crs
        ):
            pts_series = pts_series.to_crs(img_crs)

    gcps = []
    for idx, (pt, (_, row)) in enumerate(zip(pts_series, vec_gdf.iterrows())):
        if pt is None or pt.is_empty:
            continue
        col_f, row_f = ~img_transform @ (float(pt.x), float(pt.y))
        z_val = float(row[gcp_z_col]) if gcp_z_col is not None else 0.0
        gcps.append(
            GroundControlPoint(
                row=float(row_f),
                col=float(col_f),
                x=float(row[gcp_x_col]),
                y=float(row[gcp_y_col]),
                z=z_val,
                id=str(idx + 1),
            )
        )

    if not gcps:
        raise rsgispy.RSGISPyException("No valid point geometries found for GCPs.")

    nd_val = _copy_raster_with_gcps(
        in_path,
        out_path,
        gcps=gcps,
        gcp_crs=gcp_crs,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
    )

    if calc_stats:
        pop_img_stats(
            out_path,
            no_data_val=None if nd_val is None else float(nd_val),
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def warp_with_gcps(
    in_ref_img: str,
    in_process_img: str,
    output_img: str,
    interp_method: int = rsgispy.INTERP_BILINEAR,
    use_tps: bool = False,
    use_poly: bool = True,
    poly_order: int = 3,
    use_multi_threaded: bool = True,
    refine_gcps: Optional[Union[float, Tuple[float, Optional[int]]]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Warp ``in_process_img`` using its embedded GCPs onto the exact CRS, extent,
    and pixel grid of ``in_ref_img``.

    :param in_ref_img: Reference raster image defining output extent and grid.
    :param in_process_img: Floating raster image containing embedded GCPs.
    :param output_img: Path to the output warped image.
    :param interp_method: Resampling method ``rsgispy.INTERP_*`` (Default =
                          :data:`rsgispy.INTERP_BILINEAR`).
    :param use_tps: Use thin-plate spline transformation (Default = ``False``).
    :param use_poly: Use polynomial transformation when ``use_tps=False``
                     (Default = ``True``).
    :param poly_order: Polynomial order ``1``, ``2``, or ``3`` (Default = ``3``).
    :param use_multi_threaded: Enable multi-threaded warping (Default = ``True``).
    :param refine_gcps: Optional ``tolerance`` or ``(tolerance, min_gcps)`` to
                        prune outlier GCPs during warping.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imageregistration.warping as imreg_warp

        imreg_warp.warp_with_gcps(
            in_ref_img="ref.tif",
            in_process_img="float_gcps.tif",
            output_img="registered.tif",
            interp_method=rsgispy.INTERP_BILINEAR,
            poly_order=1,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ref_path = os.fspath(in_ref_img)
    proc_path = os.fspath(in_process_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(ref_path, label="Reference image")
    ensure_file_exists(proc_path, label="Process image")

    with rasterio.open(proc_path, "r") as proc_src:
        proc_gcps, _ = proc_src.gcps
        if not proc_gcps:
            raise rsgispy.RSGISPyException(
                f"Input image '{proc_path}' does not contain any GCPs."
            )

    with rasterio.open(ref_path, "r") as ref_src:
        b = ref_src.bounds
        ref_extent = (float(b.left), float(b.bottom), float(b.right), float(b.top))
        ref_size = (ref_src.width, ref_src.height)
        ref_crs = ref_src.crs

    order_val: Optional[int] = None
    if not use_tps:
        order_val = int(poly_order) if use_poly else 1
        if order_val not in (1, 2, 3):
            raise rsgispy.RSGISPyException(
                f"poly_order must be 1, 2, or 3 (got {poly_order})."
            )

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rsgispy.imageutils.warp.gdal_warp(
        input_img=proc_path,
        output_img=out_path,
        t_srs=ref_crs,
        target_extent=ref_extent,
        target_size=ref_size,
        interp_method=interp_method,
        gdalformat=gdalformat,
        datatype=datatype,
        dst_nodata=no_data_val,
        use_multi_threaded=use_multi_threaded,
        tps=use_tps,
        order=order_val,
        refine_gcps=refine_gcps,
        use_pbar=use_pbar,
    )

    if calc_stats:
        with rasterio.open(out_path, "r") as out_src:
            nd_val = out_src.nodata
        pop_img_stats(
            out_path,
            no_data_val=None if nd_val is None else float(nd_val),
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def warp_with_tie_points(
    in_ref_img: str,
    in_float_img: str,
    output_img: str,
    tie_points: Union[pd.DataFrame, gpd.GeoDataFrame, str, os.PathLike],
    interp_method: int = rsgispy.INTERP_BILINEAR,
    transform_type: str = TRANSFORM_AFFINE,
    poly_order: int = 1,
    use_tps: bool = False,
    match_ref_grid: bool = True,
    refine_gcps: Optional[Union[float, Tuple[float, Optional[int]]]] = None,
    use_multi_threaded: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Warp ``in_float_img`` using tie points onto ``in_ref_img`` via linear shift,
    polynomial, or thin-plate spline transformation.

    :param in_ref_img: Path to the reference raster image.
    :param in_float_img: Path to the floating raster image.
    :param output_img: Path to the output warped raster image.
    :param tie_points: Tie-point :class:`pandas.DataFrame`,
                       :class:`geopandas.GeoDataFrame`, or file path.
    :param interp_method: Resampling method ``rsgispy.INTERP_*`` (Default =
                          :data:`rsgispy.INTERP_BILINEAR`).
    :param transform_type: Transformation type constant ``TRANSFORM_*``
                           (:data:`TRANSFORM_SHIFT`,
                           :data:`TRANSFORM_SIMILARITY`,
                           :data:`TRANSFORM_AFFINE` / :data:`TRANSFORM_POLY1`,
                           :data:`TRANSFORM_POLY2`, :data:`TRANSFORM_POLY3`,
                           :data:`TRANSFORM_TPS`, or ``'polynomial'``;
                           Default = :data:`TRANSFORM_AFFINE`).
    :param poly_order: Polynomial order when ``transform_type='polynomial'``
                       (Default = ``1``).
    :param use_tps: Force thin-plate spline warping (Default = ``False``).
    :param match_ref_grid: Resample output to match ``in_ref_img`` extent and
                           dimensions (Default = ``True``). For
                           :data:`TRANSFORM_SHIFT` with
                           ``match_ref_grid=False``, only updates the
                           geotransform header without resampling.
    :param refine_gcps: Optional GCP outlier tolerance passed to ``gdal_warp``.
    :param use_multi_threaded: Enable multi-threaded warping (Default = ``True``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar during processing
                     (Default = ``True``).
    :return: Dictionary containing transformation summary and fit RMSE.

    .. code:: python

        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.warping as imreg_warp

        summary = imreg_warp.warp_with_tie_points(
            in_ref_img="ref.tif",
            in_float_img="float.tif",
            output_img="registered.tif",
            tie_points=inliers_df,
            transform_type=imreg.TRANSFORM_AFFINE,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ref_path = os.fspath(in_ref_img)
    flt_path = os.fspath(in_float_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(ref_path, label="Reference image")
    ensure_file_exists(flt_path, label="Floating image")

    tie_df = load_tie_points_table(tie_points)

    with rasterio.open(ref_path, "r") as ref_src, rasterio.open(
        flt_path, "r"
    ) as flt_src:
        ref_crs = ref_src.crs
        flt_crs = flt_src.crs
        b = ref_src.bounds
        ref_extent = (float(b.left), float(b.bottom), float(b.right), float(b.top))
        ref_size = (ref_src.width, ref_src.height)
        if not {"ref_x", "ref_y"}.issubset(tie_df.columns) and {
            "ref_pxl_x",
            "ref_pxl_y",
        }.issubset(tie_df.columns):
            rx, ry = ref_src.transform @ (
                tie_df["ref_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["ref_pxl_y"].to_numpy(dtype=np.float64),
            )
            tie_df["ref_x"] = rx
            tie_df["ref_y"] = ry
        if not {"flt_x", "flt_y"}.issubset(tie_df.columns) and {
            "flt_pxl_x",
            "flt_pxl_y",
        }.issubset(tie_df.columns):
            fx, fy = flt_src.transform @ (
                tie_df["flt_pxl_x"].to_numpy(dtype=np.float64),
                tie_df["flt_pxl_y"].to_numpy(dtype=np.float64),
            )
            tie_df["flt_x"] = fx
            tie_df["flt_y"] = fy

    t_key = transform_type.strip().lower()
    if t_key not in (
        TRANSFORM_SHIFT,
        "translation",
        TRANSFORM_SIMILARITY,
        TRANSFORM_AFFINE,
        TRANSFORM_POLY1,
        TRANSFORM_POLY2,
        TRANSFORM_POLY3,
        "polynomial",
        TRANSFORM_TPS,
    ):
        raise rsgispy.RSGISPyException(
            f"Unsupported transform_type '{transform_type}'."
        )

    eff_tps = use_tps or t_key == TRANSFORM_TPS
    same_crs = True
    if ref_crs is not None and flt_crs is not None:
        same_crs = pyproj.CRS.from_user_input(
            ref_crs
        ) == pyproj.CRS.from_user_input(flt_crs)
    if (
        not same_crs
        and not eff_tps
        and t_key in (TRANSFORM_SHIFT, "translation", TRANSFORM_SIMILARITY)
    ):
        raise rsgispy.RSGISPyException(
            f"transform_type='{transform_type}' requires matching reference "
            f"and floating image CRSs (got ref={ref_crs} vs float={flt_crs}). "
            "Reproject the floating image first or use an affine, polynomial, "
            "or TPS transform."
        )

    if t_key in (TRANSFORM_AFFINE, TRANSFORM_POLY1):
        eff_order = 1
        fit_key = TRANSFORM_AFFINE
    elif t_key == TRANSFORM_SIMILARITY:
        eff_order = 1
        fit_key = TRANSFORM_SIMILARITY
    elif t_key == TRANSFORM_POLY2:
        eff_order = 2
        fit_key = TRANSFORM_POLY2
    elif t_key == TRANSFORM_POLY3:
        eff_order = 3
        fit_key = TRANSFORM_POLY3
    elif t_key == "polynomial":
        eff_order = int(poly_order)
        fit_key = f"poly{eff_order}" if eff_order in (1, 2, 3) else TRANSFORM_AFFINE
    elif t_key in (TRANSFORM_SHIFT, "translation"):
        eff_order = 1
        fit_key = TRANSFORM_SHIFT
    else:
        eff_order = 1
        fit_key = TRANSFORM_AFFINE

    rmse = float("nan")
    fitted_model: Optional[Dict[str, Any]] = None
    if {"flt_x", "flt_y", "ref_x", "ref_y"}.issubset(tie_df.columns):
        sx = tie_df["flt_x"].to_numpy(dtype=np.float64)
        sy = tie_df["flt_y"].to_numpy(dtype=np.float64)
        dx = tie_df["ref_x"].to_numpy(dtype=np.float64)
        dy = tie_df["ref_y"].to_numpy(dtype=np.float64)
        fitted_model = fit_geometric_model(sx, sy, dx, dy, model=fit_key)
        px, py = evaluate_geometric_model(fitted_model, sx, sy)
        rmse = float(np.sqrt(np.mean((px - dx) ** 2 + (py - dy) ** 2)))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if t_key in (TRANSFORM_SHIFT, "translation") and not eff_tps:
        tx = float(np.mean(tie_df["ref_x"] - tie_df["flt_x"]))
        ty = float(np.mean(tie_df["ref_y"] - tie_df["flt_y"]))
        if not match_ref_grid:
            apply_offset_to_image(
                flt_path,
                out_path,
                x_offset=tx,
                y_offset=ty,
                in_pxl_units=False,
                gdalformat=gdalformat,
                datatype=datatype,
                no_data_val=no_data_val,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                shifted_tmp = os.path.join(tmp_dir, "shifted.tif")
                apply_offset_to_image(
                    flt_path,
                    shifted_tmp,
                    x_offset=tx,
                    y_offset=ty,
                    in_pxl_units=False,
                    gdalformat="GTIFF",
                    datatype=datatype,
                    calc_stats=False,
                    use_pbar=False,
                )
                rsgispy.imageutils.warp.gdal_warp(
                    input_img=shifted_tmp,
                    output_img=out_path,
                    t_srs=ref_crs,
                    target_extent=ref_extent,
                    target_size=ref_size,
                    interp_method=interp_method,
                    gdalformat=gdalformat,
                    datatype=datatype,
                    dst_nodata=no_data_val,
                    use_multi_threaded=use_multi_threaded,
                    use_pbar=use_pbar,
                )
            if calc_stats:
                with rasterio.open(out_path, "r") as out_src:
                    nd_val = out_src.nodata
                pop_img_stats(
                    out_path,
                    no_data_val=None if nd_val is None else float(nd_val),
                    calc_pyramids=True,
                    use_pbar=use_pbar,
                )
        return {
            "transform_type": TRANSFORM_SHIFT,
            "n_gcps": int(len(tie_df)),
            "x_offset_map": tx,
            "y_offset_map": ty,
            "rmse": rmse,
            "output_img": out_path,
        }

    gcp_df = tie_df
    if t_key == TRANSFORM_SIMILARITY and not eff_tps and fitted_model is not None:
        with rasterio.open(flt_path, "r") as flt_src:
            c_px_x = np.array(
                [0.0, float(flt_src.width), 0.0, float(flt_src.width)],
                dtype=np.float64,
            )
            c_px_y = np.array(
                [0.0, 0.0, float(flt_src.height), float(flt_src.height)],
                dtype=np.float64,
            )
            c_flt_x, c_flt_y = flt_src.transform @ (c_px_x, c_px_y)
        c_ref_x, c_ref_y = evaluate_geometric_model(fitted_model, c_flt_x, c_flt_y)
        gcp_df = pd.DataFrame(
            {
                "ref_x": c_ref_x,
                "ref_y": c_ref_y,
                "flt_pxl_x": c_px_x,
                "flt_pxl_y": c_px_y,
            }
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        gcp_tmp = os.path.join(tmp_dir, "with_gcps.tif")
        gcp_to_gdal(
            flt_path,
            gcp_tmp,
            gcp_df,
            gcp_crs=ref_crs,
            gdalformat="GTIFF",
            datatype=datatype,
            calc_stats=False,
            use_pbar=False,
        )
        if match_ref_grid:
            warp_with_gcps(
                in_ref_img=ref_path,
                in_process_img=gcp_tmp,
                output_img=out_path,
                interp_method=interp_method,
                use_tps=eff_tps,
                use_poly=not eff_tps,
                poly_order=eff_order,
                use_multi_threaded=use_multi_threaded,
                refine_gcps=refine_gcps,
                gdalformat=gdalformat,
                datatype=datatype,
                no_data_val=no_data_val,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )
        else:
            rsgispy.imageutils.warp.gdal_warp(
                input_img=gcp_tmp,
                output_img=out_path,
                t_srs=ref_crs,
                interp_method=interp_method,
                gdalformat=gdalformat,
                datatype=datatype,
                dst_nodata=no_data_val,
                use_multi_threaded=use_multi_threaded,
                tps=eff_tps,
                order=None if eff_tps else eff_order,
                refine_gcps=refine_gcps,
                use_pbar=use_pbar,
            )
            if calc_stats:
                with rasterio.open(out_path, "r") as out_src:
                    nd_val = out_src.nodata
                pop_img_stats(
                    out_path,
                    no_data_val=None if nd_val is None else float(nd_val),
                    calc_pyramids=True,
                    use_pbar=use_pbar,
                )

    out_type = (
        TRANSFORM_TPS
        if eff_tps
        else (TRANSFORM_SIMILARITY if t_key == TRANSFORM_SIMILARITY else f"poly{eff_order}")
    )
    return {
        "transform_type": out_type,
        "n_gcps": int(len(tie_df)),
        "rmse": rmse,
        "output_img": out_path,
    }


def coregister_images(
    in_ref_img: str,
    in_float_img: str,
    output_img: str,
    tie_method: str = TIE_METHOD_PHASE_CORR,
    outlier_method: str = OUTLIER_METHOD_RANSAC,
    transform_type: str = TRANSFORM_AFFINE,
    ref_band: int = 1,
    flt_band: int = 1,
    pixel_gap: int = 50,
    win_size: int = 64,
    search_area: int = 15,
    threshold: Optional[float] = None,
    sd_ref_thres: float = 1.0,
    sd_flt_thres: float = 1.0,
    sub_pxl_res: int = 4,
    metric_type: Union[int, str] = rsgispy.METRIC_CORRELATION,
    residual_thres: float = 2.0,
    interp_method: int = rsgispy.INTERP_BILINEAR,
    match_ref_grid: bool = True,
    out_gcp_file: Optional[Union[str, os.PathLike]] = None,
    rnd_seed: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end automated tie-point co-registration pipeline: generate tie
    points, filter outliers, and warp ``in_float_img`` onto ``in_ref_img``.

    :param in_ref_img: Path to the reference raster image.
    :param in_float_img: Path to the floating raster image.
    :param output_img: Path to the output co-registered raster image.
    :param tie_method: Tie-point generation method constant ``TIE_METHOD_*``
                       (:data:`TIE_METHOD_PHASE_CORR` or
                       :data:`TIE_METHOD_TEMPLATE_MATCH`; Default =
                       :data:`TIE_METHOD_PHASE_CORR`).
    :param outlier_method: Outlier filter method constant ``OUTLIER_METHOD_*``
                           (:data:`OUTLIER_METHOD_RANSAC`,
                           :data:`OUTLIER_METHOD_RESIDUAL`, or
                           :data:`OUTLIER_METHOD_MAD`; Default =
                           :data:`OUTLIER_METHOD_RANSAC`).
    :param transform_type: Geometric warp model constant ``TRANSFORM_*``
                           (:data:`TRANSFORM_SHIFT`,
                           :data:`TRANSFORM_SIMILARITY`,
                           :data:`TRANSFORM_AFFINE`, :data:`TRANSFORM_POLY1`,
                           :data:`TRANSFORM_POLY2`, :data:`TRANSFORM_POLY3`, or
                           :data:`TRANSFORM_TPS`; Default =
                           :data:`TRANSFORM_AFFINE`).
    :param ref_band: 1-based band index in ``in_ref_img`` (Default = ``1``).
    :param flt_band: 1-based band index in ``in_float_img`` (Default = ``1``).
    :param pixel_gap: Grid spacing in pixels between candidate tie points.
    :param win_size: Window size in pixels around each candidate tie point.
    :param search_area: Search radius in pixels.
    :param threshold: Minimum (or maximum for distance metrics) match score.
    :param sd_ref_thres: Minimum reference patch standard deviation.
    :param sd_flt_thres: Minimum floating patch standard deviation.
    :param sub_pxl_res: Sub-pixel refinement factor.
    :param metric_type: Similarity metric for :data:`TIE_METHOD_TEMPLATE_MATCH`
                        (Default = :data:`rsgispy.METRIC_CORRELATION`).
    :param residual_thres: Maximum map-unit residual for inlier tie points.
    :param interp_method: Resampling method ``rsgispy.INTERP_*``.
    :param match_ref_grid: Resample onto ``in_ref_img`` grid (Default = ``True``).
    :param out_gcp_file: Optional path to save filtered inlier tie points.
    :param rnd_seed: Optional RNG seed for reproducible RANSAC sampling.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type (``None``
                     preserves input pixel dtype).
    :param no_data_val: Optional output no-data value.
    :param calc_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param use_pbar: Display tqdm progress bar (Default = ``True``).
    :return: Dictionary containing tie-point counts, RMSE, and output info.

    .. code:: python

        import rsgispy.imageregistration as imreg
        import rsgispy.imageregistration.warping as imreg_warp

        result = imreg_warp.coregister_images(
            in_ref_img="ref.tif",
            in_float_img="float.tif",
            output_img="coregistered.tif",
            tie_method=imreg.TIE_METHOD_PHASE_CORR,
            outlier_method=imreg.OUTLIER_METHOD_RANSAC,
            transform_type=imreg.TRANSFORM_AFFINE,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    tie_df = generate_tie_points(
        in_ref_img=in_ref_img,
        in_float_img=in_float_img,
        out_gcp_file=None,
        method=tie_method,
        ref_band=ref_band,
        flt_band=flt_band,
        pixel_gap=pixel_gap,
        win_size=win_size,
        search_area=search_area,
        threshold=threshold,
        sd_ref_thres=sd_ref_thres,
        sd_flt_thres=sd_flt_thres,
        sub_pxl_res=sub_pxl_res,
        metric_type=metric_type,
        use_pbar=use_pbar,
    )
    if tie_df is None or len(tie_df) == 0:
        raise rsgispy.RSGISPyException(
            "No tie points were matched between reference and floating images."
        )

    t_norm = transform_type.strip().lower()
    if t_norm in (TRANSFORM_SHIFT, "translation"):
        filter_model = TRANSFORM_SHIFT
    elif t_norm == TRANSFORM_TPS:
        filter_model = TRANSFORM_AFFINE
    else:
        filter_model = transform_type

    inliers_df, diag = filter_tie_points_outliers(
        tie_points=tie_df,
        out_gcp_file=None,
        method=outlier_method,
        model=filter_model,
        residual_thres=residual_thres,
        rnd_seed=rnd_seed,
        return_model=True,
    )

    if out_gcp_file is not None:
        with rasterio.open(in_ref_img, "r") as ref_src:
            ref_crs = ref_src.crs
        write_tie_points_file(inliers_df, out_gcp_file, crs=ref_crs)

    warp_info = warp_with_tie_points(
        in_ref_img=in_ref_img,
        in_float_img=in_float_img,
        output_img=output_img,
        tie_points=inliers_df,
        interp_method=interp_method,
        transform_type=transform_type,
        match_ref_grid=match_ref_grid,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    return {
        "n_tie_points": diag["n_input"],
        "n_inliers": diag["n_inliers"],
        "n_outliers": diag["n_outliers"],
        "rmse": diag["rmse"],
        "max_residual": diag["max_residual"],
        "transform_type": warp_info["transform_type"],
        "output_img": warp_info["output_img"],
    }
