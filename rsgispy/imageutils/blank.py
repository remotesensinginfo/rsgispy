"""
The imageutils.blank module provides functions for creating blank, copied,
buffered, and bbox-defined raster datasets using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
    pop_stats_on_close,
)
from rsgispy.imageutils._helpers import (
    normalise_creation_options,
    parse_epsg_crs,
    rsgis_dtype_from_np_dtype,
)
from rsgispy.imageutils.metadata import get_file_img_extension
from rsgispy.tools.geometry import find_extent_on_whole_num_grid
from rsgispy.vectorprocess import (
    VectorInput,
    read_vec_gdf,
    validate_columns_exist,
)

__all__ = [
    "create_blank_img",
    "create_copy_img",
    "create_blank_buf_img_from_ref_img",
    "create_blank_img_from_bbox",
    "create_blank_ref_imgs",
]


def _write_blank_grid_img(
    output_img: str,
    n_bands: int,
    width: int,
    height: int,
    transform: Affine,
    crs_obj: Any,
    pxl_val: float,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """Create a blank raster on an arbitrary CRS and Affine transform."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_bands < 1 or width < 1 or height < 1:
        raise rsgispy.RSGISPyException(
            "n_bands, width, and height must all be >= 1."
        )

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    grid = IntersectionGrid(
        crs=crs_obj,
        transform=transform,
        width=int(width),
        height=int(height),
        offsets={},
    )
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        dst = open_output_dataset(
            stack,
            output_img,
            gdalformat,
            out_rio_dtype,
            grid,
            int(n_bands),
            False,
            norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )

        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            block = np.full(
                (int(n_bands), int(win.height), int(win.width)),
                pxl_val,
                dtype=out_np_dtype,
            )
            dst.write(block, window=win)


def _bbox_to_grid_params(
    bbox: Tuple[float, float, float, float],
    out_img_res: float,
    snap_to_grid: bool = False,
) -> Tuple[int, int, float, float]:
    """
    Convert ``bbox`` ``(x_min, x_max, y_min, y_max)`` and ``out_img_res`` into
    ``(width, height, tl_x, tl_y)``.
    """
    res_val = float(out_img_res)
    if not math.isfinite(res_val) or res_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"out_img_res must be a positive finite number, got {out_img_res!r}."
        )
    if snap_to_grid:
        bbox = find_extent_on_whole_num_grid(bbox, res_val, full_contain=True)

    x_min, x_max, y_min, y_max = (float(v) for v in bbox[:4])
    width_coord = x_max - x_min
    height_coord = y_max - y_min
    if width_coord <= 0.0 or height_coord <= 0.0:
        raise rsgispy.RSGISPyException(
            f"Invalid bounding box {bbox!r}: width and height must be positive."
        )

    width_rem = abs(width_coord / res_val - math.floor(width_coord / res_val))
    height_rem = abs(
        height_coord / res_val - math.floor(height_coord / res_val)
    )

    width = (
        int(math.floor(width_coord / res_val))
        if width_rem < 0.0001
        else int(math.ceil(width_coord / res_val))
    )
    height = (
        int(math.floor(height_coord / res_val))
        if height_rem < 0.0001
        else int(math.ceil(height_coord / res_val))
    )
    return max(1, width), max(1, height), x_min, y_max


def create_blank_img(
    output_img: str,
    n_bands: int,
    width: int,
    height: int,
    tl_x: float,
    tl_y: float,
    res_x: float,
    res_y: float,
    pxl_val: float,
    epsg_code: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a new blank raster image with the specified dimensions, resolution,
    EPSG coordinate system, and initial pixel value using ``rasterio``.

    If ``creation_opts`` is ``None`` or empty, any format creation options
    configured via environment variables (e.g.,
    :func:`set_env_vars_deflate_gtiff_outs`) are automatically applied.

    :param output_img: Path to the output image file.
    :param n_bands: Number of image bands in the output image.
    :param width: Width of the output image in pixels.
    :param height: Height of the output image in pixels.
    :param tl_x: Top-left pixel X coordinate (eastings).
    :param tl_y: Top-left pixel Y coordinate (northings).
    :param res_x: Pixel resolution along the X axis.
    :param res_y: Pixel resolution along the Y axis (positive or negative).
    :param pxl_val: Initial pixel value for all bands.
    :param epsg_code: EPSG integer code defining the output projection (e.g., 32630).
    :param gdalformat: Output GDAL format name (e.g., ``'GTIFF'``, ``'KEA'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional no-data value to set on the output dataset.
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or list of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_img(
            "blank.tif", 1, 100, 100, 500000.0, 5600000.0, 10.0, -10.0,
            0.0, 32630, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    crs_obj = parse_epsg_crs(epsg_code)
    transform = Affine(
        abs(float(res_x)),
        0.0,
        float(tl_x),
        0.0,
        -abs(float(res_y)),
        float(tl_y),
    )
    _write_blank_grid_img(
        output_img=output_img,
        n_bands=n_bands,
        width=width,
        height=height,
        transform=transform,
        crs_obj=crs_obj,
        pxl_val=pxl_val,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def create_copy_img(
    input_img: str,
    output_img: str,
    n_bands: int = 1,
    pxl_val: float = 0.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a new blank raster image matching the spatial extent, resolution,
    dimensions, transform, and CRS of ``input_img``.

    :param input_img: Path to the reference input image.
    :param output_img: Path to the output image file.
    :param n_bands: Number of image bands in the output image (Default = ``1``).
    :param pxl_val: Initial pixel value for all bands (Default = ``0.0``).
    :param gdalformat: Output GDAL format name (e.g., ``'GTIFF'``, ``'KEA'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param no_data_val: Optional no-data value to set on the output dataset.
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or list of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_copy_img(
            "in_img.tif", "out_img.tif", 1, 0,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs_obj = src.crs
        eff_dtype = (
            rsgis_dtype_from_np_dtype(src.dtypes[0])
            if datatype is None
            else datatype
        )

    _write_blank_grid_img(
        output_img=output_img,
        n_bands=n_bands,
        width=width,
        height=height,
        transform=transform,
        crs_obj=crs_obj,
        pxl_val=pxl_val,
        gdalformat=gdalformat,
        datatype=eff_dtype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def create_blank_buf_img_from_ref_img(
    input_img: str,
    output_img: str,
    buf_pxl_ext: Optional[int] = None,
    buf_spt_ext: Optional[float] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a new blank image based on ``input_img`` buffered on all sides by
    either ``buf_pxl_ext`` pixels or ``buf_spt_ext`` spatial units.

    :param input_img: Path to the reference input image.
    :param output_img: Path to the output image file.
    :param buf_pxl_ext: Buffer amount in pixels (mutually exclusive with ``buf_spt_ext``).
    :param buf_spt_ext: Buffer amount in spatial projection units (mutually exclusive
                        with ``buf_pxl_ext``).
    :param gdalformat: Output GDAL format name (e.g., ``'GTIFF'``, ``'KEA'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param no_data_val: Optional no-data value. If ``None``, read from ``input_img``.
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or list of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_buf_img_from_ref_img(
            "in_img.tif", "buf_img.tif", buf_pxl_ext=10,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (buf_pxl_ext is None) and (buf_spt_ext is None):
        raise rsgispy.RSGISPyException(
            "You must specify either the buf_pxl_ext or buf_spt_ext value."
        )
    if (buf_pxl_ext is not None) and (buf_spt_ext is not None):
        raise rsgispy.RSGISPyException(
            "You cannot specify both the buf_pxl_ext or buf_spt_ext value."
        )
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        if no_data_val is None:
            no_data_val = src.nodata
            if no_data_val is None:
                raise rsgispy.RSGISPyException(
                    "You must specify a no data value"
                )

        src_transform = src.transform
        x_res_abs = math.hypot(src_transform.a, src_transform.d)
        y_res_abs = math.hypot(src_transform.b, src_transform.e)
        x_in_size = src.width
        y_in_size = src.height
        n_bands = src.count
        crs_obj = src.crs
        eff_dtype = (
            rsgis_dtype_from_np_dtype(src.dtypes[0])
            if datatype is None
            else datatype
        )

    if buf_spt_ext is not None:
        buf_pxl_x = math.ceil(buf_spt_ext / x_res_abs)
        buf_pxl_y = math.ceil(buf_spt_ext / y_res_abs)
    elif buf_pxl_ext is not None:
        buf_pxl_x = int(buf_pxl_ext)
        buf_pxl_y = int(buf_pxl_ext)
    else:
        raise rsgispy.RSGISPyException(
            "You must specify either the buf_pxl_ext or buf_spt_ext value."
        )

    buf_transform = src_transform @ Affine.translation(-buf_pxl_x, -buf_pxl_y)

    _write_blank_grid_img(
        output_img=output_img,
        n_bands=n_bands,
        width=x_in_size + (2 * buf_pxl_x),
        height=y_in_size + (2 * buf_pxl_y),
        transform=buf_transform,
        crs_obj=crs_obj,
        pxl_val=float(no_data_val),
        gdalformat=gdalformat,
        datatype=eff_dtype,
        no_data_val=float(no_data_val),
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def create_blank_img_from_bbox(
    bbox: Tuple[float, float, float, float],
    epsg_code: int,
    output_img: str,
    out_img_res: float,
    out_img_pxl_val: float = 0.0,
    out_img_n_bands: int = 1,
    snap_to_grid: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a new blank image file from a bounding box ``(xMin, xMax, yMin, yMax)``
    and an EPSG code.

    :param bbox: Bounding box ``(xMin, xMax, yMin, yMax)`` defining the output extent.
    :param epsg_code: EPSG integer code defining the output projection (e.g., 32630).
    :param output_img: Path to the output image file.
    :param out_img_res: Output pixel resolution (square pixels).
    :param out_img_pxl_val: Initial pixel value for all bands (Default = ``0.0``).
    :param out_img_n_bands: Number of image bands in the output image (Default = ``1``).
    :param snap_to_grid: Whether to snap the bounding box onto a grid of
                         whole multiples of ``out_img_res`` (Default = ``False``).
    :param gdalformat: Output GDAL format name (e.g., ``'GTIFF'``, ``'KEA'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional no-data value to set on the output dataset.
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or list of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.blank

        rsgispy.imageutils.blank.create_blank_img_from_bbox(
            (500000.0, 501000.0, 5599000.0, 5600000.0),
            32630, "blank_bbox.tif", 10.0, 0.0, 1,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    width, height, tl_x, tl_y = _bbox_to_grid_params(
        bbox=bbox, out_img_res=out_img_res, snap_to_grid=snap_to_grid
    )

    create_blank_img(
        output_img=output_img,
        n_bands=out_img_n_bands,
        width=width,
        height=height,
        tl_x=tl_x,
        tl_y=tl_y,
        res_x=out_img_res,
        res_y=-out_img_res,
        pxl_val=out_img_pxl_val,
        epsg_code=epsg_code,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def _resolve_vec_epsg(
    gdf: gpd.GeoDataFrame, epsg_code: Optional[int]
) -> Tuple[gpd.GeoDataFrame, int]:
    """Resolve target EPSG code and reproject ``gdf`` when ``epsg_code`` differs."""
    if epsg_code is not None:
        eff_epsg = int(epsg_code)
        if gdf.crs is not None:
            vec_epsg = gdf.crs.to_epsg()
            if vec_epsg is None or int(vec_epsg) != eff_epsg:
                gdf = gdf.to_crs(epsg=eff_epsg)
        return gdf, eff_epsg

    if gdf.crs is None:
        raise rsgispy.RSGISPyException(
            "Input vector layer does not have a CRS defined; specify epsg_code."
        )
    vec_epsg = gdf.crs.to_epsg()
    if vec_epsg is None:
        raise rsgispy.RSGISPyException(
            "Could not determine an EPSG code from the vector layer CRS; "
            "specify epsg_code."
        )
    return gdf, int(vec_epsg)


def _format_unique_name_val(val: Any, col_name: str) -> str:
    """Validate and convert an attribute value into a safe filename stem."""
    if pd.isna(val):
        raise rsgispy.RSGISPyException(
            f"Column '{col_name}' contains null/missing values; all output "
            "image name values must be non-null and unique."
        )
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        val_str = str(int(val))
    else:
        val_str = str(val).strip()
    if not val_str:
        raise rsgispy.RSGISPyException(
            f"Column '{col_name}' contains empty values; all output image "
            "name values must be non-empty and unique."
        )
    if "/" in val_str or "\\" in val_str or val_str in (".", ".."):
        raise rsgispy.RSGISPyException(
            f"Column '{col_name}' contains an invalid filename value: '{val_str}'."
        )
    return val_str


def _prepare_blank_ref_tasks(
    gdf: gpd.GeoDataFrame,
    out_img_col: str,
    out_dir_str: str,
    out_img_res: float,
    out_img_prefix: str,
    ext: str,
    snap_to_grid: bool,
) -> List[Tuple[str, int, int, float, float]]:
    """Validate polygon geometries and unique names, returning blank-image tasks."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if gdf[out_img_col].duplicated().any():
        dup_val = gdf.loc[gdf[out_img_col].duplicated(), out_img_col].iloc[0]
        raise rsgispy.RSGISPyException(
            f"Values in column '{out_img_col}' must be unique; duplicate found: "
            f"{dup_val!r}."
        )

    tasks: List[Tuple[str, int, int, float, float]] = []
    seen_paths = set()
    for name_val, geom in zip(gdf[out_img_col], gdf.geometry):
        val_str = _format_unique_name_val(name_val, out_img_col)
        if geom is None or geom.is_empty:
            raise rsgispy.RSGISPyException(
                f"Feature '{val_str}' has a null or empty geometry."
            )
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise rsgispy.RSGISPyException(
                f"All features must be Polygon or MultiPolygon geometries; "
                f"found '{geom.geom_type}' for '{val_str}'."
            )

        min_x, min_y, max_x, max_y = (float(v) for v in geom.bounds)
        if not all(
            math.isfinite(val) for val in (min_x, max_x, min_y, max_y)
        ):
            raise rsgispy.RSGISPyException(
                f"Feature '{val_str}' has non-finite bounding box coordinates."
            )

        width, height, tl_x, tl_y = _bbox_to_grid_params(
            bbox=(min_x, max_x, min_y, max_y),
            out_img_res=out_img_res,
            snap_to_grid=snap_to_grid,
        )

        base_name = f"{out_img_prefix}{val_str}"
        if ext and not base_name.lower().endswith(f".{ext.lower()}"):
            file_name = f"{base_name}.{ext}"
        else:
            file_name = base_name
        out_path = os.path.join(out_dir_str, file_name)
        if out_path in seen_paths:
            raise rsgispy.RSGISPyException(
                f"Values in column '{out_img_col}' must produce unique output "
                f"filenames; duplicate output path '{out_path}'."
            )
        seen_paths.add(out_path)
        tasks.append((out_path, width, height, tl_x, tl_y))

    return tasks


def create_blank_ref_imgs(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_img_col: str,
    out_img_dir: Union[str, os.PathLike],
    out_img_res: float,
    out_img_prefix: str = "",
    out_img_ext: Optional[str] = None,
    out_img_pxl_val: float = 0.0,
    out_img_n_bands: int = 1,
    snap_to_grid: bool = False,
    epsg_code: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> List[str]:
    """
    Create a blank reference raster image using :func:`create_blank_img` for the
    bounding box of each polygon in ``vec_file``, named from unique values in
    ``out_img_col``.

    :param vec_file: Input vector file path or in-memory
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Optional vector layer name within ``vec_file`` (or ``None``
                    for the default layer / in-memory ``GeoDataFrame``).
    :param out_img_col: Name of the attribute column containing unique, non-null
                        values used for the output image filenames.
    :param out_img_dir: Directory where output blank reference images will be
                        written (created if it does not already exist).
    :param out_img_res: Output pixel resolution (square pixels, positive).
    :param out_img_prefix: Optional filename prefix prepended to each value from
                           ``out_img_col`` (Default = ``''``).
    :param out_img_ext: Optional output file extension (without leading dot).
                        When ``None``, inferred from ``gdalformat``.
    :param out_img_pxl_val: Initial pixel value for all bands (Default = ``0.0``).
    :param out_img_n_bands: Number of image bands in each output image
                            (Default = ``1``).
    :param snap_to_grid: If ``True``, snap each polygon bounding box onto whole
                         multiples of ``out_img_res`` (Default = ``False``).
    :param epsg_code: Optional EPSG integer code. When ``None``, derived from
                      ``vec_file``; when provided with a CRS-defined vector
                      layer in a different CRS, geometries are reprojected to
                      ``epsg_code``.
    :param gdalformat: Output GDAL format name (e.g., ``'GTIFF'``, ``'KEA'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output images
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional no-data value to set on the output datasets.
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar across polygons
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or list of
                          ``'KEY=VALUE'``).
    :return: List of created output image file paths.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils

        out_imgs = rsgispy.imageutils.create_blank_ref_imgs(
            "tiles.gpkg",
            "tiles",
            "tile_name",
            "blank_tiles",
            out_img_res=10.0,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if int(out_img_n_bands) < 1:
        raise rsgispy.RSGISPyException("out_img_n_bands must be >= 1.")
    if not math.isfinite(float(out_img_res)) or float(out_img_res) <= 0.0:
        raise rsgispy.RSGISPyException(
            f"out_img_res must be a positive finite number, got {out_img_res!r}."
        )

    out_dir_str = os.fspath(out_img_dir)
    if os.path.exists(out_dir_str) and not os.path.isdir(out_dir_str):
        raise rsgispy.RSGISPyException(
            f"out_img_dir exists and is not a directory: '{out_dir_str}'."
        )

    if out_img_ext is None:
        ext = get_file_img_extension(gdalformat)
    else:
        ext = out_img_ext.lstrip(".")
        if not ext:
            raise rsgispy.RSGISPyException("out_img_ext cannot be empty.")

    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    if gdf.empty:
        raise rsgispy.RSGISPyException(
            "The inputted vector layer contains no features."
        )
    validate_columns_exist(gdf, [out_img_col])
    gdf, eff_epsg = _resolve_vec_epsg(gdf, epsg_code)

    tasks = _prepare_blank_ref_tasks(
        gdf=gdf,
        out_img_col=out_img_col,
        out_dir_str=out_dir_str,
        out_img_res=float(out_img_res),
        out_img_prefix=out_img_prefix,
        ext=ext,
        snap_to_grid=snap_to_grid,
    )

    os.makedirs(out_dir_str, exist_ok=True)
    tqdm_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    for out_path, width, height, tl_x, tl_y in tqdm_cls(
        tasks,
        desc="Creating blank reference images",
        unit="img",
        disable=not use_pbar,
    ):
        create_blank_img(
            output_img=out_path,
            n_bands=out_img_n_bands,
            width=width,
            height=height,
            tl_x=tl_x,
            tl_y=tl_y,
            res_x=float(out_img_res),
            res_y=-float(out_img_res),
            pxl_val=out_img_pxl_val,
            epsg_code=eff_epsg,
            gdalformat=gdalformat,
            datatype=datatype,
            no_data_val=no_data_val,
            calc_stats=calc_stats,
            use_pbar=False,
            creation_opts=creation_opts,
        )
        out_files.append(out_path)

    return out_files
