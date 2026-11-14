"""
Rasterisation and distance-zone raster generation utilities for
:mod:`rsgispy.imageutils.rasterise` using ``geopandas``, ``rasterio``, and
:mod:`rsgispy.imagecalc`.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj
import rasterio
import rasterio.features
from rasterio.transform import Affine
import rasterio.windows
from shapely.geometry import box

import rsgispy
import rsgispy.imagecalc.distance as _img_dist
from rsgispy.imageprocess import (
    _INT_DTYPES,
    IntersectionGrid,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imageutils._helpers import normalise_creation_options
from rsgispy.imageutils.blank import (
    create_blank_img_from_bbox,
    create_copy_img,
)
from rsgispy.imageutils.metadata import get_img_res
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools.geometry import find_extent_on_whole_num_grid
import rsgispy.vectorattrs.columns
from rsgispy.vectorprocess import (
    read_vec_gdf,
    resolve_vec_layer,
    validate_columns_exist,
)
from rsgispy.vectorutils.metadata import (
    get_epsg_proj_from_vec,
    get_vec_lyr_bbox,
    get_wkt_proj_from_vec,
)

__all__ = [
    "rasterise_vec_lyr",
    "create_vec_range_lut_score_img",
    "create_vec_lst_lut_score_img",
    "create_dist_zones_to_vec_lyr",
]

_SUPPORTED_RASTERIZE_DTYPES = {
    np.dtype("uint8"),
    np.dtype("int16"),
    np.dtype("uint16"),
    np.dtype("int32"),
    np.dtype("uint32"),
    np.dtype("float32"),
    np.dtype("float64"),
}


def _create_blank_grid_image(
    output_img: str,
    width: int,
    height: int,
    tl_x: float,
    tl_y: float,
    res_x: float,
    res_y: float,
    crs_obj: Any,
    gdalformat: str,
    datatype: int,
    no_data_val: Optional[float],
    use_pbar: bool,
) -> None:
    """Create a blank single-band image when EPSG integer code is unavailable."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    transform = Affine(
        abs(float(res_x)),
        0.0,
        float(tl_x),
        0.0,
        -abs(float(res_y)),
        float(tl_y),
    )
    grid = IntersectionGrid(
        crs=crs_obj,
        transform=transform,
        width=int(width),
        height=int(height),
        offsets={},
    )
    with ExitStack() as stack:
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=normalise_creation_options(None),
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            dst.write(
                np.zeros((1, int(win.height), int(win.width)), dtype=out_np_dtype),
                window=win,
            )


def _prepare_rasterise_target(
    vec_file: str,
    vec_lyr: str,
    input_img: Optional[str],
    output_img: str,
    gdalformat: str,
    datatype: int,
    use_vec_extent: bool,
    no_data_val: float,
    use_pbar: bool,
) -> None:
    """Create or verify ``output_img`` before rasterising features onto it."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if use_vec_extent:
        if input_img is None:
            raise rsgispy.RSGISPyException(
                "input_img must be provided to supply pixel resolution when "
                "use_vec_extent=True."
            )
        ensure_file_exists(input_img)
        x_res, y_res = get_img_res(input_img, abs_vals=True)
        out_res = min(x_res, y_res)

        vec_bbox = get_vec_lyr_bbox(
            vec_file, vec_lyr=vec_lyr, compute_if_exp=True
        )
        snap_bbox = find_extent_on_whole_num_grid(
            vec_bbox, out_res, full_contain=True
        )
        epsg_code = get_epsg_proj_from_vec(vec_file, vec_lyr=vec_lyr)
        if epsg_code is not None:
            create_blank_img_from_bbox(
                bbox=snap_bbox,
                epsg_code=epsg_code,
                output_img=output_img,
                out_img_res=out_res,
                out_img_pxl_val=0.0,
                out_img_n_bands=1,
                gdalformat=gdalformat,
                datatype=datatype,
                snap_to_grid=False,
                no_data_val=no_data_val,
                use_pbar=use_pbar,
            )
            return

        wkt_str = get_wkt_proj_from_vec(vec_file, vec_lyr=vec_lyr)
        if wkt_str:
            crs_obj = pyproj.CRS.from_wkt(wkt_str)
        else:
            with rasterio.open(input_img, "r") as src:
                crs_obj = src.crs
        width = max(1, int(math.ceil((snap_bbox[1] - snap_bbox[0]) / out_res)))
        height = max(
            1, int(math.ceil((snap_bbox[3] - snap_bbox[2]) / out_res))
        )
        _create_blank_grid_image(
            output_img=output_img,
            width=width,
            height=height,
            tl_x=snap_bbox[0],
            tl_y=snap_bbox[3],
            res_x=out_res,
            res_y=out_res,
            crs_obj=crs_obj,
            gdalformat=gdalformat,
            datatype=datatype,
            no_data_val=no_data_val,
            use_pbar=use_pbar,
        )
        return

    if input_img is None:
        ensure_file_exists(output_img, label="Output image")
        return

    ensure_file_exists(input_img)
    try:
        create_copy_img(
            input_img=input_img,
            output_img=output_img,
            n_bands=1,
            pxl_val=0.0,
            gdalformat=gdalformat,
            datatype=datatype,
            no_data_val=no_data_val,
            use_pbar=use_pbar,
        )
    except rsgispy.RSGISPyException:
        with rasterio.open(input_img, "r") as src:
            _create_blank_grid_image(
                output_img=output_img,
                width=src.width,
                height=src.height,
                tl_x=src.transform.c,
                tl_y=src.transform.f,
                res_x=src.transform.a,
                res_y=src.transform.e,
                crs_obj=src.crs,
                gdalformat=gdalformat,
                datatype=datatype,
                no_data_val=no_data_val,
                use_pbar=use_pbar,
            )


def rasterise_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    input_img: Optional[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    burn_val: int = 1,
    datatype: int = rsgispy.TYPE_8UINT,
    att_column: Optional[str] = None,
    use_vec_extent: bool = False,
    thematic: bool = True,
    no_data_val: float = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Rasterise a vector layer onto an output raster grid using ``rasterio``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param input_img: Reference input raster defining pixel grid and resolution.
                      May be ``None`` if ``output_img`` already exists and
                      ``use_vec_extent=False``.
    :param output_img: Output raster image file path.
    :param gdalformat: Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    :param burn_val: Constant pixel burn value when ``att_column`` is ``None``
                     (Default = ``1``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_8UINT``).
    :param att_column: Optional vector column name whose values are burned into
                       the raster. If ``'FID'`` and no ``'FID'`` column exists,
                       a 1-based feature ID sequence is generated automatically.
    :param use_vec_extent: If ``True``, snap output raster extent to the vector
                           layer bounding box using ``input_img``'s pixel
                           resolution (Default = ``False``).
    :param thematic: If ``True`` and ``calc_stats=True``, populate thematic
                     raster statistics and colour table for integer outputs
                     (Default = ``True``).
    :param no_data_val: No-data value for continuous statistics/metadata
                        (Default = ``0.0``).
    :param calc_stats: If ``True``, compute raster statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.rasterise

        rsgispy.imageutils.rasterise.rasterise_vec_lyr(
            "crowns.gpkg",
            "crowns",
            "ref_img.tif",
            "crowns_ras.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            att_column="FID",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    _prepare_rasterise_target(
        vec_file=vec_file,
        vec_lyr=lyr_name,
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        use_vec_extent=use_vec_extent,
        no_data_val=no_data_val,
        use_pbar=use_pbar,
    )

    gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if len(gdf) > 0:
        if att_column is not None:
            if att_column == "FID" and "FID" not in gdf.columns:
                gdf["FID"] = np.arange(1, len(gdf) + 1, 1, dtype=int)
            else:
                validate_columns_exist(gdf, [att_column])

        with rasterio.open(output_img, "r+") as dst:
            if (
                dst.crs is not None
                and gdf.crs is not None
                and gdf.crs != dst.crs
            ):
                gdf = gdf.to_crs(dst.crs)

            dst_np_dtype = np.dtype(dst.dtypes[0])
            is_int_out = dst.dtypes[0] in _INT_DTYPES
            ras_dtype = (
                dst_np_dtype
                if dst_np_dtype in _SUPPORTED_RASTERIZE_DTYPES
                else np.float64
            )

            if att_column is None:
                vals = [
                    int(burn_val) if is_int_out else float(burn_val)
                    for _ in range(len(gdf))
                ]
            else:
                vals = []
                for raw_v in gdf[att_column]:
                    if raw_v is None or (
                        isinstance(raw_v, float) and math.isnan(raw_v)
                    ):
                        vals.append(None)
                    elif is_int_out:
                        vals.append(int(raw_v))
                    else:
                        vals.append(float(raw_v))

            geoms = gdf.geometry.to_numpy()
            sindex = gdf.sindex

            for win in iter_dataset_windows(
                dst.height, dst.width, use_pbar=use_pbar
            ):
                min_x, min_y, max_x, max_y = rasterio.windows.bounds(
                    win, dst.transform
                )
                cand_idxs = sindex.query(
                    box(min_x, min_y, max_x, max_y), predicate="intersects"
                )
                if len(cand_idxs) == 0:
                    continue

                win_shapes = [
                    (geoms[idx], vals[idx])
                    for idx in cand_idxs
                    if vals[idx] is not None
                ]
                if not win_shapes:
                    continue

                win_arr = dst.read(1, window=win)
                work_arr = (
                    win_arr
                    if ras_dtype == dst_np_dtype
                    else win_arr.astype(ras_dtype)
                )
                burned = rasterio.features.rasterize(
                    shapes=win_shapes,
                    out=work_arr,
                    transform=dst.window_transform(win),
                )
                if ras_dtype != dst_np_dtype:
                    burned = burned.astype(dst_np_dtype)
                dst.write(burned, 1, window=win)

    if calc_stats:
        with rasterio.open(output_img, "r") as check_ds:
            dtype_str = check_ds.dtypes[0]
        if thematic and (dtype_str in _INT_DTYPES):
            pop_thmt_img_stats(
                input_img=output_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=True,
                use_pbar=use_pbar,
            )
        else:
            pop_img_stats(
                input_img=output_img,
                no_data_val=no_data_val,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def create_vec_range_lut_score_img(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_col: str,
    tmp_vec_file: str,
    tmp_vec_lyr: str,
    tmp_vec_col: str,
    input_img: str,
    output_img: str,
    scrs_lut: Dict[int, Tuple[float, float]],
    out_format: str = "GPKG",
    gdalformat: str = "GTIFF",
    use_pbar: bool = True,
) -> None:
    """
    Recode a numeric vector attribute using a half-open range look-up table
    ``{code: (low, high)}`` and rasterise the result onto ``input_img``'s grid.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param vec_col: Numeric attribute column name to recode.
    :param tmp_vec_file: Intermediate vector file path for recoded column.
    :param tmp_vec_lyr: Intermediate vector layer name.
    :param tmp_vec_col: Intermediate numeric column name.
    :param input_img: Reference raster image defining output grid and CRS.
    :param output_img: Output raster image file path.
    :param scrs_lut: Dictionary mapping integer output code to ``(low, high)``
                     checked as ``low <= val < high``.
    :param out_format: Intermediate vector format driver (Default = ``'GPKG'``).
    :param gdalformat: Output raster format driver (Default = ``'GTIFF'``).
    :param use_pbar: If ``True``, display a tqdm progress bar during rasterising
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageutils.rasterise

        rsgispy.imageutils.rasterise.create_vec_range_lut_score_img(
            "plots.gpkg",
            "plots",
            "biomass",
            "tmp_score.gpkg",
            "plots",
            "score",
            "ref_img.tif",
            "score.tif",
            scrs_lut={1: (0.0, 50.0), 2: (50.0, 200.0)},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.vectorattrs.columns.add_numeric_col_range_lut(
        vec_file=vec_file,
        vec_lyr=vec_lyr,
        vec_col=vec_col,
        out_vec_file=tmp_vec_file,
        out_vec_lyr=tmp_vec_lyr,
        out_vec_col=tmp_vec_col,
        val_lut=scrs_lut,
        out_format=out_format,
    )
    rasterise_vec_lyr(
        vec_file=tmp_vec_file,
        vec_lyr=tmp_vec_lyr,
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        burn_val=1,
        datatype=rsgispy.TYPE_8UINT,
        att_column=tmp_vec_col,
        use_vec_extent=False,
        thematic=True,
        no_data_val=0.0,
        use_pbar=use_pbar,
    )


def create_vec_lst_lut_score_img(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_col: str,
    tmp_vec_file: str,
    tmp_vec_lyr: str,
    tmp_vec_col: str,
    input_img: str,
    output_img: str,
    scrs_lut: Sequence[Tuple[Union[str, int], int]],
    out_format: str = "GPKG",
    gdalformat: str = "GTIFF",
    use_pbar: bool = True,
) -> None:
    """
    Recode a categorical vector attribute using a list of ``(value, code)``
    look-up pairs and rasterise the result onto ``input_img``'s grid.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param vec_col: Attribute column name to match against ``scrs_lut``.
    :param tmp_vec_file: Intermediate vector file path for recoded column.
    :param tmp_vec_lyr: Intermediate vector layer name.
    :param tmp_vec_col: Intermediate numeric column name.
    :param input_img: Reference raster image defining output grid and CRS.
    :param output_img: Output raster image file path.
    :param scrs_lut: Sequence of ``(lookup_val, output_int)`` pairs.
    :param out_format: Intermediate vector format driver (Default = ``'GPKG'``).
    :param gdalformat: Output raster format driver (Default = ``'GTIFF'``).
    :param use_pbar: If ``True``, display a tqdm progress bar during rasterising
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imageutils.rasterise

        rsgispy.imageutils.rasterise.create_vec_lst_lut_score_img(
            "land.gpkg",
            "land",
            "cover",
            "tmp_score.gpkg",
            "land",
            "score",
            "ref_img.tif",
            "score.tif",
            scrs_lut=[("Forest", 1), ("Water", 2)],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.vectorattrs.columns.add_numeric_col_from_lst_lut(
        vec_file=vec_file,
        vec_lyr=vec_lyr,
        ref_col=vec_col,
        vals_lut=scrs_lut,
        out_col=tmp_vec_col,
        out_vec_file=tmp_vec_file,
        out_vec_lyr=tmp_vec_lyr,
        out_format=out_format,
    )
    rasterise_vec_lyr(
        vec_file=tmp_vec_file,
        vec_lyr=tmp_vec_lyr,
        input_img=input_img,
        output_img=output_img,
        gdalformat=gdalformat,
        burn_val=1,
        datatype=rsgispy.TYPE_8UINT,
        att_column=tmp_vec_col,
        use_vec_extent=False,
        thematic=True,
        no_data_val=0.0,
        use_pbar=use_pbar,
    )


def create_dist_zones_to_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    input_img: str,
    tmp_vec_img: str,
    tmp_dist_img: str,
    output_img: str,
    recode_lut: Sequence[Tuple[int, Tuple[float, float]]],
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_8UINT,
    max_dist_thres: Optional[float] = None,
    backgrd_val: int = 0,
    use_pbar: bool = True,
) -> None:
    """
    Rasterise ``vec_file``, compute Euclidean distance to rasterised features,
    and recode distances into categorical zones via ``recode_lut``.

    Each entry in ``recode_lut`` is ``(out_code, (low, high))`` checked as
    ``low <= dist < high``. Use ``math.nan`` (or ``None``) for open lower or
    upper bounds (e.g., ``(1, (math.nan, 50.0))`` or ``(3, (200.0, math.nan))``).

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param input_img: Reference raster defining the grid and CRS.
    :param tmp_vec_img: Intermediate binary rasterised vector image path.
    :param tmp_dist_img: Intermediate Euclidean distance raster image path.
    :param output_img: Output categorical zone raster image path.
    :param recode_lut: Sequence of ``(zone_val, (low_dist, high_dist))`` rules.
    :param gdalformat: Output raster format driver (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_8UINT``).
    :param max_dist_thres: Optional maximum distance threshold beyond which
                           pixels receive ``backgrd_val``.
    :param backgrd_val: Background value for pixels outside all LUT intervals
                        or beyond ``max_dist_thres`` (Default = ``0``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import math
        import rsgispy.imageutils.rasterise

        rsgispy.imageutils.rasterise.create_dist_zones_to_vec_lyr(
            "roads.gpkg",
            "roads",
            "ref_img.tif",
            "tmp_roads.tif",
            "tmp_dist.tif",
            "road_zones.tif",
            recode_lut=[(1, (0.0, 50.0)), (2, (50.0, math.nan))],
            max_dist_thres=200.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not hasattr(_img_dist, "calc_dist_to_img_vals"):
        raise rsgispy.RSGISPyException(
            "scipy is required for create_dist_zones_to_vec_lyr."
        )

    rasterise_vec_lyr(
        vec_file=vec_file,
        vec_lyr=vec_lyr,
        input_img=input_img,
        output_img=tmp_vec_img,
        gdalformat=gdalformat,
        burn_val=1,
        datatype=rsgispy.TYPE_8UINT,
        calc_stats=False,
        use_pbar=use_pbar,
    )

    dist_nd_val = -9999.0
    _img_dist.calc_dist_to_img_vals(
        input_img=tmp_vec_img,
        output_img=tmp_dist_img,
        pxl_vals=[1],
        img_band=1,
        gdalformat=gdalformat,
        max_dist=max_dist_thres,
        no_data_val=None,
        out_no_data_val=dist_nd_val,
        unit_geo=True,
    )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        dist_src = stack.enter_context(rasterio.open(tmp_dist_img, "r"))
        grid = IntersectionGrid(
            crs=dist_src.crs,
            transform=dist_src.transform,
            width=dist_src.width,
            height=dist_src.height,
            offsets={},
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=normalise_creation_options(None),
        )

        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            dist_arr = dist_src.read(1, window=win)
            valid = (dist_arr != dist_nd_val) & np.isfinite(dist_arr)
            out_arr = np.full(
                (int(win.height), int(win.width)),
                backgrd_val,
                dtype=out_np_dtype,
            )

            for cat_val, (low_val, high_val) in recode_lut:
                has_low = (low_val is not None) and (not math.isnan(low_val))
                has_high = (high_val is not None) and (not math.isnan(high_val))
                if has_low and has_high:
                    mask = valid & (dist_arr >= low_val) & (dist_arr < high_val)
                elif has_low:
                    mask = valid & (dist_arr >= low_val)
                elif has_high:
                    mask = valid & (dist_arr < high_val)
                else:
                    mask = valid
                out_arr[mask] = cat_val

            dst.write(out_arr, 1, window=win)
