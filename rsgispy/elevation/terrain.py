"""
The elevation.terrain module provides slope and aspect terrain derivative
functions for digital elevation models (DEMs) using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
from typing import Optional

import numpy as np
import rasterio

import rsgispy
from rsgispy.elevation._helpers import (
    calc_slope_aspect_from_diffs as _calc_slope_aspect_from_diffs,
    process_pxl_res_dem_3x3 as _process_pxl_res_dem_3x3,
    process_single_dem_3x3 as _process_single_dem_3x3,
    validate_slope_unit as _validate_slope_unit,
)
from rsgispy.imageprocess import (
    cast_result_block,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "slope",
    "slope_pxl_res_img",
    "aspect",
    "aspect_pxl_res_img",
    "categorise_aspect",
]


def slope(
    input_img: str,
    output_img: str,
    unit: str = "degrees",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a slope layer from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output slope image file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.slope(
            input_img="dem.tif",
            output_img="slope.tif",
            unit="degrees",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    unit_norm = _validate_slope_unit(unit)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        slope_rad, _, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        if unit_norm == "degrees":
            return np.degrees(slope_rad)
        return slope_rad

    _process_single_dem_3x3(
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        _block,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def slope_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    unit: str = "degrees",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a slope layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    Useful for geographic projections (e.g., EPSG:4326) whose horizontal units
    differ from the elevation units.

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output slope image file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.slope_pxl_res_img(
            in_dem_img="dem.tif",
            in_pxl_res_img="pxl_res.tif",
            output_img="slope.tif",
            unit="degrees",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    unit_norm = _validate_slope_unit(unit)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: np.ndarray,
        ns_res: np.ndarray,
    ) -> np.ndarray:
        slope_rad, _, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        if unit_norm == "degrees":
            return np.degrees(slope_rad)
        return slope_rad

    _process_pxl_res_dem_3x3(
        in_dem_img,
        in_pxl_res_img,
        output_img,
        gdalformat,
        eff_dtype,
        _block,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def aspect(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate an aspect layer (in degrees clockwise from North, ``[0, 360)``,
    ``NaN`` for flat areas) from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.aspect(
            input_img="dem.tif",
            output_img="aspect.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: float,
        ns_res: float,
    ) -> np.ndarray:
        _, aspect_deg, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        return aspect_deg

    _process_single_dem_3x3(
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        _block,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def aspect_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate an aspect layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.aspect_pxl_res_img(
            in_dem_img="dem.tif",
            in_pxl_res_img="pxl_res.tif",
            output_img="aspect.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: np.ndarray,
        ns_res: np.ndarray,
    ) -> np.ndarray:
        _, aspect_deg, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        return aspect_deg

    _process_pxl_res_dem_3x3(
        in_dem_img,
        in_pxl_res_img,
        output_img,
        gdalformat,
        eff_dtype,
        _block,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def categorise_aspect(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Categorise an aspect image (in degrees) into 8 classes (1-8 for 45-degree
    bins: 0-45, 45-90, ..., 315-360) and ``0`` for ``NaN``/flat pixels.

    :param input_img: Path to the input aspect image file (in degrees).
    :param output_img: Path to the output categorised image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_8UINT`` when ``None``).
    :param no_data_val: Optional input aspect no-data value treated as
                        invalid/flat (read from header when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.categorise_aspect(
            input_img="aspect.tif",
            output_img="aspect_cats.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )

        for out_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            arr = src.read(1, window=out_win, out_dtype=np.float64)
            invalid_mask = np.isnan(arr)
            if eff_nd is not None and not np.isnan(eff_nd):
                invalid_mask |= arr == eff_nd
            valid = ~invalid_mask
            if np.any(arr[valid] < 0.0):
                raise rsgispy.RSGISPyException(
                    "The input image pixel values must be between 0 and 360 degrees."
                )

            norm = np.where(arr > 360.0, np.mod(arr, 360.0), arr)
            out_cls = np.zeros(arr.shape, dtype=out_np_dtype)
            bins = np.clip((norm[valid] // 45.0).astype(np.int64) + 1, 1, 8)
            out_cls[valid] = bins.astype(out_np_dtype)
            dst.write(
                cast_result_block(
                    out_cls, int(out_win.height), int(out_win.width), out_np_dtype
                ),
                1,
                window=out_win,
            )

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=None,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )
