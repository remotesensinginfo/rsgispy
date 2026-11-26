"""
The elevation.illumination module provides solar hillshading, ray-traced shadow
masking, and local solar/sensor incidence and exitance angle functions using
``rasterio`` and ``numpy``.

Hillshade and the incidence and exitance angles also have tile-graph node
builders (:func:`hillshade_node`, :func:`hillshade_pxl_res_img_node`,
:func:`local_incidence_angle_node`, :func:`local_exitance_angle_node`) for
chaining them with other operations in a
:class:`~rsgispy.imageprocess.TileGraph`. :func:`shadow_mask` has no node:
its solar rays are traced across the whole DEM (a cast shadow can come from
terrain any distance away), so it cannot be computed tile by tile with a
fixed halo.
"""

from contextlib import ExitStack
import math
from typing import Any, Optional, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy._io import TileNode
from rsgispy._numba import optional_njit, prange, use_compiled
from rsgispy.elevation._helpers import (
    Horn3x3Node as _Horn3x3Node,
    HornBlockFunc as _HornBlockFunc,
    calc_slope_aspect_from_diffs as _calc_slope_aspect_from_diffs,
    prepare_3x3_horn_diffs as _prepare_3x3_horn_diffs,
    run_terrain_node_to_file as _run_terrain_node_to_file,
    validate_sun_angles as _validate_sun_angles,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask as _valid_mask,
    ensure_file_exists,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "hillshade",
    "hillshade_pxl_res_img",
    "shadow_mask",
    "local_incidence_angle",
    "local_exitance_angle",
    "hillshade_node",
    "hillshade_pxl_res_img_node",
    "local_incidence_angle_node",
    "local_exitance_angle_node",
]


def _calc_hillshade_block(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: Any,
    ns_res: Any,
    azimuth: float,
    zenith: float,
) -> np.ndarray:
    """Compute uint8 hillshade values (range 1..255) matching RSGISLib."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sun_elev_rad = math.radians(90.0 - zenith)
    sun_az_trans = 360.0 - azimuth + 90.0
    if sun_az_trans > 360.0:
        sun_az_trans -= 360.0
    sun_az_rad = math.radians(sun_az_trans)

    dx = -dx_w_minus_e / (ew_res * 8.0)
    dy = -dy_s_minus_n / (ns_res * 8.0)
    xx_plus_yy = (dx * dx) + (dy * dy)
    aspect_math = np.arctan2(dy, dx)

    cang = (
        math.sin(sun_elev_rad)
        - math.cos(sun_elev_rad)
        * np.sqrt(xx_plus_yy)
        * np.sin(aspect_math - (sun_az_rad - (math.pi / 2.0)))
    ) / np.sqrt(1.0 + xx_plus_yy)

    out = np.where((n_vals > 1) & (cang > 0.0), 1.0 + (254.0 * cang), 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _calc_ray_angle_block(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: float,
    ns_res: float,
    azimuth: float,
    zenith: float,
    flat_aspect_val: float,
    fallback_val: float,
) -> np.ndarray:
    """Shared ray incidence/exitance angle calculation for a 3x3 DEM block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    slope_rad, aspect_deg, flat_mask = _calc_slope_aspect_from_diffs(
        dx_w_minus_e, dy_s_minus_n, n_vals, ew_res, ns_res
    )
    if not math.isnan(flat_aspect_val):
        aspect_deg = np.where((n_vals > 1) & flat_mask, flat_aspect_val, aspect_deg)

    aspect_rad = np.radians(aspect_deg)
    pa = np.sin(slope_rad) * np.cos(aspect_rad)
    pb = np.sin(slope_rad) * np.sin(aspect_rad)
    pc = np.cos(slope_rad)

    zen_rad = math.radians(zenith)
    az_rad = math.radians(azimuth)
    ra = math.sin(zen_rad) * np.cos(az_rad)
    rb = math.sin(zen_rad) * np.sin(az_rad)
    rc = math.cos(zen_rad)

    dot = np.clip((pa * ra) + (pb * rb) + (pc * rc), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(dot))
    invalid = (n_vals <= 1) | np.isnan(ang_deg)
    return np.where(invalid, fallback_val, ang_deg)


def _make_hillshade_node(
    source: TileNode,
    pxl_res: Optional[TileNode],
    azimuth: float,
    zenith: float,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
    out_datatype: int = rsgispy.TYPE_8UINT,
    res_nodata: Optional[Tuple[Any, Any]] = None,
) -> TileNode:
    """Hillshade node (``out_datatype`` sets the no-data fill as the file function)."""
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    az = float(azimuth)
    zen = float(zenith)

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: Any,
        ns_res: Any,
    ) -> np.ndarray:
        return _calc_hillshade_block(dx, dy, n_vals, ew_res, ns_res, az, zen)

    return _Horn3x3Node(
        source,
        _block,
        out_datatype,
        pxl_res=pxl_res,
        no_data_val=no_data_val,
        res_nodata=res_nodata,
        uint8_values=True,
        op_desc=f"hillshade azimuth={az} zenith={zen}",
        name=name or ("hillshade_pxl_res_img" if pxl_res is not None else "hillshade"),
    )


def _ray_angle_block_fn(
    azimuth: float, zenith: float, fallback_val: float
) -> _HornBlockFunc:
    """Block function of the local incidence/exitance angle."""

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: Any,
        ns_res: Any,
    ) -> np.ndarray:
        return _calc_ray_angle_block(
            dx,
            dy,
            n_vals,
            ew_res,
            ns_res,
            azimuth,
            zenith,
            flat_aspect_val=0.0,
            fallback_val=fallback_val,
        )

    return _block


def _make_ray_angle_node(
    source: TileNode,
    azimuth: float,
    zenith: float,
    exitance: bool,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
    out_datatype: int = rsgispy.TYPE_32FLOAT,
) -> TileNode:
    """Local incidence (``exitance=False``) or exitance angle node."""
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    # Pixels without a slope get the zenith angle (incidence) or 0 (exitance).
    fallback = 0.0 if exitance else float(zenith)
    label = "local_exitance_angle" if exitance else "local_incidence_angle"
    return _Horn3x3Node(
        source,
        _ray_angle_block_fn(azimuth, zenith, fallback),
        out_datatype,
        no_data_val=no_data_val,
        op_desc=f"{label} azimuth={float(azimuth)} zenith={float(zenith)}",
        name=name or label,
    )


def hillshade(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a hillshade layer from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output hillshade image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]`` (e.g., 315).
    :param zenith: Solar zenith angle in degrees ``[0, 90]`` (e.g., 45).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_8UINT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.illumination

        rsgispy.elevation.illumination.hillshade(
            input_img="dem.tif",
            output_img="hillshade.tif",
            azimuth=315.0,
            zenith=45.0,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, _res, _res_nd: _make_hillshade_node(
            dem, None, azimuth, zenith, no_data_val=no_data_val, out_datatype=eff_dtype
        ),
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def hillshade_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a hillshade layer from a DEM and an external 2-band pixel
    resolution image (Band 1: East-West resolution, Band 2: North-South resolution).

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output hillshade image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]`` (e.g., 315).
    :param zenith: Solar zenith angle in degrees ``[0, 90]`` (e.g., 45).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_8UINT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.illumination

        rsgispy.elevation.illumination.hillshade_pxl_res_img(
            in_dem_img="dem.tif",
            in_pxl_res_img="pxl_res.tif",
            output_img="hillshade.tif",
            azimuth=315.0,
            zenith=45.0,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, res, res_nd: _make_hillshade_node(
            dem,
            res,
            azimuth,
            zenith,
            no_data_val=no_data_val,
            out_datatype=eff_dtype,
            res_nodata=res_nd,
        ),
        in_dem_img,
        output_img,
        gdalformat,
        eff_dtype,
        in_pxl_res_img=in_pxl_res_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


@optional_njit(parallel=True, cache=True, warn=False)
def _shadow_ray_kernel(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    cand_mask: np.ndarray,
    col_step: float,
    row_step: float,
    z_step: float,
    max_height: float,
    max_steps: int,
) -> np.ndarray:
    """
    Compiled per-pixel solar ray-marching kernel (parallel over rows).

    For each ``cand_mask`` pixel a ray starts at the pixel centre and its DEM
    elevation and advances by ``(col_step, row_step, z_step)`` per step,
    stopping early once it leaves the raster, rises above ``max_height``, or
    passes beneath a valid DEM cell (an occluding hit). Returns a boolean hit
    mask. Uses the same cumulative step arithmetic as
    :func:`_shadow_ray_trace_numpy`, so results are identical.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,chained-comparison
    height, width = dem.shape
    hit = np.zeros((height, width), dtype=np.bool_)
    for r in prange(height):
        for c in range(width):
            if not cand_mask[r, c]:
                continue
            col_f = np.float64(c) + 0.5
            row_f = np.float64(r) + 0.5
            z_f = dem[r, c]
            for _ in range(max_steps):
                col_f += col_step
                row_f += row_step
                z_f += z_step
                if not (
                    col_f >= 0.0
                    and col_f < width
                    and row_f >= 0.0
                    and row_f < height
                    and z_f <= max_height
                ):
                    break
                rr = int(row_f)
                cc = int(col_f)
                if (rr != r or cc != c) and dem_valid[rr, cc] and z_f < dem[rr, cc]:
                    hit[r, c] = True
                    break
    return hit


def _shadow_ray_trace_numpy(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    cand_mask: np.ndarray,
    col_step: float,
    row_step: float,
    z_step: float,
    max_height: float,
    max_steps: int,
    use_pbar: bool,
) -> np.ndarray:
    """Vectorised NumPy fallback for :func:`_shadow_ray_kernel`."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = dem.shape
    hit_mask = np.zeros((height, width), dtype=bool)
    cand_r, cand_c = np.where(cand_mask)
    if cand_r.size == 0:
        return hit_mask
    orig_r = cand_r.copy()
    orig_c = cand_c.copy()
    col_f = orig_c.astype(np.float64) + 0.5
    row_f = orig_r.astype(np.float64) + 0.5
    z_f = dem[orig_r, orig_c].copy()

    tqdm_cls = rsgispy.get_tqdm()
    for _ in tqdm_cls(
        range(max_steps),
        desc="Shadow mask",
        unit="step",
        disable=not use_pbar,
    ):
        col_f += col_step
        row_f += row_step
        z_f += z_step

        in_bounds = (
            (col_f >= 0.0)
            & (col_f < width)
            & (row_f >= 0.0)
            & (row_f < height)
            & (z_f <= max_height)
        )
        if not np.any(in_bounds):
            break

        orig_r = orig_r[in_bounds]
        orig_c = orig_c[in_bounds]
        col_f = col_f[in_bounds]
        row_f = row_f[in_bounds]
        z_f = z_f[in_bounds]

        rr = row_f.astype(np.int64)
        cc = col_f.astype(np.int64)
        moved = (rr != orig_r) | (cc != orig_c)
        hit = moved & dem_valid[rr, cc] & (z_f < dem[rr, cc])
        if np.any(hit):
            hit_mask[orig_r[hit], orig_c[hit]] = True
            keep = ~hit
            if not np.any(keep):
                break
            orig_r = orig_r[keep]
            orig_c = orig_c[keep]
            col_f = col_f[keep]
            row_f = row_f[keep]
            z_f = z_f[keep]
    return hit_mask


def shadow_mask(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    max_height: float,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a binary shadow mask (``1`` = shadow, ``0`` = illuminated/nodata)
    given an input elevation model using self-shadowing and solar ray tracing.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output binary mask image file.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]``.
    :param zenith: Solar zenith angle in degrees ``[0, 90]``.
    :param max_height: Maximum elevation threshold terminating upward ray tracing.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_8UINT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during ray tracing (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.illumination

        rsgispy.elevation.illumination.shadow_mask(
            input_img="dem.tif",
            output_img="shadow_mask.tif",
            azimuth=315.0,
            zenith=45.0,
            max_height=3000.0,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    _validate_sun_angles(azimuth, zenith)
    ensure_file_exists(input_img)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = IntersectionGrid.from_dataset(src)
        ew_res = abs(float(src.transform.a))
        ns_res = abs(float(src.transform.e))
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        dem = src.read(1, out_dtype=np.float64)

        dem_valid = _valid_mask(dem, eff_nd)
        dem_halo = np.pad(dem, 1, mode="edge")
        dx, dy, n_vals = _prepare_3x3_horn_diffs(dem_halo, eff_nd)
        slope_rad, aspect_deg, flat_mask = _calc_slope_aspect_from_diffs(
            dx, dy, n_vals, ew_res, ns_res
        )

        sun_zen_rad = math.radians(zenith)
        sun_az_rad = math.radians(azimuth)

        shadow = np.zeros(dem.shape, dtype=out_np_dtype)
        has_slope = dem_valid & (n_vals > 1) & ~flat_mask
        ic = (
            math.cos(sun_zen_rad) * np.cos(slope_rad)
            + math.sin(sun_zen_rad)
            * np.sin(slope_rad)
            * np.cos(sun_az_rad - np.radians(aspect_deg))
        )
        shadow[has_slope & (ic < 0.0)] = 1

        # Ray tracing for remaining valid pixels not already self-shadowed
        sun_az_trans = (360.0 - azimuth + 90.0) % 360.0
        sun_az_trans_rad = math.radians(sun_az_trans)
        step_range = min(ew_res, ns_res)
        x_step = (step_range * math.sin(sun_zen_rad) * math.cos(sun_az_trans_rad)) / 3.0
        y_step = (step_range * math.sin(sun_zen_rad) * math.sin(sun_az_trans_rad)) / 3.0
        z_step = (step_range * math.cos(sun_zen_rad)) / 3.0

        col_step = x_step / ew_res
        row_step = -y_step / ns_res

        if abs(col_step) > 1e-12 or abs(row_step) > 1e-12:
            max_steps = int(
                math.ceil(
                    math.hypot(src.width, src.height)
                    * 3.0
                    * max(ew_res, ns_res)
                    / step_range
                )
            ) + 2
            cand_mask = dem_valid & (shadow == 0)
            if use_compiled(
                _shadow_ray_kernel,
                "shadow_mask",
                fallback_desc="the vectorised NumPy ray tracer",
            ):
                hit = _shadow_ray_kernel(
                    np.ascontiguousarray(dem, dtype=np.float64),
                    np.ascontiguousarray(dem_valid),
                    np.ascontiguousarray(cand_mask),
                    float(col_step),
                    float(row_step),
                    float(z_step),
                    float(max_height),
                    int(max_steps),
                )
            else:
                hit = _shadow_ray_trace_numpy(
                    dem,
                    dem_valid,
                    cand_mask,
                    col_step,
                    row_step,
                    z_step,
                    float(max_height),
                    max_steps,
                    use_pbar,
                )
            shadow[hit] = 1

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        dst.write(shadow, 1)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=None,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def local_incidence_angle(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the local solar incidence angle (degrees) for each pixel of a DEM.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output incidence angle image file.
    :param azimuth: Solar azimuth angle in degrees.
    :param zenith: Solar zenith angle in degrees.
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

        import rsgispy.elevation.illumination

        rsgispy.elevation.illumination.local_incidence_angle(
            input_img="dem.tif",
            output_img="incidence.tif",
            azimuth=315.0,
            zenith=45.0,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, _res, _res_nd: _make_ray_angle_node(
            dem, azimuth, zenith, False, no_data_val=no_data_val, out_datatype=eff_dtype
        ),
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def local_exitance_angle(
    input_img: str,
    output_img: str,
    azimuth: float,
    zenith: float,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the local sensor exitance (viewing) angle (degrees) for each pixel
    of a DEM.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output exitance angle image file.
    :param azimuth: Sensor/view azimuth angle in degrees.
    :param zenith: Sensor/view zenith angle in degrees.
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

        import rsgispy.elevation.illumination

        rsgispy.elevation.illumination.local_exitance_angle(
            input_img="dem.tif",
            output_img="exitance.tif",
            azimuth=180.0,
            zenith=15.0,
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    _validate_sun_angles(azimuth, zenith)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, _res, _res_nd: _make_ray_angle_node(
            dem, azimuth, zenith, True, no_data_val=no_data_val, out_datatype=eff_dtype
        ),
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def hillshade_node(
    source: TileNode,
    azimuth: float,
    zenith: float,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`hillshade`: the hillshade (values 1-255) of a
    single-band DEM node for a sun at ``azimuth`` / ``zenith``, for chaining
    in a :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The pixel size is that of the graph grid (the DEM's geotransform; use
    :func:`hillshade_pxl_res_img_node` for geographic DEMs). The node has a
    halo radius of 1 and pads the image borders by repeating the edge
    pixels. It outputs one ``uint8`` band; pixels whose DEM value is no data
    are set to the DEM no-data value if it fits in ``uint8``, otherwise to
    ``0`` (``0`` without a no-data value too), and that is the node's no-data
    value (none when the DEM has none). Written with a
    :class:`~rsgispy.imageprocess.DiskSink` of ``rsgispy.TYPE_8UINT``, the
    output equals that of :func:`hillshade`.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]`` (e.g., 315).
    :param zenith: Solar zenith angle in degrees ``[0, 90]`` (e.g., 45).
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``, e.g. the DEM file's).
    :param name: Optional node name (Default = ``'hillshade'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    Hillshade of a smoothed DEM, and a 3-band stack of hillshades from three
    sun directions, written in one pass:

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, StackBands, TileGraph
        from rsgispy.imagefilter.stats import median_filter_node
        from rsgispy.elevation.illumination import hillshade_node

        dem = RasterSource("dem.tif", bands=1)
        smooth = median_filter_node(dem, 3, no_data_val=-9999)
        hs = StackBands([hillshade_node(smooth, az, 45.0) for az in (270, 315, 360)])
        TileGraph(DiskSink(hs, "hillshades.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    return _make_hillshade_node(
        source, None, azimuth, zenith, no_data_val=no_data_val, name=name
    )


def hillshade_pxl_res_img_node(
    source: TileNode,
    pxl_res: TileNode,
    azimuth: float,
    zenith: float,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`hillshade_pxl_res_img`: the hillshade of a
    single-band DEM node using the per-pixel resolution in bands 1
    (East-West) and 2 (North-South) of ``pxl_res``.

    As :func:`hillshade_node`, with pixels whose resolution is no data, zero
    or not finite also set to the no-data value. Both inputs must be on the
    same pixel grid.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param pxl_res: :class:`~rsgispy.imageprocess.TileNode` with at least 2
                    bands (East-West and North-South pixel resolution).
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]``.
    :param zenith: Solar zenith angle in degrees ``[0, 90]``.
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'hillshade_pxl_res_img'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.elevation.illumination import hillshade_pxl_res_img_node

        dem = RasterSource("dem_wgs84.tif", bands=1)
        res = RasterSource("pxl_res_m.tif", bands=[1, 2])
        hs = hillshade_pxl_res_img_node(dem, res, 315.0, 45.0)
        TileGraph(DiskSink(hs, "hillshade.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    # pylint: disable=too-many-arguments
    return _make_hillshade_node(
        source, pxl_res, azimuth, zenith, no_data_val=no_data_val, name=name
    )


def local_incidence_angle_node(
    source: TileNode,
    azimuth: float,
    zenith: float,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`local_incidence_angle`: the local solar
    incidence angle (degrees) of each pixel of a single-band DEM node.

    The pixel size is that of the graph grid (the DEM's geotransform). The
    node has a halo radius of 1 and pads the image borders by repeating the
    edge pixels. It outputs one ``float64`` band; pixels whose DEM value is
    no data are set to the DEM no-data value (``NaN`` when the DEM has
    none), which is the node's no-data value. Written with a
    :class:`~rsgispy.imageprocess.DiskSink` of ``rsgispy.TYPE_32FLOAT``, the
    output equals that of :func:`local_incidence_angle`.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param azimuth: Solar azimuth angle in degrees ``[0, 360]``.
    :param zenith: Solar zenith angle in degrees ``[0, 90]``.
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'local_incidence_angle'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    Cosine of the incidence angle (e.g. for a C-correction), in one pass:

    .. code:: python

        from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
        from rsgispy.elevation.illumination import local_incidence_angle_node

        dem = RasterSource("dem.tif", bands=1)
        inc = local_incidence_angle_node(dem, azimuth=150.0, zenith=35.0)
        cos_i = BandMathOp({"i": inc}, "cos(i * pi / 180)", nodata=-9999)
        TileGraph(DiskSink(cos_i, "cos_incidence.tif", nodata=-9999)).run()
    """
    return _make_ray_angle_node(
        source, azimuth, zenith, False, no_data_val=no_data_val, name=name
    )


def local_exitance_angle_node(
    source: TileNode,
    azimuth: float,
    zenith: float,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`local_exitance_angle`: the local sensor
    exitance (viewing) angle (degrees) of each pixel of a single-band DEM
    node.

    As :func:`local_incidence_angle_node` (pixels without a slope get ``0``
    rather than the zenith angle). Written with a
    :class:`~rsgispy.imageprocess.DiskSink` of ``rsgispy.TYPE_32FLOAT``, the
    output equals that of :func:`local_exitance_angle`.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param azimuth: Sensor/view azimuth angle in degrees ``[0, 360]``.
    :param zenith: Sensor/view zenith angle in degrees ``[0, 90]``.
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'local_exitance_angle'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    Incidence and exitance angles of one DEM, written as a 2-band image in
    one pass:

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, StackBands, TileGraph
        from rsgispy.elevation.illumination import (
            local_exitance_angle_node, local_incidence_angle_node,
        )

        dem = RasterSource("dem.tif", bands=1)
        angles = StackBands([
            local_incidence_angle_node(dem, 150.0, 35.0),
            local_exitance_angle_node(dem, 100.0, 8.0),
        ])
        TileGraph(DiskSink(angles, "angles.tif", band_names=["inc", "exit"])).run()
    """
    return _make_ray_angle_node(
        source, azimuth, zenith, True, no_data_val=no_data_val, name=name
    )
