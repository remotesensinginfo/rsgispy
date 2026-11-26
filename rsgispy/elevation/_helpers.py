"""
Internal shared Horn 3x3 derivative helpers and tile-graph nodes for the
terrain and illumination functions of :mod:`rsgispy.elevation`.
"""

from contextlib import ExitStack
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    create_output_from_dataset,
    iter_dataset_windows,
    read_window_with_halo,
)
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy._io import (
    DiskSink,
    HaloBorder,
    NodeMeta,
    RasterSource,
    SourceGrid,
    TileContext,
    TileGraph,
    TileNode,
    TileTensor,
)
from rsgispy._tilegraph._tensor import _repad_oob
from rsgispy.imageprocess import (
    build_valid_mask,
    ensure_file_exists,
    resolve_nodata_and_fill as _resolve_output_nodata,
)

#: Block function ``(dx_w_minus_e, dy_s_minus_n, n_vals, ew_res, ns_res)``
#: of a Horn 3x3 terrain node (``ew_res`` / ``ns_res`` are floats, or arrays
#: for a pixel-resolution image).
HornBlockFunc = Callable[[np.ndarray, np.ndarray, np.ndarray, Any, Any], np.ndarray]

_EDGE = HaloBorder("edge")


def validate_sun_angles(azimuth: float, zenith: float) -> None:
    """Validate solar/view azimuth in ``[0, 360]`` and zenith in ``[0, 90]``."""
    if zenith < 0.0 or zenith > 90.0:
        raise rsgispy.RSGISPyException(
            "The zenith angle should be between 0 and 90 degrees."
        )
    if azimuth < 0.0 or azimuth > 360.0:
        raise rsgispy.RSGISPyException(
            "The azimuth angle should be between 0 and 360 degrees."
        )


def validate_slope_unit(unit: str) -> str:
    """Validate and normalise slope output unit ('degrees' or 'radians')."""
    unit_norm = unit.strip().lower()
    if unit_norm not in ("degrees", "radians"):
        raise rsgispy.RSGISPyException(
            "The unit option needs to be specified as either 'degrees' or 'radians'."
        )
    return unit_norm


def prepare_3x3_horn_diffs(
    dem_halo: np.ndarray,
    no_data_val: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cleaned 3x3 windows (replacing isolated nodata values with window
    means when ``n_vals > 1``) and return ``(dx_w_minus_e, dy_s_minus_n, n_vals)``.

    Windows whose centre pixel is invalid receive ``n_vals == 0`` so downstream
    terrain/illumination calculations do not fabricate values over nodata cells.
    """
    win = np.lib.stride_tricks.sliding_window_view(dem_halo, (3, 3))
    valid = build_valid_mask(win, no_data_val)
    center_valid = valid[..., 1, 1]
    raw_n_vals = valid.sum(axis=(2, 3))
    n_vals = np.where(center_valid, raw_n_vals, 0)

    has_nodata = raw_n_vals < 9
    if np.any(has_nodata):
        sum_vals = np.where(valid, win, 0.0).sum(axis=(2, 3))
        mean_vals = np.divide(
            sum_vals,
            raw_n_vals,
            out=np.zeros_like(sum_vals, dtype=np.float64),
            where=raw_n_vals > 0,
        )
        fill_mask = (has_nodata & (n_vals > 1))[..., None, None]
        w_clean = np.where(valid | ~fill_mask, win, mean_vals[..., None, None])
    else:
        w_clean = win

    dx_w_minus_e = (
        w_clean[..., 0, 0] + 2.0 * w_clean[..., 1, 0] + w_clean[..., 2, 0]
    ) - (w_clean[..., 0, 2] + 2.0 * w_clean[..., 1, 2] + w_clean[..., 2, 2])
    dy_s_minus_n = (
        w_clean[..., 2, 0] + 2.0 * w_clean[..., 2, 1] + w_clean[..., 2, 2]
    ) - (w_clean[..., 0, 0] + 2.0 * w_clean[..., 0, 1] + w_clean[..., 0, 2])
    if not np.all(center_valid):
        dx_w_minus_e = np.where(center_valid, dx_w_minus_e, 0.0)
        dy_s_minus_n = np.where(center_valid, dy_s_minus_n, 0.0)
    return dx_w_minus_e, dy_s_minus_n, n_vals


def calc_slope_aspect_from_diffs(
    dx_w_minus_e: np.ndarray,
    dy_s_minus_n: np.ndarray,
    n_vals: np.ndarray,
    ew_res: Any,
    ns_res: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ``(slope_rad, aspect_deg, flat_mask)`` from 3x3 Horn differences.

    Flat areas and cells with ``n_vals <= 1`` have ``slope_rad = 0`` and
    ``aspect_deg = NaN``.
    """
    dx_s = dx_w_minus_e / ew_res
    dy_s = dy_s_minus_n / ns_res
    slope_rad = np.where(
        n_vals > 1, np.arctan(np.sqrt((dx_s * dx_s) + (dy_s * dy_s)) / 8.0), 0.0
    )

    flat_mask = (dx_s == 0.0) & (dy_s == 0.0)
    valid_asp = (n_vals > 1) & ~flat_mask
    aspect_deg = np.full(dx_s.shape, np.nan, dtype=np.float64)
    if np.any(valid_asp):
        raw_deg = np.degrees(np.arctan2(dx_s[valid_asp], dy_s[valid_asp]))
        wrapped = np.mod(raw_deg, 360.0)
        wrapped[wrapped == 360.0] = 0.0
        aspect_deg[valid_asp] = wrapped
    return slope_rad, aspect_deg, flat_mask


def pixel_size_from_ctx(ctx: TileContext, label: str) -> Tuple[float, float]:
    """
    Return the ``(ew_res, ns_res)`` pixel size of the graph grid (the
    absolute values of the transform's ``a`` and ``e`` terms, as the
    file-based functions use), raising if the grid has no valid pixel size.
    """
    if ctx.transform is None:
        raise rsgispy.RSGISPyException(
            f"{label} needs the pixel size of the DEM, but the graph grid has "
            "no transform (give the ArraySource a transform, or use the "
            "*_pxl_res_img node with a pixel-resolution input)."
        )
    ew_res = abs(float(ctx.transform.a))
    ns_res = abs(float(ctx.transform.e))
    if not (
        math.isfinite(ew_res)
        and ew_res > 0.0
        and math.isfinite(ns_res)
        and ns_res > 0.0
    ):
        raise rsgispy.RSGISPyException(
            f"Input DEM has invalid pixel resolution: ({ew_res}, {ns_res})."
        )
    return ew_res, ns_res


class Horn3x3Node(TileNode):
    """
    Tile-graph node computing a terrain or illumination layer from the Horn
    3x3 differences of a single-band DEM (halo radius 1, edge padding at the
    image borders).

    The pixel size comes from the graph grid (the DEM's transform) or, when
    ``pxl_res`` is given, from bands 1 (East-West) and 2 (North-South) of the
    pixel-resolution node. Output pixels whose DEM centre value (or
    resolution) is no data are set to the fill value
    :func:`rsgispy.imageprocess.resolve_nodata_and_fill` gives for the DEM
    no-data value and the output data type ``out_datatype``, exactly as the
    file-based functions do.

    :param dem: Single-band DEM node.
    :param block_fn: Block function (see :data:`HornBlockFunc`).
    :param out_datatype: ``rsgispy.TYPE_*`` output data type the no-data fill
                         value is resolved for.
    :param pxl_res: Optional pixel-resolution node (at least 2 bands).
    :param no_data_val: DEM no-data value (the DEM node's no-data value when
                        ``None``).
    :param res_nodata: Optional ``(ew, ns)`` no-data values of the
                       resolution bands (the resolution node's no-data value
                       when ``None``).
    :param uint8_values: ``True`` if ``block_fn`` returns ``uint8`` values
                         (hillshade): the blocks are then ``uint8`` when the
                         fill value fits in ``uint8``, otherwise ``float64``.
    :param op_desc: Parameter text for :meth:`TileGraph.describe`.
    :param name: Optional node name.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    def __init__(
        self,
        dem: TileNode,
        block_fn: HornBlockFunc,
        out_datatype: int,
        *,
        pxl_res: Optional[TileNode] = None,
        no_data_val: Optional[float] = None,
        res_nodata: Optional[Tuple[Optional[float], Optional[float]]] = None,
        uint8_values: bool = False,
        op_desc: str = "",
        name: Optional[str] = None,
    ) -> None:
        for node in (dem, pxl_res):
            if node is not None and not isinstance(node, TileNode):
                raise rsgispy.RSGISPyException(
                    "Terrain node inputs must be TileNode objects, got "
                    f"{type(node).__name__}."
                )
        inputs = (dem,) if pxl_res is None else (dem, pxl_res)
        super().__init__(inputs=inputs, name=name)
        self.block_fn = block_fn
        self.out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(int(out_datatype)))
        self.has_pxl_res = pxl_res is not None
        self.no_data_val = no_data_val
        self.res_nodata = res_nodata
        self.uint8_values = bool(uint8_values)
        self.op_desc = op_desc
        # Resolved from the inputs' metadata by infer_meta().
        self._eff_nd: Optional[float] = no_data_val
        self._res_nd: Tuple[Optional[float], Optional[float]] = (
            res_nodata if res_nodata is not None else (None, None)
        )
        self._fill: float = float("nan")
        self._block_dtype = np.dtype(np.float64)

    @property
    def halo_radius(self) -> Tuple[int, int]:
        return (1, 1)

    def input_halos(self, halo_y: int, halo_x: int) -> List[Tuple[int, int]]:
        halos = [(halo_y + 1, halo_x + 1)]
        if self.has_pxl_res:
            halos.append((halo_y, halo_x))
        return halos

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        dem_meta = inputs[0]
        if dem_meta.n_bands is not None and dem_meta.n_bands != 1:
            raise rsgispy.RSGISPyException(
                f"{self.label}: the DEM input must have 1 band, got "
                f"{dem_meta.n_bands} (select one with SelectBands)."
            )
        if self.has_pxl_res:
            res_meta = inputs[1]
            if res_meta.n_bands is not None and res_meta.n_bands < 2:
                raise rsgispy.RSGISPyException(
                    "in_pxl_res_img must contain at least 2 bands "
                    "(EW and NS resolution)."
                )
            if self.res_nodata is None:
                self._res_nd = (res_meta.nodata, res_meta.nodata)
        self._eff_nd = (
            self.no_data_val if self.no_data_val is not None else dem_meta.nodata
        )
        out_nd, fill = _resolve_output_nodata(self._eff_nd, self.out_np_dtype)
        self._fill = fill
        if (
            self.uint8_values
            and math.isfinite(fill)
            and float(fill).is_integer()
            and 0.0 <= fill <= 255.0
        ):
            self._block_dtype = np.dtype(np.uint8)
        else:
            self._block_dtype = np.dtype(np.float64)
        return NodeMeta(n_bands=1, dtype=self._block_dtype, nodata=out_nd)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # pylint: disable=too-many-locals
        eff_nd = self._eff_nd
        dem_t = inputs[0].crop(halo_y + 1, halo_x + 1)
        dem_halo = _repad_oob(dem_t, _EDGE)
        if dem_halo.shape[0] != 1:
            raise rsgispy.RSGISPyException(
                f"{self.label}: the DEM input must have 1 band, got "
                f"{dem_halo.shape[0]}."
            )
        dem_halo = dem_halo[0].astype(np.float64, copy=False)
        dx, dy, n_vals = prepare_3x3_horn_diffs(dem_halo, eff_nd)
        center_valid = build_valid_mask(dem_halo[1:-1, 1:-1], eff_nd)
        if self.has_pxl_res:
            res_blk = inputs[1].crop(halo_y, halo_x).data
            if res_blk.shape[0] < 2:
                raise rsgispy.RSGISPyException(
                    "in_pxl_res_img must contain at least 2 bands "
                    "(EW and NS resolution)."
                )
            raw_ew = res_blk[0].astype(np.float64, copy=False)
            raw_ns = res_blk[1].astype(np.float64, copy=False)
            ew_valid = build_valid_mask(raw_ew, self._res_nd[0]) & (raw_ew != 0.0)
            ns_valid = build_valid_mask(raw_ns, self._res_nd[1]) & (raw_ns != 0.0)
            res_valid = ew_valid & ns_valid
            ew_res: Any = np.where(res_valid, np.abs(raw_ew), 1.0)
            ns_res: Any = np.where(res_valid, np.abs(raw_ns), 1.0)
            center_valid = center_valid & res_valid
            n_vals = np.where(res_valid, n_vals, 0)
        else:
            ew_res, ns_res = pixel_size_from_ctx(ctx, self.label)
        block_out = self.block_fn(dx, dy, n_vals, ew_res, ns_res)
        if not np.all(center_valid):
            block_out = np.where(center_valid, block_out, self._fill)
        block_out = np.asarray(block_out).astype(self._block_dtype, copy=False)
        return TileTensor(
            data=block_out[np.newaxis],
            halo_y=halo_y,
            halo_x=halo_x,
            oob_top=max(0, dem_t.oob_top - 1),
            oob_bottom=max(0, dem_t.oob_bottom - 1),
            oob_left=max(0, dem_t.oob_left - 1),
            oob_right=max(0, dem_t.oob_right - 1),
        )

    def describe_params(self) -> str:
        nd = "from input" if self.no_data_val is None else repr(self.no_data_val)
        res = " pxl_res=input 2" if self.has_pxl_res else ""
        return f"{self.op_desc} no_data_val={nd}{res}".strip()


class _ResolutionSource(RasterSource):
    """
    :class:`RasterSource` for the pixel-resolution image of the file-based
    ``*_pxl_res_img`` functions, which (as before they were tile-graph
    wrappers) only require the image to have the DEM's dimensions: its
    transform and CRS are not compared with the DEM's.
    """

    class _Reader:
        """Reader proxy without a ``dataset`` attribute (no grid re-check)."""

        # pylint: disable=too-few-public-methods

        def __init__(self, reader: Any) -> None:
            self.read = reader.read

    def source_info(self, headers: Any) -> Tuple[SourceGrid, NodeMeta]:
        grid, meta = super().source_info(headers)
        return SourceGrid(height=grid.height, width=grid.width, label=grid.label), meta

    def start_run(self, resources: Any, halo_y: int, halo_x: int) -> Any:
        return self._Reader(super().start_run(resources, halo_y, halo_x))


TerrainNodeFactory = Callable[
    [TileNode, Optional[TileNode], Optional[Tuple[Any, Any]]], TileNode
]


def run_terrain_node_to_file(
    make_node: TerrainNodeFactory,
    input_img: str,
    output_img: str,
    gdalformat: str,
    out_datatype: int,
    *,
    in_pxl_res_img: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Run a terrain node as a file-based function:
    ``RasterSource -> node -> DiskSink`` in a :class:`TileGraph` with the
    package default block size.

    ``make_node(dem_source, pxl_res_source, res_nodata)`` builds the node
    (``pxl_res_source`` and ``res_nodata`` are ``None`` without
    ``in_pxl_res_img``). The output has one band of ``out_datatype``, no band
    names, the no-data value the node resolved, and (with ``calc_stats``)
    statistics and pyramids.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(
        input_img, "Input image" if in_pxl_res_img is None else "Input DEM image"
    )
    res_src: Optional[TileNode] = None
    res_nodata: Optional[Tuple[Any, Any]] = None
    if in_pxl_res_img is not None:
        ensure_file_exists(in_pxl_res_img, "Input pixel resolution image")
        with rasterio.open(input_img, "r") as dem_ds, rasterio.open(
            in_pxl_res_img, "r"
        ) as res_ds:
            if res_ds.count < 2:
                raise rsgispy.RSGISPyException(
                    "in_pxl_res_img must contain at least 2 bands "
                    "(EW and NS resolution)."
                )
            if dem_ds.width != res_ds.width or dem_ds.height != res_ds.height:
                raise rsgispy.RSGISPyException(
                    "Input DEM and pixel resolution images must have matching "
                    "dimensions."
                )
            nd_vals = res_ds.nodatavals
            res_nodata = (
                nd_vals[0] if nd_vals else res_ds.nodata,
                nd_vals[1] if nd_vals and len(nd_vals) >= 2 else res_ds.nodata,
            )
        res_src = _ResolutionSource(in_pxl_res_img, bands=[1, 2])
    dem_src = RasterSource(input_img, bands=1)
    node = make_node(dem_src, res_src, res_nodata)
    sink = DiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=out_datatype,
        clamp_int=False,
        calc_stats=calc_stats,
        inherit_meta=False,
    )
    graph = TileGraph(sink, block_size=None)
    # The no-data value written is the one the node resolved for the output
    # data type (the DEM no-data value, 0 when it does not fit an integer
    # type, or none).
    sink.nodata = graph.node_meta[node].nodata
    graph.run(use_pbar=use_pbar)


def process_single_dem_3x3(
    input_img: str,
    output_img: str,
    gdalformat: str,
    out_datatype: int,
    block_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray],
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Run a 3x3 windowed calculation block-by-block over a single DEM raster."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    out_rio_dtype = rsgispy.get_rasterio_datatype(out_datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(out_datatype))

    out_nd: Optional[float] = None
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        ew_res = abs(float(src.transform.a))
        ns_res = abs(float(src.transform.e))
        if not (
            math.isfinite(ew_res)
            and ew_res > 0.0
            and math.isfinite(ns_res)
            and ns_res > 0.0
        ):
            raise rsgispy.RSGISPyException(
                f"Input DEM has invalid pixel resolution: ({ew_res}, {ns_res})."
            )
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        out_nd, fill_nd = _resolve_output_nodata(eff_nd, out_np_dtype)

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )
        if out_nd is not None:
            dst.nodata = out_nd

        for out_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            dem_halo = read_window_with_halo(src, 1, out_win, pad=1)
            dx, dy, n_vals = prepare_3x3_horn_diffs(dem_halo, eff_nd)
            center_valid = build_valid_mask(dem_halo[1:-1, 1:-1], eff_nd)
            block_out = block_fn(dx, dy, n_vals, ew_res, ns_res)
            if not np.all(center_valid):
                block_out = np.where(center_valid, block_out, fill_nd)
            out_arr = cast_result_block(
                block_out,
                int(out_win.height),
                int(out_win.width),
                out_np_dtype,
            )
            dst.write(out_arr, 1, window=out_win)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )


def process_pxl_res_dem_3x3(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    gdalformat: str,
    out_datatype: int,
    block_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ],
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Run a 3x3 windowed calculation using per-pixel E-W/N-S resolution bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_dem_img, "Input DEM image")
    ensure_file_exists(in_pxl_res_img, "Input pixel resolution image")
    out_rio_dtype = rsgispy.get_rasterio_datatype(out_datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(out_datatype))

    out_nd: Optional[float] = None
    with ExitStack() as stack:
        dem_src = stack.enter_context(rasterio.open(in_dem_img, "r"))
        res_src = stack.enter_context(rasterio.open(in_pxl_res_img, "r"))
        if res_src.count < 2:
            raise rsgispy.RSGISPyException(
                "in_pxl_res_img must contain at least 2 bands (EW and NS resolution)."
            )
        if dem_src.width != res_src.width or dem_src.height != res_src.height:
            raise rsgispy.RSGISPyException(
                "Input DEM and pixel resolution images must have matching dimensions."
            )

        eff_nd = no_data_val if no_data_val is not None else dem_src.nodata
        out_nd, fill_nd = _resolve_output_nodata(eff_nd, out_np_dtype)
        res_nd_ew = res_src.nodatavals[0] if res_src.nodatavals else res_src.nodata
        res_nd_ns = (
            res_src.nodatavals[1]
            if res_src.nodatavals and len(res_src.nodatavals) >= 2
            else res_src.nodata
        )

        dst = create_output_from_dataset(
            stack=stack,
            src=dem_src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=1,
        )
        if out_nd is not None:
            dst.nodata = out_nd

        for out_win in iter_dataset_windows(
            dem_src.height, dem_src.width, use_pbar=use_pbar
        ):
            dem_halo = read_window_with_halo(dem_src, 1, out_win, pad=1)
            dx, dy, n_vals = prepare_3x3_horn_diffs(dem_halo, eff_nd)
            raw_ew = res_src.read(1, window=out_win, out_dtype=np.float64)
            raw_ns = res_src.read(2, window=out_win, out_dtype=np.float64)
            ew_valid = build_valid_mask(raw_ew, res_nd_ew) & (raw_ew != 0.0)
            ns_valid = build_valid_mask(raw_ns, res_nd_ns) & (raw_ns != 0.0)
            res_valid = ew_valid & ns_valid

            ew_res = np.where(res_valid, np.abs(raw_ew), 1.0)
            ns_res = np.where(res_valid, np.abs(raw_ns), 1.0)
            center_valid = (
                build_valid_mask(dem_halo[1:-1, 1:-1], eff_nd) & res_valid
            )
            eff_n_vals = np.where(res_valid, n_vals, 0)

            block_out = block_fn(dx, dy, eff_n_vals, ew_res, ns_res)
            if not np.all(center_valid):
                block_out = np.where(center_valid, block_out, fill_nd)
            out_arr = cast_result_block(
                block_out,
                int(out_win.height),
                int(out_win.width),
                out_np_dtype,
            )
            dst.write(out_arr, 1, window=out_win)

    if calc_stats:
        pop_img_stats(
            input_img=output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            ignore_cog=gdalformat.upper().strip() == "COG",
            use_pbar=use_pbar,
        )
