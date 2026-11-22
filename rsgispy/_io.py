"""
Composable in-memory, halo-fused tile-graph raster execution engine for
:mod:`rsgispy`.

This module allows multi-stage raster workflows — including chains of
moving-window spatial filters, morphological operators, skip connections, and
point-wise / ``pymu-eval`` band-math expressions — to execute per tile in
memory before writing final output tile(s) once to disk.

Key guarantees
--------------
* **Cumulative halo propagation**: reverse topological traversal computes the
  exact cumulative halo radius ``(sum_ry, sum_rx)`` required at every upstream
  node so interior pixels of every stage are unaffected by tile boundaries.
* **Bit-exact image-border semantics**: each intermediate tensor tracks its
  exact out-of-image border overhang ``(oob_top, oob_bottom, oob_left,
  oob_right)`` and re-pads out-of-bounds margins using each :class:`WindowOp`'s
  configured :class:`HaloBorder` before evaluation, matching multi-pass disk
  execution even at image edges.
* **Skip connections & multi-sink fan-out**: nodes with multiple consumers
  compute the maximum required halo across consumers; downstream
  :class:`PointwiseOp`, :class:`BandMathOp`, and sinks crop mismatched halos
  symmetrically to align pixel-for-pixel.
* **Low peak memory**: intermediate tile tensors are released via per-tile
  reference counting as soon as their last consumer finishes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pymu_eval
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy

__all__ = [
    "HaloBorder",
    "TileContext",
    "TileNode",
    "RasterSource",
    "ArraySource",
    "WindowOp",
    "PointwiseOp",
    "BandMathOp",
    "DiskSink",
    "MemorySink",
    "TileGraph",
    "run_tile_graph",
]

_VALID_BORDER_MODES = ("edge", "constant", "reflect", "symmetric")


@dataclass(frozen=True)
class HaloBorder:
    """
    Border padding policy applied outside the image boundary for a
    :class:`RasterSource`, :class:`ArraySource`, or :class:`WindowOp`.

    :param mode: NumPy padding mode: ``'edge'`` (default), ``'constant'``,
                 ``'reflect'``, or ``'symmetric'``.
    :param fill_value: Constant fill value used when ``mode == 'constant'``
                       (Default = ``0.0``).
    """

    mode: str = "edge"
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in _VALID_BORDER_MODES:
            raise rsgispy.RSGISPyException(
                f"Unsupported HaloBorder mode '{self.mode}'. "
                f"Choose from {_VALID_BORDER_MODES}."
            )


@dataclass(frozen=True)
class TileContext:
    """
    Spatial metadata describing the current tile being evaluated by a
    :class:`TileGraph` node.
    """

    row_off: int
    col_off: int
    height: int
    width: int
    img_height: int
    img_width: int
    halo_y: int
    halo_x: int
    oob_top: int
    oob_bottom: int
    oob_left: int
    oob_right: int
    transform: Optional[Affine] = None
    crs: Any = None
    nodata: Optional[float] = None


@dataclass
class _TileTensor:
    """Internal 3D ``(bands, H + 2*hy, W + 2*hx)`` tile tensor with halo/OOB metadata."""

    data: np.ndarray
    halo_y: int
    halo_x: int
    oob_top: int
    oob_bottom: int
    oob_left: int
    oob_right: int


def _normalise_radius(
    radius: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """Normalise an integer or ``(ry, rx)`` halo radius to non-negative ints."""
    if isinstance(radius, tuple):
        if len(radius) != 2:
            raise rsgispy.RSGISPyException(
                f"radius tuple must have 2 elements (ry, rx), got {radius!r}."
            )
        ry, rx = int(radius[0]), int(radius[1])
    else:
        ry = rx = int(radius)
    if ry < 0 or rx < 0:
        raise rsgispy.RSGISPyException(
            f"Halo radius must be >= 0, got {(ry, rx)!r}."
        )
    return (ry, rx)


def _pad_3d(
    arr: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border: HaloBorder,
) -> np.ndarray:
    """Pad a 3D ``(C, H, W)`` array on spatial axes using ``border``."""
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return arr
    pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    if border.mode == "constant":
        return np.pad(
            arr,
            pad_width=pad_width,
            mode="constant",
            constant_values=border.fill_value,
        )
    return np.pad(arr, pad_width=pad_width, mode=border.mode)  # type: ignore[arg-type]


def _crop_tensor(
    tensor: _TileTensor,
    target_hy: int,
    target_hx: int,
) -> _TileTensor:
    """Symmetrically crop ``tensor`` down to ``(target_hy, target_hx)``."""
    dy = tensor.halo_y - target_hy
    dx = tensor.halo_x - target_hx
    if dy < 0 or dx < 0:
        raise rsgispy.RSGISPyException(
            f"Cannot expand tile halo from ({tensor.halo_y}, {tensor.halo_x}) "
            f"to ({target_hy}, {target_hx})."
        )
    if dy == 0 and dx == 0:
        return tensor
    arr = tensor.data
    h_end = arr.shape[1] - dy if dy > 0 else arr.shape[1]
    w_end = arr.shape[2] - dx if dx > 0 else arr.shape[2]
    cropped = arr[:, dy:h_end, dx:w_end]
    return _TileTensor(
        data=cropped,
        halo_y=target_hy,
        halo_x=target_hx,
        oob_top=max(0, tensor.oob_top - dy),
        oob_bottom=max(0, tensor.oob_bottom - dy),
        oob_left=max(0, tensor.oob_left - dx),
        oob_right=max(0, tensor.oob_right - dx),
    )


def _repad_oob(tensor: _TileTensor, border: HaloBorder) -> np.ndarray:
    """
    Re-apply ``border`` padding strictly to out-of-image overhang margins so
    chained window operations match multi-pass disk execution at image edges.
    """
    ot = tensor.oob_top
    ob = tensor.oob_bottom
    ol = tensor.oob_left
    or_ = tensor.oob_right
    if ot == 0 and ob == 0 and ol == 0 and or_ == 0:
        return tensor.data
    arr = tensor.data
    y1 = arr.shape[1] - ob if ob > 0 else arr.shape[1]
    x1 = arr.shape[2] - or_ if or_ > 0 else arr.shape[2]
    valid_interior = arr[:, ot:y1, ol:x1]
    return _pad_3d(valid_interior, ot, ob, ol, or_, border)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Ensure a node output array has 3D ``(C, H, W)`` shape."""
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise rsgispy.RSGISPyException(
        f"Tile node output must be 2D or 3D, got shape {arr.shape}."
    )


class TileNode:
    """Base class for nodes in a :class:`TileGraph`."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        inputs: Sequence["TileNode"] = (),
        name: Optional[str] = None,
    ) -> None:
        self.inputs: Tuple[TileNode, ...] = tuple(inputs)
        self.name: Optional[str] = name


class RasterSource(TileNode):
    """
    Source node reading tiled blocks with halo padding from an on-disk raster.

    :param path: Input raster file path.
    :param bands: Optional 1-based band index or sequence of band indices to
                  read (reads all bands when ``None``).
    :param dtype: NumPy dtype to cast read blocks into (Default = ``np.float64``).
    :param border: Border padding policy outside image bounds
                   (Default = ``HaloBorder('edge')``).
    :param nodata: Optional override for the source nodata value (uses raster
                   metadata nodata when ``None``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods

    def __init__(
        self,
        path: Union[str, os.PathLike],
        bands: Optional[Union[int, Sequence[int]]] = None,
        dtype: Any = np.float64,
        border: Optional[HaloBorder] = None,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(), name=name)
        self.path: str = os.fspath(path)
        if bands is None:
            self.bands: Optional[List[int]] = None
        elif isinstance(bands, int):
            self.bands = [int(bands)]
        else:
            self.bands = [int(b) for b in bands]
        self.dtype = np.dtype(dtype)
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.nodata: Optional[float] = nodata


class ArraySource(TileNode):
    """
    Source node reading tiled blocks with halo padding from an in-memory NumPy
    array of shape ``(H, W)`` or ``(C, H, W)``.

    :param array: Input 2D ``(H, W)`` or 3D ``(C, H, W)`` array.
    :param transform: Optional affine geotransform for downstream sinks.
    :param crs: Optional CRS for downstream sinks.
    :param border: Border padding policy outside array bounds
                   (Default = ``HaloBorder('edge')``).
    :param nodata: Optional nodata value associated with ``array``.
    :param dtype: Optional NumPy dtype to cast tiles into (preserves array
                  dtype when ``None``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods

    def __init__(
        self,
        array: np.ndarray,
        transform: Optional[Affine] = None,
        crs: Any = None,
        border: Optional[HaloBorder] = None,
        nodata: Optional[float] = None,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(), name=name)
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise rsgispy.RSGISPyException(
                f"ArraySource expects a 2D or 3D array, got shape {arr.shape}."
            )
        self.array: np.ndarray = arr
        self.transform: Optional[Affine] = transform
        self.crs: Any = crs
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.nodata: Optional[float] = nodata
        self.dtype: np.dtype = np.dtype(dtype) if dtype is not None else arr.dtype


class WindowOp(TileNode):
    """
    Moving-window neighbourhood operation consuming a single upstream node
    with halo radius ``radius``.

    ``func`` receives a 3D array of shape ``(C, H_out + 2*ry, W_out + 2*rx)``
    (and optionally a :class:`TileContext` when ``pass_ctx=True``) whose
    out-of-image margins have already been padded using ``border``.  ``func``
    may return either:

    * a valid-cropped array of shape ``(C_out, H_out, W_out)``, or
    * a same-size array of shape ``(C_out, H_out + 2*ry, W_out + 2*rx)``,
      which is automatically cropped by ``(ry, rx)``.

    :param source: Upstream :class:`TileNode`.
    :param radius: Non-negative integer or ``(ry, rx)`` halo radius.
    :param func: Callable transforming the padded tile array.
    :param border: Border padding policy applied to out-of-image edges before
                   calling ``func`` (Default = ``HaloBorder('edge')``).
    :param pass_ctx: If ``True``, pass ``(arr, ctx)`` to ``func``; otherwise
                     pass ``(arr,)`` (Default = ``False``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods

    def __init__(
        self,
        source: TileNode,
        radius: Union[int, Tuple[int, int]],
        func: Callable[..., np.ndarray],
        border: Optional[HaloBorder] = None,
        pass_ctx: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.radius_y, self.radius_x = _normalise_radius(radius)
        self.func = func
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.pass_ctx: bool = bool(pass_ctx)


class PointwiseOp(TileNode):
    """
    Pixel-wise operation combining one or more upstream :class:`TileNode`
    objects at identical spatial alignment (automatically cropping mismatched
    halos for skip connections).

    :param inputs: Single :class:`TileNode` or sequence of :class:`TileNode`s.
    :param func: Callable receiving unpacked 3D tile arrays ``(*arrays)`` (or
                 ``(*arrays, ctx)`` when ``pass_ctx=True``).
    :param pass_ctx: If ``True``, append :class:`TileContext` as the final
                     argument to ``func`` (Default = ``False``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        inputs: Union[TileNode, Sequence[TileNode]],
        func: Callable[..., np.ndarray],
        pass_ctx: bool = False,
        name: Optional[str] = None,
    ) -> None:
        node_list = (inputs,) if isinstance(inputs, TileNode) else tuple(inputs)
        if not node_list:
            raise rsgispy.RSGISPyException(
                "PointwiseOp requires at least one upstream input node."
            )
        super().__init__(inputs=node_list, name=name)
        self.func = func
        self.pass_ctx: bool = bool(pass_ctx)


class BandMathOp(TileNode):
    """
    Point-wise band-math expression evaluated via ``pymu-eval`` over named
    upstream :class:`TileNode` inputs.

    When all upstream nodes have the same number of bands ``C``, the expression
    is evaluated band-by-band producing ``C`` output bands.  Single-band nodes
    broadcast automatically against multi-band nodes.

    :param inputs: Mapping of ``{var_name: TileNode}`` referenced in ``expr``.
    :param expr: ``pymu-eval`` expression string (e.g. ``'(nir - red) / (nir + red)'``
                 or ``'raw - smooth'``).
    :param nodata: Optional nodata value; when set, pixels matching ``nodata``
                   in any input are preserved as ``nodata`` in the output.
    :param name: Optional node identifier.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        inputs: Mapping[str, TileNode],
        expr: str,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        if not inputs:
            raise rsgispy.RSGISPyException(
                "BandMathOp requires a non-empty inputs mapping."
            )
        if not expr or not expr.strip():
            raise rsgispy.RSGISPyException("BandMathOp expr must be non-empty.")
        self.var_names: Tuple[str, ...] = tuple(inputs.keys())
        super().__init__(inputs=tuple(inputs.values()), name=name)
        self.expr: str = str(expr)
        self.nodata: Optional[float] = nodata


class DiskSink(TileNode):
    """
    Terminal sink writing a node's tiled output to an on-disk raster dataset.

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param output_img: Output raster file path.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant (Default = ``TYPE_32FLOAT``).
    :param nodata: Optional output nodata value.
    :param clamp_int: Clamp values to integer type bounds before casting
                      (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    :param band_names: Optional sequence of output band description strings.
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       after writing (Default = ``False``).
    :param name: Optional sink identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods

    def __init__(
        self,
        source: TileNode,
        output_img: Union[str, os.PathLike],
        gdalformat: str = "GTIFF",
        datatype: int = rsgispy.TYPE_32FLOAT,
        nodata: Optional[float] = None,
        clamp_int: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
        band_names: Optional[Sequence[str]] = None,
        calc_stats: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.output_img: str = os.fspath(output_img)
        self.gdalformat: str = str(gdalformat)
        self.datatype: int = int(datatype)
        self.nodata: Optional[float] = nodata
        self.clamp_int: bool = bool(clamp_int)
        self.creation_opts: Optional[Mapping[str, Any]] = creation_opts
        self.band_names: Optional[List[str]] = (
            list(band_names) if band_names is not None else None
        )
        self.calc_stats: bool = bool(calc_stats)


class MemorySink(TileNode):
    """
    Terminal sink assembling a node's tiled output into an in-memory NumPy
    array accessible on :attr:`result` after :meth:`TileGraph.run`.

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param dtype: Optional output NumPy dtype (inferred from first tile when
                  ``None``).
    :param clamp_int: Clamp values to integer bounds when casting
                      (Default = ``True``).
    :param name: Optional sink identifier (used as key in
                 :meth:`TileGraph.run` return dictionary).
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        source: TileNode,
        dtype: Any = None,
        clamp_int: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.dtype: Optional[np.dtype] = np.dtype(dtype) if dtype is not None else None
        self.clamp_int: bool = bool(clamp_int)
        self.result: Optional[np.ndarray] = None


class TileGraph:
    """
    Directed acyclic graph of raster tile operations executed tile-by-tile
    in memory with automatic cumulative halo propagation.

    :param sinks: Single sink (:class:`DiskSink` or :class:`MemorySink`) or
                  sequence of sinks defining graph outputs.
    :param block_size: Core processing tile size in pixels (Default = ``256``).
    """

    def __init__(
        self,
        sinks: Union[DiskSink, MemorySink, Sequence[Union[DiskSink, MemorySink]]],
        block_size: int = 256,
    ) -> None:
        if isinstance(sinks, (DiskSink, MemorySink)):
            sink_list: List[Union[DiskSink, MemorySink]] = [sinks]
        else:
            sink_list = list(sinks)
        if not sink_list:
            raise rsgispy.RSGISPyException(
                "TileGraph requires at least one DiskSink or MemorySink."
            )
        for sink in sink_list:
            if not isinstance(sink, (DiskSink, MemorySink)):
                raise rsgispy.RSGISPyException(
                    f"TileGraph sinks must be DiskSink or MemorySink, got {type(sink).__name__}."
                )
        if block_size <= 0:
            raise rsgispy.RSGISPyException(
                f"block_size must be > 0, got {block_size}."
            )
        self.sinks: Tuple[Union[DiskSink, MemorySink], ...] = tuple(sink_list)
        self.block_size: int = int(block_size)
        self.topo_order: Tuple[TileNode, ...] = self._build_topo_order()
        self.required_halos: Dict[TileNode, Tuple[int, int]] = (
            self._compute_required_halos()
        )
        self.consumer_counts: Dict[TileNode, int] = self._compute_consumer_counts()

    def _build_topo_order(self) -> Tuple[TileNode, ...]:
        """Topologically sort all reachable nodes and reject cycles."""
        visited: Dict[TileNode, int] = {}
        order: List[TileNode] = []

        def _dfs(node: TileNode) -> None:
            state = visited.get(node, 0)
            if state == 1:
                raise rsgispy.RSGISPyException(
                    "Cycle detected in TileGraph definition."
                )
            if state == 2:
                return
            visited[node] = 1
            for parent in node.inputs:
                _dfs(parent)
            visited[node] = 2
            order.append(node)

        for sink in self.sinks:
            _dfs(sink)
        return tuple(order)

    def _compute_required_halos(self) -> Dict[TileNode, Tuple[int, int]]:
        """Propagate cumulative halo radii backward from sinks to sources."""
        req: Dict[TileNode, Tuple[int, int]] = {
            node: (0, 0) for node in self.topo_order
        }
        for node in reversed(self.topo_order):
            hy, hx = req[node]
            if isinstance(node, WindowOp):
                parent = node.inputs[0]
                phy, phx = req[parent]
                req[parent] = (
                    max(phy, hy + node.radius_y),
                    max(phx, hx + node.radius_x),
                )
            elif isinstance(node, (PointwiseOp, BandMathOp, DiskSink, MemorySink)):
                for parent in node.inputs:
                    phy, phx = req[parent]
                    req[parent] = (max(phy, hy), max(phx, hx))
        return req

    def _compute_consumer_counts(self) -> Dict[TileNode, int]:
        """Count downstream consumers per node for eager per-tile tensor freeing."""
        counts: Dict[TileNode, int] = {node: 0 for node in self.topo_order}
        for node in self.topo_order:
            for parent in node.inputs:
                counts[parent] += 1
        return counts

    def _read_raster_source(
        self,
        node: RasterSource,
        ds: rasterio.io.DatasetReader,
        win: Window,
        hy: int,
        hx: int,
    ) -> _TileTensor:
        """Read a halo-padded block from ``ds`` tracking exact OOB overhangs."""
        r0 = int(win.row_off)
        c0 = int(win.col_off)
        r1 = r0 + int(win.height)
        c1 = c0 + int(win.width)
        img_h = int(ds.height)
        img_w = int(ds.width)

        read_r0 = max(0, r0 - hy)
        read_c0 = max(0, c0 - hx)
        read_r1 = min(img_h, r1 + hy)
        read_c1 = min(img_w, c1 + hx)

        oob_top = max(0, hy - r0)
        oob_bottom = max(0, (r1 + hy) - img_h)
        oob_left = max(0, hx - c0)
        oob_right = max(0, (c1 + hx) - img_w)

        read_win = Window(
            read_c0, read_r0, read_c1 - read_c0, read_r1 - read_r0
        )
        if node.bands is None:
            raw = ds.read(window=read_win).astype(node.dtype, copy=False)
        else:
            raw = ds.read(node.bands, window=read_win).astype(
                node.dtype, copy=False
            )
        padded = _pad_3d(
            raw, oob_top, oob_bottom, oob_left, oob_right, node.border
        )
        return _TileTensor(
            data=padded,
            halo_y=hy,
            halo_x=hx,
            oob_top=oob_top,
            oob_bottom=oob_bottom,
            oob_left=oob_left,
            oob_right=oob_right,
        )

    def _read_array_source(
        self,
        node: ArraySource,
        win: Window,
        hy: int,
        hx: int,
    ) -> _TileTensor:
        """Slice a halo-padded block from an in-memory ``ArraySource``."""
        r0 = int(win.row_off)
        c0 = int(win.col_off)
        r1 = r0 + int(win.height)
        c1 = c0 + int(win.width)
        img_h = int(node.array.shape[1])
        img_w = int(node.array.shape[2])

        read_r0 = max(0, r0 - hy)
        read_c0 = max(0, c0 - hx)
        read_r1 = min(img_h, r1 + hy)
        read_c1 = min(img_w, c1 + hx)

        oob_top = max(0, hy - r0)
        oob_bottom = max(0, (r1 + hy) - img_h)
        oob_left = max(0, hx - c0)
        oob_right = max(0, (c1 + hx) - img_w)

        raw = node.array[:, read_r0:read_r1, read_c0:read_c1].astype(
            node.dtype, copy=False
        )
        padded = _pad_3d(
            raw, oob_top, oob_bottom, oob_left, oob_right, node.border
        )
        return _TileTensor(
            data=padded,
            halo_y=hy,
            halo_x=hx,
            oob_top=oob_top,
            oob_bottom=oob_bottom,
            oob_left=oob_left,
            oob_right=oob_right,
        )

    def _eval_window_op(
        self,
        node: WindowOp,
        in_tensor: _TileTensor,
        out_hy: int,
        out_hx: int,
        base_ctx: TileContext,
    ) -> _TileTensor:
        """Evaluate a :class:`WindowOp` with exact OOB edge re-padding."""
        req_hy = out_hy + node.radius_y
        req_hx = out_hx + node.radius_x
        aligned = _crop_tensor(in_tensor, req_hy, req_hx)
        padded = _repad_oob(aligned, node.border)

        if node.pass_ctx:
            ctx = TileContext(
                row_off=base_ctx.row_off,
                col_off=base_ctx.col_off,
                height=base_ctx.height,
                width=base_ctx.width,
                img_height=base_ctx.img_height,
                img_width=base_ctx.img_width,
                halo_y=req_hy,
                halo_x=req_hx,
                oob_top=aligned.oob_top,
                oob_bottom=aligned.oob_bottom,
                oob_left=aligned.oob_left,
                oob_right=aligned.oob_right,
                transform=base_ctx.transform,
                crs=base_ctx.crs,
                nodata=base_ctx.nodata,
            )
            res = _ensure_3d(node.func(padded, ctx))
        else:
            res = _ensure_3d(node.func(padded))

        expected_h = base_ctx.height + 2 * out_hy
        expected_w = base_ctx.width + 2 * out_hx
        in_h, in_w = padded.shape[1], padded.shape[2]
        if res.shape[1] == in_h and res.shape[2] == in_w and (
            node.radius_y > 0 or node.radius_x > 0
        ):
            y1 = in_h - node.radius_y if node.radius_y > 0 else in_h
            x1 = in_w - node.radius_x if node.radius_x > 0 else in_w
            res = res[:, node.radius_y:y1, node.radius_x:x1]
        elif res.shape[1] != expected_h or res.shape[2] != expected_w:
            raise rsgispy.RSGISPyException(
                f"WindowOp returned unexpected spatial shape {res.shape[1:]}, "
                f"expected {(expected_h, expected_w)} or {(in_h, in_w)}."
            )

        return _TileTensor(
            data=res,
            halo_y=out_hy,
            halo_x=out_hx,
            oob_top=max(0, aligned.oob_top - node.radius_y),
            oob_bottom=max(0, aligned.oob_bottom - node.radius_y),
            oob_left=max(0, aligned.oob_left - node.radius_x),
            oob_right=max(0, aligned.oob_right - node.radius_x),
        )

    def _eval_pointwise_op(
        self,
        node: PointwiseOp,
        in_tensors: Sequence[_TileTensor],
        out_hy: int,
        out_hx: int,
        base_ctx: TileContext,
    ) -> _TileTensor:
        """Evaluate a :class:`PointwiseOp` after aligning input halos."""
        aligned = [_crop_tensor(t, out_hy, out_hx) for t in in_tensors]
        arrays = [t.data for t in aligned]
        first = aligned[0]
        if node.pass_ctx:
            ctx = TileContext(
                row_off=base_ctx.row_off,
                col_off=base_ctx.col_off,
                height=base_ctx.height,
                width=base_ctx.width,
                img_height=base_ctx.img_height,
                img_width=base_ctx.img_width,
                halo_y=out_hy,
                halo_x=out_hx,
                oob_top=first.oob_top,
                oob_bottom=first.oob_bottom,
                oob_left=first.oob_left,
                oob_right=first.oob_right,
                transform=base_ctx.transform,
                crs=base_ctx.crs,
                nodata=base_ctx.nodata,
            )
            res = _ensure_3d(node.func(*arrays, ctx))
        else:
            res = _ensure_3d(node.func(*arrays))
        return _TileTensor(
            data=res,
            halo_y=out_hy,
            halo_x=out_hx,
            oob_top=first.oob_top,
            oob_bottom=first.oob_bottom,
            oob_left=first.oob_left,
            oob_right=first.oob_right,
        )

    def _eval_band_math_op(
        self,
        node: BandMathOp,
        in_tensors: Sequence[_TileTensor],
        out_hy: int,
        out_hx: int,
        parsers: Dict[BandMathOp, pymu_eval.Parser],
    ) -> _TileTensor:
        """Evaluate a ``pymu-eval`` :class:`BandMathOp` across bands."""
        aligned = [_crop_tensor(t, out_hy, out_hx) for t in in_tensors]
        first = aligned[0]
        band_counts = [t.data.shape[0] for t in aligned]
        n_bands = max(band_counts)
        for cnt in band_counts:
            if cnt not in (1, n_bands):
                raise rsgispy.RSGISPyException(
                    f"BandMathOp inputs have incompatible band counts: {band_counts}."
                )

        parser = parsers.get(node)
        needs_parse = parser is None
        if parser is None:
            parser = pymu_eval.Parser()
            var_name_set = set(node.var_names)
            if "pi" not in var_name_set:
                parser.define_const("pi", pymu_eval.MathImpl.CONST_PI)
            if "e" not in var_name_set:
                parser.define_const("e", pymu_eval.MathImpl.CONST_E)

        out_bands: List[np.ndarray] = []
        for b_idx in range(n_bands):
            invalid_mask: Optional[np.ndarray] = None
            for var_name, tensor in zip(node.var_names, aligned):
                arr_2d = np.ascontiguousarray(
                    tensor.data[0 if tensor.data.shape[0] == 1 else b_idx],
                    dtype=np.float64,
                )
                parser.define_var(var_name, arr_2d)
                if node.nodata is not None:
                    if np.isnan(node.nodata):
                        cur_inv = ~np.isfinite(arr_2d)
                    else:
                        cur_inv = (~np.isfinite(arr_2d)) | (arr_2d == node.nodata)
                    invalid_mask = (
                        cur_inv if invalid_mask is None else (invalid_mask | cur_inv)
                    )
            if needs_parse and b_idx == 0:
                try:
                    parser.set_expr(node.expr)
                except pymu_eval.ParserError as err:
                    raise rsgispy.RSGISPyException(
                        f"Invalid BandMathOp expression '{node.expr}': {err}"
                    ) from err
                parsers[node] = parser
            try:
                with np.errstate(all="ignore"):
                    val = np.asarray(parser.eval(), dtype=np.float64)
            except pymu_eval.ParserError as err:
                raise rsgispy.RSGISPyException(
                    f"Error evaluating BandMathOp expression '{node.expr}': {err}"
                ) from err
            if val.ndim == 0:
                val = np.full(first.data.shape[1:], float(val), dtype=np.float64)
            if invalid_mask is not None and np.any(invalid_mask):
                val = np.where(invalid_mask, float(node.nodata), val)  # type: ignore[arg-type]
            out_bands.append(val)

        return _TileTensor(
            data=np.stack(out_bands, axis=0),
            halo_y=out_hy,
            halo_x=out_hx,
            oob_top=first.oob_top,
            oob_bottom=first.oob_bottom,
            oob_left=first.oob_left,
            oob_right=first.oob_right,
        )

    def run(self, use_pbar: bool = True) -> Dict[str, np.ndarray]:
        """
        Execute the tile graph across all tiles, writing every :class:`DiskSink`
        and populating every :class:`MemorySink`.

        :param use_pbar: If ``True`` (default), display a ``tqdm`` progress bar
                         over output tiles.
        :return: Dictionary mapping each :class:`MemorySink`'s ``name`` (or
                 ``'memory_<i>'`` when unnamed) to its assembled 3D
                 ``(C, H, W)`` NumPy array.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import (
            IntersectionGrid,
            cast_result_block,
            ensure_file_exists,
            iter_dataset_windows,
            open_output_dataset,
        )
        from rsgispy.imageutils._helpers import normalise_creation_options

        sources = [
            n for n in self.topo_order if isinstance(n, (RasterSource, ArraySource))
        ]
        if not sources:
            raise rsgispy.RSGISPyException(
                "TileGraph must contain at least one RasterSource or ArraySource."
            )

        with ExitStack() as stack:
            open_rasters: Dict[RasterSource, rasterio.io.DatasetReader] = {}
            img_h: Optional[int] = None
            img_w: Optional[int] = None
            ref_transform: Optional[Affine] = None
            ref_crs: Any = None
            ref_nodata: Optional[float] = None

            for src_node in sources:
                if isinstance(src_node, RasterSource):
                    ensure_file_exists(src_node.path)
                    ds = stack.enter_context(rasterio.open(src_node.path, "r"))
                    open_rasters[src_node] = ds
                    cur_h, cur_w = int(ds.height), int(ds.width)
                    cur_transform, cur_crs = ds.transform, ds.crs
                    cur_nd = (
                        src_node.nodata if src_node.nodata is not None else ds.nodata
                    )
                else:
                    cur_h = int(src_node.array.shape[1])
                    cur_w = int(src_node.array.shape[2])
                    cur_transform = src_node.transform
                    cur_crs = src_node.crs
                    cur_nd = src_node.nodata

                if img_h is None:
                    img_h, img_w = cur_h, cur_w
                    ref_transform = cur_transform
                    ref_crs = cur_crs
                    ref_nodata = cur_nd
                elif (cur_h, cur_w) != (img_h, img_w):
                    raise rsgispy.RSGISPyException(
                        f"TileGraph sources must have matching dimensions; "
                        f"got {(cur_h, cur_w)} vs {(img_h, img_w)}."
                    )

            if img_h is None or img_w is None:
                raise rsgispy.RSGISPyException(
                    "TileGraph could not determine image dimensions from sources."
                )
            grid = IntersectionGrid(
                crs=ref_crs,
                transform=(
                    ref_transform if ref_transform is not None else Affine.identity()
                ),
                width=img_w,
                height=img_h,
                offsets={},
            )

            open_sinks: Dict[DiskSink, rasterio.io.DatasetWriter] = {}
            mem_arrays: Dict[MemorySink, np.ndarray] = {}
            band_math_parsers: Dict[BandMathOp, pymu_eval.Parser] = {}

            for win in iter_dataset_windows(
                img_h, img_w, block_size=self.block_size, use_pbar=use_pbar
            ):
                base_ctx = TileContext(
                    row_off=int(win.row_off),
                    col_off=int(win.col_off),
                    height=int(win.height),
                    width=int(win.width),
                    img_height=img_h,
                    img_width=img_w,
                    halo_y=0,
                    halo_x=0,
                    oob_top=0,
                    oob_bottom=0,
                    oob_left=0,
                    oob_right=0,
                    transform=ref_transform,
                    crs=ref_crs,
                    nodata=ref_nodata,
                )

                tensors: Dict[TileNode, _TileTensor] = {}
                remaining_uses = dict(self.consumer_counts)

                for node in self.topo_order:
                    hy, hx = self.required_halos[node]
                    if isinstance(node, RasterSource):
                        tensors[node] = self._read_raster_source(
                            node, open_rasters[node], win, hy, hx
                        )
                    elif isinstance(node, ArraySource):
                        tensors[node] = self._read_array_source(node, win, hy, hx)
                    elif isinstance(node, WindowOp):
                        parent = node.inputs[0]
                        tensors[node] = self._eval_window_op(
                            node, tensors[parent], hy, hx, base_ctx
                        )
                    elif isinstance(node, PointwiseOp):
                        in_t = [tensors[p] for p in node.inputs]
                        tensors[node] = self._eval_pointwise_op(
                            node, in_t, hy, hx, base_ctx
                        )
                    elif isinstance(node, BandMathOp):
                        in_t = [tensors[p] for p in node.inputs]
                        tensors[node] = self._eval_band_math_op(
                            node, in_t, hy, hx, band_math_parsers
                        )
                    elif isinstance(node, DiskSink):
                        core = _crop_tensor(tensors[node.inputs[0]], 0, 0).data
                        if node not in open_sinks:
                            out_rio_dt = rsgispy.get_rasterio_datatype(node.datatype)
                            dst = open_output_dataset(
                                stack=stack,
                                output_img=node.output_img,
                                gdalformat=node.gdalformat,
                                out_rio_dtype=out_rio_dt,
                                grid=grid,
                                count=int(core.shape[0]),
                                output_exists=False,
                                creation_opts=normalise_creation_options(
                                    node.creation_opts
                                ),
                            )
                            if node.nodata is not None:
                                dst.nodata = node.nodata
                            open_sinks[node] = dst
                        out_np_dt = np.dtype(
                            rsgispy.get_numpy_datatype(node.datatype)
                        )
                        cast_blk = cast_result_block(
                            core,
                            int(win.height),
                            int(win.width),
                            out_np_dt,
                            clamp_int=node.clamp_int,
                        )
                        open_sinks[node].write(cast_blk, window=win)
                    elif isinstance(node, MemorySink):
                        core = _crop_tensor(tensors[node.inputs[0]], 0, 0).data
                        out_np_dt = (
                            node.dtype if node.dtype is not None else core.dtype
                        )
                        if node not in mem_arrays:
                            mem_arrays[node] = np.empty(
                                (int(core.shape[0]), img_h, img_w),
                                dtype=out_np_dt,
                            )
                        cast_blk = cast_result_block(
                            core,
                            int(win.height),
                            int(win.width),
                            out_np_dt,
                            clamp_int=node.clamp_int,
                        )
                        r0 = int(win.row_off)
                        c0 = int(win.col_off)
                        mem_arrays[node][
                            :, r0 : r0 + int(win.height), c0 : c0 + int(win.width)
                        ] = cast_blk

                    for parent in node.inputs:
                        remaining_uses[parent] -= 1
                        if remaining_uses[parent] <= 0:
                            tensors.pop(parent, None)

        # Post-close metadata / statistics for DiskSinks
        for sink in self.sinks:
            if isinstance(sink, DiskSink):
                if sink.band_names is not None:
                    # pylint: disable=import-outside-toplevel
                    from rsgispy.imageutils.metadata import set_band_names

                    set_band_names(
                        sink.output_img, sink.band_names, feedback=False
                    )
                if sink.calc_stats:
                    # pylint: disable=import-outside-toplevel
                    from rsgispy.imageutils.stats import pop_img_stats

                    pop_img_stats(
                        sink.output_img,
                        no_data_val=sink.nodata,
                        calc_pyramids=True,
                        use_pbar=use_pbar,
                    )

        mem_results: Dict[str, np.ndarray] = {}
        mem_idx = 0
        for sink in self.sinks:
            if isinstance(sink, MemorySink):
                arr_out = mem_arrays[sink]
                sink.result = arr_out
                key = sink.name if sink.name is not None else f"memory_{mem_idx}"
                mem_results[key] = arr_out
                mem_idx += 1
        return mem_results


def run_tile_graph(
    sinks: Union[DiskSink, MemorySink, Sequence[Union[DiskSink, MemorySink]]],
    block_size: int = 256,
    use_pbar: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Construct and immediately execute a :class:`TileGraph`.

    :param sinks: Single :class:`DiskSink` / :class:`MemorySink` or sequence of
                  sinks.
    :param block_size: Core tile size in pixels (Default = ``256``).
    :param use_pbar: Display a ``tqdm`` progress bar (Default = ``True``).
    :return: Dictionary of :class:`MemorySink` outputs (empty when only
             :class:`DiskSink`s are used).
    """
    return TileGraph(sinks=sinks, block_size=block_size).run(use_pbar=use_pbar)
