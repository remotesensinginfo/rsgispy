"""
Tile-graph node definitions, border configuration, tile context, and halo
tensor helpers for :mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from rasterio.transform import Affine

import rsgispy

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
