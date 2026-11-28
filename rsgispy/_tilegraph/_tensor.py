"""
Border configuration, tile context, tile tensors, node metadata and halo
tensor helpers for :mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

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
class TileTensor:
    """
    Block of one node's output for the current tile: a 3D
    ``(bands, height + 2*halo_y, width + 2*halo_x)`` array plus how many of
    the halo rows/columns on each side lie outside the image
    (``oob_top``, ``oob_bottom``, ``oob_left``, ``oob_right``).

    Custom :class:`~rsgispy.imageprocess.TileNode` subclasses receive their
    inputs as :class:`TileTensor` objects in
    :meth:`~rsgispy.imageprocess.TileNode.evaluate` and return one.
    """

    data: np.ndarray
    halo_y: int
    halo_x: int
    oob_top: int
    oob_bottom: int
    oob_left: int
    oob_right: int

    def crop(self, halo_y: int, halo_x: int) -> "TileTensor":
        """
        Return this tensor cropped symmetrically to a smaller halo.

        :param halo_y: Target halo in rows (``<= self.halo_y``).
        :param halo_x: Target halo in columns (``<= self.halo_x``).
        :return: Cropped :class:`TileTensor` (a view; no copy).
        """
        return _crop_tensor(self, halo_y, halo_x)

    def with_data(self, data: np.ndarray) -> "TileTensor":
        """
        Return a tensor with the same halo and out-of-image margins as this
        one but holding ``data`` (2D or 3D with the same spatial shape).

        :param data: New block array.
        :return: New :class:`TileTensor`.
        """
        arr = _ensure_3d(np.asarray(data))
        if arr.shape[1:] != self.data.shape[1:]:
            raise rsgispy.RSGISPyException(
                f"Tile data shape {arr.shape[1:]} does not match the tensor "
                f"shape {self.data.shape[1:]}."
            )
        return TileTensor(
            data=arr,
            halo_y=self.halo_y,
            halo_x=self.halo_x,
            oob_top=self.oob_top,
            oob_bottom=self.oob_bottom,
            oob_left=self.oob_left,
            oob_right=self.oob_right,
        )


# Backwards-compatible private alias.
_TileTensor = TileTensor


@dataclass(frozen=True)
class NodeMeta:
    """
    Metadata of the blocks a :class:`~rsgispy.imageprocess.TileNode`
    produces, derived from its inputs and parameters when a
    :class:`~rsgispy.imageprocess.TileGraph` is built. ``None`` means
    unknown (e.g. for a :class:`~rsgispy.imageprocess.PointwiseOp` running an
    arbitrary function).

    :param n_bands: Number of output bands.
    :param band_names: Output band names (one per band).
    :param dtype: NumPy dtype of the output blocks.
    :param nodata: No-data value of the output.
    """

    n_bands: Optional[int] = None
    band_names: Optional[Tuple[str, ...]] = None
    dtype: Optional[np.dtype] = None
    nodata: Optional[float] = None

    def __post_init__(self) -> None:
        if self.band_names is not None:
            object.__setattr__(
                self, "band_names", tuple(str(b) for b in self.band_names)
            )
            if self.n_bands is not None and len(self.band_names) != self.n_bands:
                raise rsgispy.RSGISPyException(
                    f"NodeMeta has {len(self.band_names)} band names for "
                    f"{self.n_bands} band(s)."
                )
        if self.dtype is not None:
            object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def replace(self, **changes: Any) -> "NodeMeta":
        """
        Return a copy with some fields replaced.

        :param changes: Field values to change (``n_bands``, ``band_names``,
                        ``dtype``, ``nodata``).
        :return: New :class:`NodeMeta`.
        """
        return dataclasses.replace(self, **changes)

    def describe(self) -> str:
        """Return a one-line text summary of the metadata."""
        bands = "?" if self.n_bands is None else str(self.n_bands)
        names = "" if self.band_names is None else f" names={list(self.band_names)}"
        dtype = "?" if self.dtype is None else str(self.dtype)
        nodata = "None" if self.nodata is None else repr(self.nodata)
        return f"bands={bands}{names} dtype={dtype} nodata={nodata}"


@dataclass(frozen=True)
class SourceGrid:
    """
    Pixel grid of a source node, used to check that all sources of a
    :class:`~rsgispy.imageprocess.TileGraph` are on the same grid.

    :param height: Number of rows.
    :param width: Number of columns.
    :param transform: Affine geotransform (``None`` when unknown).
    :param crs: Coordinate reference system (``None`` when unknown).
    :param nodata: No-data value of the source (``None`` when not set).
    :param label: Name used in error messages (e.g. the file path).
    """

    height: int
    width: int
    transform: Optional[Affine] = None
    crs: Any = None
    nodata: Optional[float] = None
    label: str = "source"


def same_bands_meta(
    dtype: Any = np.float64, nodata: Optional[float] = None
) -> Any:
    """
    Return a metadata function for a node whose output has the same bands
    (count and names) as its first input, with the given ``dtype`` and
    ``nodata``. Pass it as ``meta_fn`` to
    :class:`~rsgispy.imageprocess.WindowOp` or
    :class:`~rsgispy.imageprocess.PointwiseOp`.

    :param dtype: NumPy dtype of the output blocks (Default = ``float64``).
    :param nodata: Output no-data value (Default = ``None``).
    :return: Callable ``(input_metas) -> NodeMeta``.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import RasterSource, WindowOp, same_bands_meta

        src = RasterSource("in.tif")
        blur = WindowOp(
            src, 1, lambda a: a[:, 1:-1, 1:-1], meta_fn=same_bands_meta(np.float64)
        )
    """
    out_dtype = np.dtype(dtype) if dtype is not None else None

    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        first = inputs[0] if inputs else NodeMeta()
        return NodeMeta(
            n_bands=first.n_bands,
            band_names=first.band_names,
            dtype=out_dtype,
            nodata=nodata,
        )

    return _meta


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


def _fits_int(value: float, np_dtype: np.dtype) -> bool:
    """``True`` if ``value`` is an integer that ``np_dtype`` can store:
    :func:`rsgispy.imageprocess.fits_dtype_range`."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess import fits_dtype_range

    return fits_dtype_range(value, np_dtype)


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
        if np.issubdtype(arr.dtype, np.integer) and not _fits_int(
            border.fill_value, arr.dtype
        ):
            # e.g. a NaN fill around integer blocks: pad a float64 copy, as
            # the fill cannot be stored in the integer type.
            arr = arr.astype(np.float64)
        return np.pad(
            arr,
            pad_width=pad_width,
            mode="constant",
            constant_values=border.fill_value,
        )
    return np.pad(arr, pad_width=pad_width, mode=border.mode)  # type: ignore[arg-type]


def _crop_tensor(
    tensor: TileTensor,
    target_hy: int,
    target_hx: int,
) -> TileTensor:
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
    return TileTensor(
        data=cropped,
        halo_y=target_hy,
        halo_x=target_hx,
        oob_top=max(0, tensor.oob_top - dy),
        oob_bottom=max(0, tensor.oob_bottom - dy),
        oob_left=max(0, tensor.oob_left - dx),
        oob_right=max(0, tensor.oob_right - dx),
    )


def _repad_oob(tensor: TileTensor, border: HaloBorder) -> np.ndarray:
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
