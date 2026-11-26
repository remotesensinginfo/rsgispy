"""
Per-tile source readers (including the shared per-file reader) and operator
evaluators for :mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymu_eval
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._tensor import (
    HaloBorder,
    TileContext,
    TileTensor,
    _crop_tensor,
    _ensure_3d,
    _pad_3d,
    _repad_oob,
)

if TYPE_CHECKING:  # pragma: no cover
    from rsgispy._tilegraph._nodes import (
        ArraySource,
        BandMathOp,
        PointwiseOp,
        RasterSource,
        WindowOp,
    )


def halo_read_bounds(
    win: Window, img_h: int, img_w: int, hy: int, hx: int
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Return ``(read_r0, read_c0, read_r1, read_c1, oob_top, oob_bottom,
    oob_left, oob_right)`` for reading ``win`` with a ``(hy, hx)`` halo
    clipped to an ``img_h`` x ``img_w`` image.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    r0 = int(win.row_off)
    c0 = int(win.col_off)
    r1 = r0 + int(win.height)
    c1 = c0 + int(win.width)
    return (
        max(0, r0 - hy),
        max(0, c0 - hx),
        min(img_h, r1 + hy),
        min(img_w, c1 + hx),
        max(0, hy - r0),
        max(0, (r1 + hy) - img_h),
        max(0, hx - c0),
        max(0, (c1 + hx) - img_w),
    )


def _finish_read(
    raw: np.ndarray,
    dtype: np.dtype,
    border: HaloBorder,
    hy: int,
    hx: int,
    oob: Tuple[int, int, int, int],
) -> TileTensor:
    """Cast a clipped block read, pad the out-of-image margins and wrap it."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    raw = raw.astype(dtype, copy=False)
    padded = _pad_3d(raw, oob[0], oob[1], oob[2], oob[3], border)
    return TileTensor(
        data=padded,
        halo_y=hy,
        halo_x=hx,
        oob_top=oob[0],
        oob_bottom=oob[1],
        oob_left=oob[2],
        oob_right=oob[3],
    )


class SharedRasterReader:
    """
    Reader shared by all :class:`~rsgispy.imageprocess.RasterSource` nodes
    of one :class:`~rsgispy.imageprocess.TileGraph` run on the same file.

    The file is opened once. For each tile the union of the bands that the
    nodes need is read once, with the largest halo any of them needs, and each
    node takes its bands and its halo from that block. The result is the
    same, pixel for pixel, as each node reading the file itself.

    :param dataset: Open rasterio dataset.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, dataset: rasterio.io.DatasetReader) -> None:
        self.dataset = dataset
        self._nodes: Dict[Any, Tuple[List[int], int, int]] = {}
        self._union: List[int] = []
        self._index: Dict[int, int] = {}
        self._halo: Tuple[int, int] = (0, 0)
        self._shared = True
        self._cache_key: Optional[Tuple[int, int, int, int]] = None
        self._cache: Optional[np.ndarray] = None
        self._cache_uses = 0

    def register(self, node: Any, bands: Sequence[int], hy: int, hx: int) -> None:
        """
        Register a node reading ``bands`` (1-based) with a ``(hy, hx)`` halo.
        """
        self._nodes[node] = (list(bands), int(hy), int(hx))
        union = sorted({b for b_list, _, _ in self._nodes.values() for b in b_list})
        self._union = union
        self._index = {b: i for i, b in enumerate(union)}
        self._halo = (
            max(h for _, h, _ in self._nodes.values()),
            max(w for _, _, w in self._nodes.values()),
        )
        dtypes = {self.dataset.dtypes[b - 1] for b in union}
        self._shared = len(self._nodes) > 1 and len(dtypes) == 1

    def read(self, node: Any, win: Window, dtype: Any, border: HaloBorder) -> TileTensor:
        """
        Return the halo-padded block of ``node`` for the tile ``win``.

        :param node: Registered node.
        :param win: Core tile window.
        :param dtype: NumPy dtype to cast the block to.
        :param border: Padding outside the image.
        :return: :class:`TileTensor` for ``node``.
        """
        # pylint: disable=too-many-locals
        bands, hy, hx = self._nodes[node]
        img_h = int(self.dataset.height)
        img_w = int(self.dataset.width)
        bounds = halo_read_bounds(win, img_h, img_w, hy, hx)
        if not self._shared:
            read_win = Window(
                bounds[1], bounds[0], bounds[3] - bounds[1], bounds[2] - bounds[0]
            )
            raw = self.dataset.read(bands, window=read_win)
            return _finish_read(raw, dtype, border, hy, hx, bounds[4:])

        key = (int(win.row_off), int(win.col_off), int(win.height), int(win.width))
        mhy, mhx = self._halo
        big = halo_read_bounds(win, img_h, img_w, mhy, mhx)
        if self._cache_key != key:
            read_win = Window(big[1], big[0], big[3] - big[1], big[2] - big[0])
            self._cache = self.dataset.read(self._union, window=read_win)
            self._cache_key = key
            self._cache_uses = 0
        cache = self._cache
        if cache is None:  # pragma: no cover - defensive
            raise rsgispy.RSGISPyException("Shared raster read failed.")
        y0 = bounds[0] - big[0]
        x0 = bounds[1] - big[1]
        y1 = y0 + (bounds[2] - bounds[0])
        x1 = x0 + (bounds[3] - bounds[1])
        idx = [self._index[b] for b in bands]
        raw = cache[idx, y0:y1, x0:x1]
        self._cache_uses += 1
        if self._cache_uses >= len(self._nodes):
            # Every node has read this tile: free the block.
            self._cache = None
            self._cache_key = None
        return _finish_read(raw, dtype, border, hy, hx, bounds[4:])


def read_raster_source(
    node: "RasterSource",
    ds: rasterio.io.DatasetReader,
    win: Window,
    hy: int,
    hx: int,
) -> TileTensor:
    """Read a halo-padded block from ``ds`` tracking exact OOB overhangs."""
    bounds = halo_read_bounds(win, int(ds.height), int(ds.width), hy, hx)
    read_win = Window(
        bounds[1], bounds[0], bounds[3] - bounds[1], bounds[2] - bounds[0]
    )
    if node.bands is None:
        raw = ds.read(window=read_win)
    else:
        raw = ds.read(node.bands, window=read_win)
    return _finish_read(raw, node.dtype, node.border, hy, hx, bounds[4:])


def read_array_source(
    node: "ArraySource",
    win: Window,
    hy: int,
    hx: int,
) -> TileTensor:
    """Slice a halo-padded block from an in-memory ``ArraySource``."""
    img_h = int(node.array.shape[1])
    img_w = int(node.array.shape[2])
    bounds = halo_read_bounds(win, img_h, img_w, hy, hx)
    raw = node.array[:, bounds[0]:bounds[2], bounds[1]:bounds[3]]
    return _finish_read(raw, node.dtype, node.border, hy, hx, bounds[4:])


def eval_window_op(
    node: "WindowOp",
    in_tensor: TileTensor,
    out_hy: int,
    out_hx: int,
    base_ctx: TileContext,
) -> TileTensor:
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

    return TileTensor(
        data=res,
        halo_y=out_hy,
        halo_x=out_hx,
        oob_top=max(0, aligned.oob_top - node.radius_y),
        oob_bottom=max(0, aligned.oob_bottom - node.radius_y),
        oob_left=max(0, aligned.oob_left - node.radius_x),
        oob_right=max(0, aligned.oob_right - node.radius_x),
    )


def eval_pointwise_op(
    node: "PointwiseOp",
    in_tensors: Sequence[TileTensor],
    out_hy: int,
    out_hx: int,
    base_ctx: TileContext,
) -> TileTensor:
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
    return TileTensor(
        data=res,
        halo_y=out_hy,
        halo_x=out_hx,
        oob_top=first.oob_top,
        oob_bottom=first.oob_bottom,
        oob_left=first.oob_left,
        oob_right=first.oob_right,
    )


def eval_band_math_op(
    node: "BandMathOp",
    in_tensors: Sequence[TileTensor],
    out_hy: int,
    out_hx: int,
    parsers: Dict[Any, pymu_eval.Parser],
) -> TileTensor:
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

    return TileTensor(
        data=np.stack(out_bands, axis=0),
        halo_y=out_hy,
        halo_x=out_hx,
        oob_top=first.oob_top,
        oob_bottom=first.oob_bottom,
        oob_left=first.oob_left,
        oob_right=first.oob_right,
    )
