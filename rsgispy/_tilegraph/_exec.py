"""
Per-tile stage readers and operator evaluators for :mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pymu_eval
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._nodes import (
    ArraySource,
    BandMathOp,
    PointwiseOp,
    RasterSource,
    TileContext,
    WindowOp,
    _TileTensor,
    _crop_tensor,
    _ensure_3d,
    _pad_3d,
    _repad_oob,
)


def read_raster_source(
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


def read_array_source(
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


def eval_window_op(
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


def eval_pointwise_op(
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


def eval_band_math_op(
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
