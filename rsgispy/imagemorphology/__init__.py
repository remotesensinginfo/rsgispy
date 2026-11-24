# pylint: disable=too-many-lines
"""
The imagemorphology module contains functions for performing morphological
operations on raster images using ``rasterio`` and ``numpy``.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio

import rsgispy
from rsgispy._io import (
    DiskSink,
    HaloBorder,
    PointwiseOp,
    RasterSource,
    TileGraph,
    TileNode,
    WindowOp,
)
from rsgispy.imageprocess import (
    ensure_file_exists,
    run_window_halo_blocks,
    validate_odd_window_size,
)
from rsgispy.imageutils.metadata import get_img_no_data_value
from rsgispy.imageutils.stats import pop_img_stats

__all__ = [
    "create_circular_op",
    "image_dilate",
    "image_erode",
    "image_gradient",
    "image_dilate_combined_out",
    "image_erode_combined_out",
    "image_gradient_combined_out",
    "image_local_minima",
    "image_local_minima_combined_out",
    "image_opening",
    "image_closing",
    "image_black_top_hat",
    "image_white_top_hat",
]


def _resolve_morph_datatype(
    input_img: str,
    datatype: Optional[int],
    default_dtype: Optional[int] = None,
) -> int:
    """Resolve output RSGIS datatype (inferring from ``input_img`` when ``None``)."""
    if datatype is not None:
        return int(datatype)
    if default_dtype is not None:
        return int(default_dtype)
    ensure_file_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        return rsgispy.get_rsgis_datatype(src.dtypes[0])


def _resolve_input_nodata(
    input_img: str, no_data_val: Optional[float]
) -> Optional[float]:
    """Resolve effective input nodata value for masking invalid pixels."""
    if no_data_val is not None:
        return float(no_data_val)
    hdr_nd = get_img_no_data_value(input_img)
    return float(hdr_nd) if hdr_nd is not None else None


def _resolve_morph_nodata(
    input_img: str,
    no_data_val: Optional[float],
    out_np_dtype: np.dtype,
) -> Optional[float]:
    """Resolve output nodata value, preserving input header nodata when representable."""
    eff_nd = _resolve_input_nodata(input_img, no_data_val)
    if eff_nd is None or not math.isfinite(float(eff_nd)):
        return None
    nd_val = float(eff_nd)
    if np.issubdtype(out_np_dtype, np.floating):
        info = np.finfo(out_np_dtype)
        if float(info.min) <= nd_val <= float(info.max):  # pylint: disable=no-member
            return nd_val
        return None
    if np.issubdtype(out_np_dtype, np.integer):
        if nd_val.is_integer():
            info_i = np.iinfo(out_np_dtype)
            if int(info_i.min) <= int(nd_val) <= int(info_i.max):
                return nd_val
        return None
    return None


def _resolve_fill_nodata(
    out_nd: Optional[float], out_np_dtype: np.dtype
) -> float:
    """Resolve numeric fill value written at invalid/nodata output pixels."""
    if out_nd is not None:
        return float(out_nd)
    if np.issubdtype(out_np_dtype, np.floating):
        return float(np.nan)
    return 0.0


def _valid_morph_mask(
    arr: np.ndarray, in_nd: Optional[float]
) -> np.ndarray:
    """Return boolean mask of finite, non-nodata pixels in ``arr``."""
    valid = np.isfinite(arr)
    if in_nd is not None and math.isfinite(float(in_nd)):
        valid &= arr != float(in_nd)
    return valid


def _parse_morph_op_file(morph_op_file: str) -> np.ndarray:
    """
    Parse an RSGISLib ``.gmtxt`` structuring-element text file into a boolean
    mask array.
    """
    ensure_file_exists(morph_op_file, label="Morphology operator")

    with open(morph_op_file, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 3 or not lines[0].startswith("m=") or not lines[1].startswith("n="):
        raise rsgispy.RSGISPyException(
            f"Invalid morphology operator file header in '{morph_op_file}'."
        )

    try:
        cols = int(lines[0][2:].strip())
        rows = int(lines[1][2:].strip())
    except ValueError as err:
        raise rsgispy.RSGISPyException(
            f"Could not parse dimensions from '{morph_op_file}': {err}"
        ) from err

    if rows != cols:
        raise rsgispy.RSGISPyException(
            "Morphological operator must be a square matrix."
        )
    validate_odd_window_size(
        rows, min_size=3, name="Morphological operator size"
    )

    tokens: List[float] = []
    for row_line in lines[2:]:
        for tok in row_line.split(","):
            tok_s = tok.strip()
            if tok_s:
                tokens.append(float(tok_s))

    expected = rows * cols
    if len(tokens) != expected:
        raise rsgispy.RSGISPyException(
            f"Expected {expected} elements in '{morph_op_file}', found {len(tokens)}."
        )

    mat = np.asarray(tokens, dtype=np.float64).reshape((rows, cols))
    op_mask = mat > 0.0
    if not np.any(op_mask):
        raise rsgispy.RSGISPyException(
            "Morphological operator must contain at least one positive entry."
        )
    return op_mask


def _load_morph_operator(
    morph_op_file: str, use_op_file: bool, op_size: int
) -> np.ndarray:
    """
    Load a boolean morphological structuring element either from ``morph_op_file``
    or as a square all-ones matrix of size ``op_size``.
    """
    if use_op_file:
        return _parse_morph_op_file(morph_op_file)

    size_int = validate_odd_window_size(op_size, min_size=3, name="op_size")
    return np.ones((size_int, size_int), dtype=bool)


def _fill_minima_band(
    out_band: np.ndarray,
    is_minima: np.ndarray,
    sequencial_out: bool,
    seq_counters: np.ndarray,
    counter_idx: int,
) -> None:
    """Populate a single 2D minima output band with sequential IDs or binary 1s."""
    if sequencial_out:
        flat_mask = is_minima.ravel()
        n_hits = int(np.sum(flat_mask))
        if n_hits > 0:
            start = int(seq_counters[counter_idx])
            out_band.ravel()[flat_mask] = np.arange(
                start, start + n_hits, dtype=np.float64
            )
            seq_counters[counter_idx] += n_hits
    else:
        out_band[is_minima] = 1.0


def _compute_local_minima_block(
    padded: np.ndarray,
    op_mask: np.ndarray,
    combined: bool,
    allow_equal: bool,
    sequencial_out: bool,
    seq_counters: np.ndarray,
    in_nd: Optional[float] = None,
) -> np.ndarray:
    """Evaluate local minima on an edge-padded block of shape ``(bands, H+2h, W+2h)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_size = op_mask.shape[0]
    half = win_size // 2
    n_bands = padded.shape[0]
    out_h = padded.shape[1] - (2 * half)
    out_w = padded.shape[2] - (2 * half)

    ii, jj = np.ogrid[:win_size, :win_size]
    nbr_mask = op_mask & ((ii != half) | (jj != half))
    has_nbrs = bool(np.any(nbr_mask))

    valid_pad = _valid_morph_mask(padded, in_nd)
    centres = padded[:, half:half + out_h, half:half + out_w]
    centre_valid = valid_pad[:, half:half + out_h, half:half + out_w]
    windows = sliding_window_view(padded, (win_size, win_size), axis=(1, 2))
    valid_wins = sliding_window_view(valid_pad, (win_size, win_size), axis=(1, 2))
    nbr_vals = windows[:, :, :, nbr_mask] if has_nbrs else None
    nbr_valid = valid_wins[:, :, :, nbr_mask] if has_nbrs else None

    if combined:
        any_centre_valid = np.any(centre_valid, axis=0)
        centre_min = np.min(np.where(centre_valid, centres, np.inf), axis=0)
        if nbr_vals is not None and nbr_valid is not None:
            cmp_mask = (
                nbr_valid & (nbr_vals < centre_min[None, :, :, None])
                if allow_equal
                else nbr_valid & (nbr_vals <= centre_min[None, :, :, None])
            )
            is_minima = any_centre_valid & ~np.any(cmp_mask, axis=(0, 3))
        else:
            is_minima = any_centre_valid

        out = np.zeros((1, out_h, out_w), dtype=np.float64)
        _fill_minima_band(out[0], is_minima, sequencial_out, seq_counters, 0)
        return out

    out = np.zeros((n_bands, out_h, out_w), dtype=np.float64)
    for b_idx in range(n_bands):
        if nbr_vals is not None and nbr_valid is not None:
            band_nbrs = nbr_vals[b_idx]
            band_nbr_valid = nbr_valid[b_idx]
            cmp_mask = (
                band_nbr_valid & (band_nbrs < centres[b_idx, :, :, None])
                if allow_equal
                else band_nbr_valid & (band_nbrs <= centres[b_idx, :, :, None])
            )
            is_minima = centre_valid[b_idx] & ~np.any(cmp_mask, axis=-1)
        else:
            is_minima = centre_valid[b_idx]

        _fill_minima_band(out[b_idx], is_minima, sequencial_out, seq_counters, b_idx)
    return out


def _morph_reduce_block(
    padded: np.ndarray,
    op_mask: np.ndarray,
    op_kind: str,
    combined: bool,
    in_nd: Optional[float],
    fill_nd: float,
) -> np.ndarray:
    """Compute nodata-aware dilate/erode/gradient over an edge-padded block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_size = int(op_mask.shape[0])
    half = win_size // 2
    out_h = padded.shape[1] - (2 * half)
    out_w = padded.shape[2] - (2 * half)
    reduce_axes: Tuple[int, ...] = (0, -1) if combined else (-1,)

    valid_pad = _valid_morph_mask(padded, in_nd)
    centre_valid = valid_pad[:, half:half + out_h, half:half + out_w]
    if combined:
        centre_keep = np.any(centre_valid, axis=0)
    else:
        centre_keep = centre_valid

    win_view = sliding_window_view(padded, (win_size, win_size), axis=(1, 2))
    valid_view = sliding_window_view(valid_pad, (win_size, win_size), axis=(1, 2))
    active = win_view[:, :, :, op_mask]
    active_valid = valid_view[:, :, :, op_mask]
    keep = centre_keep & np.any(active_valid, axis=reduce_axes)

    if op_kind == "dilate":
        raw_out = np.max(np.where(active_valid, active, -np.inf), axis=reduce_axes)
    elif op_kind == "erode":
        raw_out = np.min(np.where(active_valid, active, np.inf), axis=reduce_axes)
    else:
        max_v = np.max(np.where(active_valid, active, -np.inf), axis=reduce_axes)
        min_v = np.min(np.where(active_valid, active, np.inf), axis=reduce_axes)
        raw_out = max_v - min_v

    raw_out = np.where(keep, raw_out, fill_nd)
    if combined:
        raw_out = raw_out[np.newaxis, :, :]
    return raw_out


def _run_window_morphology(
    input_img: str,
    output_img: str,
    op_mask: np.ndarray,
    gdalformat: str,
    datatype: Optional[int],
    op_kind: str,
    combined: bool = False,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    default_dtype: Optional[int] = None,
) -> None:
    """Apply a windowed morphological operation across ``input_img`` block-by-block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if op_kind not in ("dilate", "erode", "gradient", "local_minima"):
        raise rsgispy.RSGISPyException(
            f"Unsupported morphology operation: '{op_kind}'."
        )

    eff_dtype = _resolve_morph_datatype(
        input_img, datatype, default_dtype=default_dtype
    )
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    in_nd = _resolve_input_nodata(input_img, no_data_val)
    out_nd = _resolve_morph_nodata(input_img, no_data_val, out_np_dtype)
    fill_nd = _resolve_fill_nodata(out_nd, out_np_dtype)
    win_size = int(op_mask.shape[0])
    use_strips = op_kind == "local_minima" and sequencial_out
    with rasterio.open(input_img, "r") as src:
        out_count = 1 if combined else int(src.count)
    seq_counters = np.ones(out_count, dtype=np.int64)

    def _morph_block(padded: np.ndarray) -> np.ndarray:
        if op_kind == "local_minima":
            return _compute_local_minima_block(
                padded=padded,
                op_mask=op_mask,
                combined=combined,
                allow_equal=allow_equal,
                sequencial_out=sequencial_out,
                seq_counters=seq_counters,
                in_nd=in_nd,
            )
        return _morph_reduce_block(
            padded=padded,
            op_mask=op_mask,
            op_kind=op_kind,
            combined=combined,
            in_nd=in_nd,
            fill_nd=fill_nd,
        )

    run_window_halo_blocks(
        input_img=input_img,
        output_img=output_img,
        win_shape=(win_size, win_size),
        block_fn=_morph_block,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        out_np_dtype=out_np_dtype,
        out_count=out_count,
        per_band=False,
        read_dtype=np.float64,
        pad_mode="edge",
        out_no_data_val=out_nd,
        clamp_int=True,
        full_width_strips=use_strips,
        use_pbar=use_pbar,
    )
    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _make_morph_step_node(
    source: TileNode,
    op_mask: np.ndarray,
    op_kind: str,
    out_np_dtype: np.dtype,
    in_nd: Optional[float] = None,
    fill_nd: float = 0.0,
    name: Optional[str] = None,
) -> WindowOp:
    """Build a :class:`WindowOp` node applying a single dilation/erosion step."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    win_size = int(op_mask.shape[0])
    half = win_size // 2
    is_int = np.issubdtype(out_np_dtype, np.integer)
    int_info = np.iinfo(out_np_dtype) if is_int else None

    def _step_fn(arr: np.ndarray) -> np.ndarray:
        raw_out = _morph_reduce_block(
            padded=arr,
            op_mask=op_mask,
            op_kind=op_kind,
            combined=False,
            in_nd=in_nd,
            fill_nd=fill_nd,
        )
        if int_info is not None:
            raw_out = np.clip(
                raw_out, float(int_info.min), float(int_info.max)
            ).astype(out_np_dtype).astype(np.float64)
        return raw_out

    return WindowOp(
        source=source,
        radius=(half, half),
        func=_step_fn,
        border=HaloBorder(mode="edge"),
        name=name,
    )


def _run_two_step_seq(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    op_mask: np.ndarray,
    gdalformat: str,
    datatype: Optional[int],
    first_op: str,
    second_op: str,
    niters: int = 1,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Execute ``niters`` iterations of ``first_op`` followed by ``niters``
    iterations of ``second_op`` (e.g., ``n`` erosions -> ``n`` dilations for
    opening, or ``n`` dilations -> ``n`` erosions for closing) in a single
    halo-fused :class:`~rsgispy._io.TileGraph` pass with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly requested.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if niters < 1:
        raise rsgispy.RSGISPyException(
            f"niters must be >= 1 (got {niters})."
        )

    eff_dtype = _resolve_morph_datatype(input_img, datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    in_nd = _resolve_input_nodata(input_img, no_data_val)
    out_nd = _resolve_morph_nodata(input_img, no_data_val, out_np_dtype)
    fill_nd = _resolve_fill_nodata(out_nd, out_np_dtype)
    cur_node: TileNode = RasterSource(input_img, dtype=np.float64, name="src")

    for idx in range(niters):
        step_nd = in_nd if idx == 0 else out_nd
        cur_node = _make_morph_step_node(
            source=cur_node,
            op_mask=op_mask,
            op_kind=first_op,
            out_np_dtype=out_np_dtype,
            in_nd=step_nd,
            fill_nd=fill_nd,
            name=f"{first_op}_{idx}",
        )
    mid_node = cur_node

    for idx in range(niters):
        cur_node = _make_morph_step_node(
            source=cur_node,
            op_mask=op_mask,
            op_kind=second_op,
            out_np_dtype=out_np_dtype,
            in_nd=out_nd,
            fill_nd=fill_nd,
            name=f"{second_op}_{idx}",
        )

    sinks: List[DiskSink] = [
        DiskSink(
            source=cur_node,
            output_img=output_img,
            gdalformat=gdalformat,
            datatype=eff_dtype,
            nodata=out_nd,
            clamp_int=True,
            calc_stats=calc_stats,
            name="output",
        )
    ]
    if tmp_img:
        sinks.append(
            DiskSink(
                source=mid_node,
                output_img=str(tmp_img),
                gdalformat=gdalformat,
                datatype=eff_dtype,
                nodata=out_nd,
                clamp_int=True,
                calc_stats=False,
                name="tmp_mid",
            )
        )

    TileGraph(sinks).run(use_pbar=use_pbar)


def create_circular_op(output_file: str, op_size: int) -> None:
    """
    Create a circular binary structuring-element matrix file (``.gmtxt``).

    If ``output_file`` does not end with ``.gmtxt``, ``.gmtxt`` is appended to
    match RSGISLib's behaviour.

    :param output_file: Output file path for the ``.gmtxt`` operator matrix.
    :param op_size: Odd integer width/height of the square operator matrix.

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.create_circular_op(
            output_file="circ_op_5.gmtxt",
            op_size=5,
        )
    """
    size_int = validate_odd_window_size(op_size, min_size=1, name="op_size")

    out_path = output_file if output_file.endswith(".gmtxt") else f"{output_file}.gmtxt"
    radius = (size_int - 1) // 2
    radius_sq = (radius + 0.25) ** 2

    y_coords = np.arange(radius, -radius - 1, -1, dtype=np.int64)[:, np.newaxis]
    x_coords = np.arange(-radius, radius + 1, dtype=np.int64)[np.newaxis, :]
    mat = ((x_coords**2 + y_coords**2) <= radius_sq).astype(np.int64)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"m={size_int}\n")
        handle.write(f"n={size_int}\n")
        for row in mat:
            handle.write(",".join(str(int(v)) for v in row) + "\n")


def image_dilate(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological dilation independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate(
            input_img="mask.tif",
            output_img="mask_dilate.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_erode(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological erosion independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode(
            input_img="mask.tif",
            output_img="mask_erode.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_gradient(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate morphological gradient (dilation minus erosion) independently for
    each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output morphological gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradient(
            input_img="img.tif",
            output_img="img_grad.tif",
            op_size=3,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradient",
        combined=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_dilate_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological dilation across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate_combined_out(
            input_img="masks.tif",
            output_img="masks_dilate_comb.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_erode_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological erosion across all bands of ``input_img`` and write a
    single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode_combined_out(
            input_img="masks.tif",
            output_img="masks_erode_comb.tif",
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_gradient_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate morphological gradient across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradient_combined_out(
            input_img="img.tif",
            output_img="img_grad_comb.tif",
            op_size=3,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradient",
        combined=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_local_minima(
    input_img: str,
    output_img: str,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Identify local minima independently in each band of ``input_img`` using the
    morphological structuring element.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially per band
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``; Default = ``False``).
    :param allow_equal: If ``True``, neighbour pixels equal to the centre pixel
                        do not disqualify a local minimum (Default = ``True``).
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, resolves to
                     :data:`rsgispy.TYPE_32UINT` when ``sequencial_out=True``
                     else :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima(
            input_img="dem.tif",
            output_img="minima.tif",
            sequencial_out=False,
            allow_equal=True,
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    default_dtype = rsgispy.TYPE_32UINT if sequencial_out else rsgispy.TYPE_8UINT
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="local_minima",
        combined=False,
        sequencial_out=sequencial_out,
        allow_equal=allow_equal,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        default_dtype=default_dtype,
    )


def image_local_minima_combined_out(
    input_img: str,
    output_img: str,
    sequencial_out: bool = False,
    allow_equal: bool = True,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Identify local minima across all bands of ``input_img`` and write a single
    combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``; Default = ``False``).
    :param allow_equal: If ``True``, neighbour pixels equal to the minimum band
                        centre value do not disqualify a local minimum
                        (Default = ``True``).
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, resolves to
                     :data:`rsgispy.TYPE_32UINT` when ``sequencial_out=True``
                     else :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima_combined_out(
            input_img="bands.tif",
            output_img="minima_comb.tif",
            sequencial_out=False,
            allow_equal=True,
            op_size=3,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    default_dtype = rsgispy.TYPE_32UINT if sequencial_out else rsgispy.TYPE_8UINT
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="local_minima",
        combined=True,
        sequencial_out=sequencial_out,
        allow_equal=allow_equal,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        default_dtype=default_dtype,
    )


def image_opening(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    niters: int = 1,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological opening (erosion followed by dilation) on ``input_img``.

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output opened image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate eroded raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param niters: Number of opening iterations (Default = ``1``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_opening(
            input_img="mask.tif",
            output_img="mask_open.tif",
            op_size=3,
            niters=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_two_step_seq(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        first_op="erode",
        second_op="dilate",
        niters=niters,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_closing(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    niters: int = 1,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform morphological closing (dilation followed by erosion) on ``input_img``.

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output closed image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate dilated raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param niters: Number of closing iterations (Default = ``1``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_closing(
            input_img="mask.tif",
            output_img="mask_close.tif",
            op_size=3,
            niters=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_two_step_seq(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        first_op="dilate",
        second_op="erode",
        niters=niters,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def _run_top_hat(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    op_mask: np.ndarray,
    gdalformat: str,
    datatype: Optional[int],
    black: bool,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Execute black (``closing - input``) or white (``input - opening``) top-hat
    transform in a single halo-fused :class:`~rsgispy._io.TileGraph` pass.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    eff_dtype = _resolve_morph_datatype(input_img, datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    in_nd = _resolve_input_nodata(input_img, no_data_val)
    out_nd = _resolve_morph_nodata(input_img, no_data_val, out_np_dtype)
    fill_nd = _resolve_fill_nodata(out_nd, out_np_dtype)
    src_node = RasterSource(input_img, dtype=np.float64, name="src")
    first_op, second_op = ("dilate", "erode") if black else ("erode", "dilate")
    step1 = _make_morph_step_node(
        source=src_node,
        op_mask=op_mask,
        op_kind=first_op,
        out_np_dtype=out_np_dtype,
        in_nd=in_nd,
        fill_nd=fill_nd,
        name=first_op,
    )
    step2 = _make_morph_step_node(
        source=step1,
        op_mask=op_mask,
        op_kind=second_op,
        out_np_dtype=out_np_dtype,
        in_nd=out_nd,
        fill_nd=fill_nd,
        name=second_op,
    )

    def _top_hat_diff(first_arr: np.ndarray, second_arr: np.ndarray) -> np.ndarray:
        orig_arr = second_arr if black else first_arr
        orig_valid = _valid_morph_mask(orig_arr, in_nd)
        other_arr = first_arr if black else second_arr
        other_valid = _valid_morph_mask(other_arr, out_nd)
        return np.where(orig_valid & other_valid, first_arr - second_arr, fill_nd)

    if black:
        diff_node = PointwiseOp(
            inputs=(step2, src_node),
            func=_top_hat_diff,
            name="black_top_hat_diff",
        )
    else:
        diff_node = PointwiseOp(
            inputs=(src_node, step2),
            func=_top_hat_diff,
            name="white_top_hat_diff",
        )

    sinks: List[DiskSink] = [
        DiskSink(
            source=diff_node,
            output_img=output_img,
            gdalformat=gdalformat,
            datatype=eff_dtype,
            nodata=out_nd,
            clamp_int=True,
            calc_stats=calc_stats,
            name="output",
        )
    ]
    if tmp_img:
        sinks.append(
            DiskSink(
                source=step2,
                output_img=str(tmp_img),
                gdalformat=gdalformat,
                datatype=eff_dtype,
                nodata=out_nd,
                clamp_int=True,
                calc_stats=False,
                name="tmp_mid",
            )
        )

    TileGraph(sinks).run(use_pbar=use_pbar)


def image_black_top_hat(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform a morphological black top-hat transform (closing minus input image).

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output black top-hat image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate closed raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_black_top_hat(
            input_img="dem.tif",
            output_img="dem_bth.tif",
            op_size=5,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_top_hat(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        black=True,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def image_white_top_hat(
    input_img: str,
    output_img: str,
    morph_op_file: str = "",
    use_op_file: bool = False,
    op_size: int = 3,
    tmp_img: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform a morphological white top-hat transform (input image minus opening).

    Executed in a single halo-fused tile pass in memory with zero intermediate
    disk I/O unless ``tmp_img`` is explicitly provided.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output white top-hat image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file
                          (Default = ``''``).
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``
                        (Default = ``False``).
    :param op_size: Square operator size used when ``use_op_file`` is ``False``
                    (Default = ``3``).
    :param tmp_img: Optional path to also save the intermediate opened raster;
                    if empty (``''``) or ``None``, no intermediate file is
                    written (Default = ``None``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: RSGIS data type constant (Default = ``None``, matches
                     ``input_img`` data type).
    :param no_data_val: Optional output no-data value (Default = ``None``,
                        inherits ``input_img`` header no-data when representable).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_white_top_hat(
            input_img="dem.tif",
            output_img="dem_wth.tif",
            op_size=5,
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_top_hat(
        input_img=input_img,
        output_img=output_img,
        tmp_img=tmp_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        black=False,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
