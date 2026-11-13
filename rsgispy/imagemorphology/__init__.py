"""
The imagemorphology module contains functions for performing morphological
operations on raster images using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import os
import tempfile
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    ensure_file_exists,
    get_default_block_size,
    open_output_dataset,
)

__all__ = [
    "create_circular_op",
    "image_dilate",
    "image_erode",
    "image_gradiant",
    "image_dilate_combined_out",
    "image_erode_combined_out",
    "image_gradiant_combined_out",
    "image_local_minima",
    "image_local_minima_combined_out",
    "image_opening",
    "image_closing",
    "image_black_top_hat",
    "image_white_top_hat",
]


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
    if rows < 3 or rows % 2 == 0:
        raise rsgispy.RSGISPyException(
            f"Morphological operator size must be an odd integer >= 3 (got {rows})."
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

    size_int = int(op_size)
    if size_int < 3 or size_int % 2 == 0:
        raise rsgispy.RSGISPyException(
            f"op_size must be an odd integer >= 3 (got {op_size})."
        )
    return np.ones((size_int, size_int), dtype=bool)


def _cast_to_output_dtype(arr: np.ndarray, np_dtype: Any) -> np.ndarray:
    """
    Cast ``arr`` to ``np_dtype``, clamping integer types to their valid range
    to match GDAL raster casting semantics.
    """
    target_dtype = np.dtype(np_dtype)
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        return np.clip(arr, info.min, info.max).astype(target_dtype)
    return arr.astype(target_dtype)


def _iter_block_windows(
    width: int, height: int, block_size: int, full_width_strips: bool
) -> List[Window]:
    """Build block windows covering ``width x height``."""
    windows: List[Window] = []
    if full_width_strips:
        for row_off in range(0, height, block_size):
            win_h = min(block_size, height - row_off)
            windows.append(Window(0, row_off, width, win_h))
    else:
        for row_off in range(0, height, block_size):
            win_h = min(block_size, height - row_off)
            for col_off in range(0, width, block_size):
                win_w = min(block_size, width - col_off)
                windows.append(Window(col_off, row_off, win_w, win_h))
    return windows


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
) -> np.ndarray:
    """Evaluate local minima on a zero-padded block of shape ``(bands, H+2h, W+2h)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    win_size = op_mask.shape[0]
    half = win_size // 2
    n_bands = padded.shape[0]
    out_h = padded.shape[1] - (2 * half)
    out_w = padded.shape[2] - (2 * half)

    ii, jj = np.ogrid[:win_size, :win_size]
    nbr_mask = op_mask & (ii != half) & (jj != half)
    has_nbrs = bool(np.any(nbr_mask))

    centers = padded[:, half : half + out_h, half : half + out_w]
    windows = sliding_window_view(padded, (win_size, win_size), axis=(1, 2))
    nbr_vals = windows[:, :, :, nbr_mask] if has_nbrs else None

    if combined:
        center_min = np.min(centers, axis=0)
        if nbr_vals is not None:
            cmp_mask = (
                nbr_vals < center_min[None, :, :, None]
                if allow_equal
                else nbr_vals <= center_min[None, :, :, None]
            )
            is_minima = ~np.any(cmp_mask, axis=(0, 3))
        else:
            is_minima = np.ones((out_h, out_w), dtype=bool)

        out = np.zeros((1, out_h, out_w), dtype=np.float64)
        _fill_minima_band(out[0], is_minima, sequencial_out, seq_counters, 0)
        return out

    out = np.zeros((n_bands, out_h, out_w), dtype=np.float64)
    for b_idx in range(n_bands):
        if nbr_vals is not None:
            band_nbrs = nbr_vals[b_idx]
            cmp_mask = (
                band_nbrs < centers[b_idx, :, :, None]
                if allow_equal
                else band_nbrs <= centers[b_idx, :, :, None]
            )
            is_minima = ~np.any(cmp_mask, axis=-1)
        else:
            is_minima = np.ones((out_h, out_w), dtype=bool)

        _fill_minima_band(out[b_idx], is_minima, sequencial_out, seq_counters, b_idx)
    return out


def _run_window_morphology(
    input_img: str,
    output_img: str,
    op_mask: np.ndarray,
    gdalformat: str,
    datatype: int,
    op_kind: str,
    combined: bool = False,
    sequencial_out: bool = False,
    allow_equal: bool = True,
) -> None:
    """Apply a windowed morphological operation across ``input_img`` block-by-block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = rsgispy.get_numpy_datatype(datatype)

    win_size = op_mask.shape[0]
    half = win_size // 2
    block_size = get_default_block_size()
    tqdm_cls = rsgispy.get_tqdm()

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        grid = IntersectionGrid.from_dataset(src)
        out_count = 1 if combined else src.count
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=out_count,
        )

        use_strips = op_kind == "local_minima" and sequencial_out
        windows = _iter_block_windows(
            src.width, src.height, block_size, full_width_strips=use_strips
        )
        seq_counters = np.ones(out_count, dtype=np.int64)

        with tqdm_cls(total=len(windows)) as pbar:
            for out_win in windows:
                halo_win = Window(
                    int(out_win.col_off) - half,
                    int(out_win.row_off) - half,
                    int(out_win.width) + (2 * half),
                    int(out_win.height) + (2 * half),
                )
                padded = src.read(
                    window=halo_win,
                    boundless=True,
                    fill_value=0.0,
                ).astype(np.float32)

                if op_kind == "local_minima":
                    raw_out = _compute_local_minima_block(
                        padded=padded,
                        op_mask=op_mask,
                        combined=combined,
                        allow_equal=allow_equal,
                        sequencial_out=sequencial_out,
                        seq_counters=seq_counters,
                    )
                else:
                    win_view = sliding_window_view(
                        padded, (win_size, win_size), axis=(1, 2)
                    )
                    active = win_view[:, :, :, op_mask]
                    reduce_axes: Tuple[int, ...] = (0, -1) if combined else (-1,)

                    if op_kind == "dilate":
                        raw_out = np.max(active, axis=reduce_axes)
                    elif op_kind == "erode":
                        raw_out = np.min(active, axis=reduce_axes)
                    elif op_kind == "gradiant":
                        raw_out = np.max(active, axis=reduce_axes) - np.min(
                            active, axis=reduce_axes
                        )
                    else:
                        raise rsgispy.RSGISPyException(
                            f"Unsupported morphology operation: '{op_kind}'."
                        )

                    if combined:
                        raw_out = raw_out[np.newaxis, :, :]

                dst.write(_cast_to_output_dtype(raw_out, out_np_dtype), window=out_win)
                pbar.update(1)


def _subtract_images(
    pos_img: str,
    neg_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
) -> None:
    """Write ``pos_img - neg_img`` block-by-block to ``output_img``."""
    # pylint: disable=too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = rsgispy.get_numpy_datatype(datatype)
    block_size = get_default_block_size()
    tqdm_cls = rsgispy.get_tqdm()

    with ExitStack() as stack:
        pos_ds = stack.enter_context(rasterio.open(pos_img, "r"))
        neg_ds = stack.enter_context(rasterio.open(neg_img, "r"))
        grid = IntersectionGrid.from_dataset(pos_ds)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=pos_ds.count,
        )

        windows = _iter_block_windows(
            pos_ds.width, pos_ds.height, block_size, full_width_strips=False
        )
        with tqdm_cls(total=len(windows)) as pbar:
            for out_win in windows:
                diff = pos_ds.read(window=out_win).astype(np.float64) - neg_ds.read(
                    window=out_win
                ).astype(np.float64)
                dst.write(_cast_to_output_dtype(diff, out_np_dtype), window=out_win)
                pbar.update(1)


def _run_two_step_seq(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    op_mask: np.ndarray,
    gdalformat: str,
    datatype: int,
    first_op: str,
    second_op: str,
    niters: int = 1,
) -> None:
    """
    Execute ``niters`` iterations of ``first_op`` followed by ``second_op``
    (e.g., ``erode`` -> ``dilate`` for opening, or ``dilate`` -> ``erode`` for closing).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if niters < 1:
        raise rsgispy.RSGISPyException(
            f"niters must be >= 1 (got {niters})."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        use_user_tmp = bool(tmp_img)
        cur_in = input_img

        for idx in range(niters):
            step_tmp = (
                str(tmp_img)
                if use_user_tmp
                else os.path.join(tmp_dir, f"morph_tmp_{idx}.tif")
            )
            step_fmt = gdalformat if use_user_tmp else "GTIFF"
            is_last = idx == (niters - 1)
            step_out = (
                output_img
                if is_last
                else os.path.join(tmp_dir, f"morph_iter_{idx}.tif")
            )
            out_fmt = gdalformat if is_last else "GTIFF"

            _run_window_morphology(
                input_img=cur_in,
                output_img=step_tmp,
                op_mask=op_mask,
                gdalformat=step_fmt,
                datatype=datatype,
                op_kind=first_op,
            )
            _run_window_morphology(
                input_img=step_tmp,
                output_img=step_out,
                op_mask=op_mask,
                gdalformat=out_fmt,
                datatype=datatype,
                op_kind=second_op,
            )
            cur_in = step_out


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
    size_int = int(op_size)
    if size_int < 1 or size_int % 2 == 0:
        raise rsgispy.RSGISPyException(
            f"op_size must be a positive odd integer (got {op_size})."
        )

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
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform morphological dilation independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate(
            input_img="mask.tif",
            output_img="mask_dilate.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=False,
    )


def image_erode(
    input_img: str,
    output_img: str,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform morphological erosion independently on each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode(
            input_img="mask.tif",
            output_img="mask_erode.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=False,
    )


def image_gradiant(
    input_img: str,
    output_img: str,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Calculate morphological gradient (dilation minus erosion) independently for
    each band of ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output morphological gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_32FLOAT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradiant(
            input_img="img.tif",
            output_img="img_grad.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradiant",
        combined=False,
    )


def image_dilate_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform morphological dilation across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band dilated image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_dilate_combined_out(
            input_img="masks.tif",
            output_img="masks_dilate_comb.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="dilate",
        combined=True,
    )


def image_erode_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform morphological erosion across all bands of ``input_img`` and write a
    single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band eroded image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_erode_combined_out(
            input_img="masks.tif",
            output_img="masks_erode_comb.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="erode",
        combined=True,
    )


def image_gradiant_combined_out(
    input_img: str,
    output_img: str,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Calculate morphological gradient across all bands of ``input_img`` and write
    a single combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band gradient image file.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_32FLOAT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_gradiant_combined_out(
            input_img="img.tif",
            output_img="img_grad_comb.tif",
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    _run_window_morphology(
        input_img=input_img,
        output_img=output_img,
        op_mask=op_mask,
        gdalformat=gdalformat,
        datatype=datatype,
        op_kind="gradiant",
        combined=True,
    )


def image_local_minima(
    input_img: str,
    output_img: str,
    sequencial_out: bool,
    allow_equal: bool,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Identify local minima independently in each band of ``input_img`` using the
    morphological structuring element.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially per band
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``).
    :param allow_equal: If ``True``, neighbour pixels equal to the centre pixel
                        do not disqualify a local minimum.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima(
            input_img="dem.tif",
            output_img="minima.tif",
            sequencial_out=False,
            allow_equal=True,
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
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
    )


def image_local_minima_combined_out(
    input_img: str,
    output_img: str,
    sequencial_out: bool,
    allow_equal: bool,
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Identify local minima across all bands of ``input_img`` and write a single
    combined output band.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output single-band local-minima image file.
    :param sequencial_out: If ``True``, number minima sequentially
                           (``1, 2, 3, ...``); if ``False``, write a binary
                           mask (``1`` / ``0``).
    :param allow_equal: If ``True``, neighbour pixels equal to the minimum band
                        centre value do not disqualify a local minimum.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_local_minima_combined_out(
            input_img="bands.tif",
            output_img="minima_comb.tif",
            sequencial_out=False,
            allow_equal=True,
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
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
    )


def image_opening(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
    niters: int = 1,
) -> None:
    """
    Perform morphological opening (erosion followed by dilation) on ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output opened image file.
    :param tmp_img: Optional path to an intermediate temporary file; if empty
                    (``''``) or ``None``, a temporary directory is used.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).
    :param niters: Number of opening iterations (Default = ``1``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_opening(
            input_img="mask.tif",
            output_img="mask_open.tif",
            tmp_img=None,
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
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
    )


def image_closing(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
    niters: int = 1,
) -> None:
    """
    Perform morphological closing (dilation followed by erosion) on ``input_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output closed image file.
    :param tmp_img: Optional path to an intermediate temporary file; if empty
                    (``''``) or ``None``, a temporary directory is used.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).
    :param niters: Number of closing iterations (Default = ``1``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_closing(
            input_img="mask.tif",
            output_img="mask_close.tif",
            tmp_img=None,
            morph_op_file="",
            use_op_file=False,
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
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
    )


def image_black_top_hat(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform a morphological black top-hat transform (closing minus input image).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output black top-hat image file.
    :param tmp_img: Optional path to an intermediate temporary file; if empty
                    (``''``) or ``None``, a temporary directory is used.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_black_top_hat(
            input_img="dem.tif",
            output_img="dem_bth.tif",
            tmp_img=None,
            morph_op_file="",
            use_op_file=False,
            op_size=5,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img)
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    with tempfile.TemporaryDirectory() as tmp_dir:
        closed_img = (
            str(tmp_img)
            if tmp_img
            else os.path.join(tmp_dir, "black_top_hat_closed.tif")
        )
        closed_fmt = gdalformat if tmp_img else "GTIFF"
        _run_two_step_seq(
            input_img=input_img,
            output_img=closed_img,
            tmp_img=None,
            op_mask=op_mask,
            gdalformat=closed_fmt,
            datatype=datatype,
            first_op="dilate",
            second_op="erode",
            niters=1,
        )
        _subtract_images(
            pos_img=closed_img,
            neg_img=input_img,
            output_img=output_img,
            gdalformat=gdalformat,
            datatype=datatype,
        )


def image_white_top_hat(
    input_img: str,
    output_img: str,
    tmp_img: Optional[str],
    morph_op_file: str,
    use_op_file: bool,
    op_size: int,
    gdalformat: str,
    datatype: int,
) -> None:
    """
    Perform a morphological white top-hat transform (input image minus opening).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output white top-hat image file.
    :param tmp_img: Optional path to an intermediate temporary file; if empty
                    (``''``) or ``None``, a temporary directory is used.
    :param morph_op_file: Path to a ``.gmtxt`` binary structuring-element file.
    :param use_op_file: If ``True``, load operator from ``morph_op_file``;
                        otherwise use a square operator of size ``op_size``.
    :param op_size: Square operator size (used when ``use_op_file`` is ``False``).
    :param gdalformat: GDAL/rasterio output format (e.g., ``'GTIFF'``).
    :param datatype: RSGIS data type constant (e.g., ``rsgispy.TYPE_8UINT``).

    .. code:: python

        import rsgispy
        import rsgispy.imagemorphology

        rsgispy.imagemorphology.image_white_top_hat(
            input_img="dem.tif",
            output_img="dem_wth.tif",
            tmp_img=None,
            morph_op_file="",
            use_op_file=False,
            op_size=5,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img)
    op_mask = _load_morph_operator(morph_op_file, use_op_file, op_size)
    with tempfile.TemporaryDirectory() as tmp_dir:
        opened_img = (
            str(tmp_img)
            if tmp_img
            else os.path.join(tmp_dir, "white_top_hat_opened.tif")
        )
        opened_fmt = gdalformat if tmp_img else "GTIFF"
        _run_two_step_seq(
            input_img=input_img,
            output_img=opened_img,
            tmp_img=None,
            op_mask=op_mask,
            gdalformat=opened_fmt,
            datatype=datatype,
            first_op="erode",
            second_op="dilate",
            niters=1,
        )
        _subtract_images(
            pos_img=input_img,
            neg_img=opened_img,
            output_img=output_img,
            gdalformat=gdalformat,
            datatype=datatype,
        )
