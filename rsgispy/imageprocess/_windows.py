"""
Window iteration, halo-padded reads, and block-processing drivers for
:mod:`rsgispy.imageprocess`.
"""

from contextlib import ExitStack
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import rasterio
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess._arrays import (
    BandDefn,
    IntersectionGrid,
    compute_spatial_intersection,
)
from rsgispy.imageprocess._dtype_nodata import (
    cast_result_block,
    get_default_block_size,
)
from rsgispy.imageprocess._io import create_output_from_dataset
from rsgispy.imageprocess._masks import ensure_file_exists, validate_band_index


def read_window_with_halo(
    src: DatasetReader,
    band: Union[int, Sequence[int], None],
    out_win: Window,
    pad: Union[int, Tuple[int, int]],
    out_dtype: Any = np.float64,
    pad_mode: str = "edge",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Read a raster window expanded by ``pad`` pixels, padding image borders
    using ``pad_mode`` (``'edge'`` by default, or ``'constant'`` with
    ``fill_value``).

    :param src: Opened rasterio :class:`rasterio.io.DatasetReader`.
    :param band: 1-based band index (returns 2D array), sequence of 1-based
                 band indices, or ``None`` for all bands (returns 3D array).
    :param out_win: Core output :class:`rasterio.windows.Window`.
    :param pad: Halo padding around ``out_win`` as an integer (symmetric) or
                ``(pad_y, pad_x)`` tuple.
    :param out_dtype: NumPy dtype to cast read values into (Default = ``float64``).
    :param pad_mode: ``numpy.pad`` border mode (e.g., ``'edge'`` or
                     ``'constant'``, Default = ``'edge'``).
    :param fill_value: Fill constant used when ``pad_mode='constant'``
                       (Default = ``0.0``).
    :return: Padded NumPy array of shape ``(win_h + 2*pad_y, win_w + 2*pad_x)``
             for a single band or ``(bands, win_h + 2*pad_y, win_w + 2*pad_x)``.

    .. code:: python

        import rasterio
        from rasterio.windows import Window
        import rsgispy.imageprocess

        with rasterio.open("img.tif", "r") as src:
            blk = rsgispy.imageprocess.read_window_with_halo(
                src, 1, Window(0, 0, 128, 128), pad=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(pad, tuple):
        pad_y, pad_x = int(pad[0]), int(pad[1])
    else:
        pad_y = pad_x = int(pad)

    col_off = int(out_win.col_off)
    row_off = int(out_win.row_off)
    win_w = int(out_win.width)
    win_h = int(out_win.height)

    r0 = row_off - pad_y
    r1 = row_off + win_h + pad_y
    c0 = col_off - pad_x
    c1 = col_off + win_w + pad_x

    cr0 = max(0, r0)
    cr1 = min(src.height, r1)
    cc0 = max(0, c0)
    cc1 = min(src.width, c1)

    read_win = Window(cc0, cr0, cc1 - cc0, cr1 - cr0)
    if band is None:
        arr = src.read(window=read_win, out_dtype=out_dtype)
    else:
        arr = src.read(band, window=read_win, out_dtype=out_dtype)

    spatial_pads = ((cr0 - r0, r1 - cr1), (cc0 - c0, c1 - cc1))
    if any(p > 0 for pair in spatial_pads for p in pair):
        pads = ((0, 0), *spatial_pads) if arr.ndim == 3 else spatial_pads
        if pad_mode == "constant":
            arr = np.pad(arr, pads, mode="constant", constant_values=fill_value)
        else:
            arr = np.pad(arr, pads, mode=pad_mode)  # type: ignore[call-overload]
    return arr


def iter_dataset_windows(
    height: int,
    width: int,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    block_width: Optional[int] = None,
) -> Iterator[Window]:
    """
    Iterate over processing block windows across a raster of dimensions
    ``(height, width)``, optionally displaying a tqdm progress bar.

    :param height: Raster height in pixels.
    :param width: Raster width in pixels.
    :param block_size: Window block height (and default width) in pixels. If
                       ``None``, resolved at runtime via
                       :func:`get_default_block_size`.
    :param use_pbar: If ``True``, display a tqdm progress bar during iteration
                     (Default = ``True``).
    :param block_width: Optional window block width in pixels. Defaults to
                        ``block_size`` when ``None``.
    :return: Iterator yielding :class:`rasterio.windows.Window` tiles.

    .. code:: python

        import rsgispy.imageprocess

        for win in rsgispy.imageprocess.iter_dataset_windows(1024, 1024, block_size=256):
            print(win)
    """
    if block_size is None:
        blk_sz = get_default_block_size()
    elif int(block_size) <= 0:
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size}."
        )
    else:
        blk_sz = int(block_size)

    if block_width is None:
        blk_w = blk_sz
    elif int(block_width) <= 0:
        raise rsgispy.RSGISPyException(
            f"block_width must be a positive integer, got {block_width}."
        )
    else:
        blk_w = int(block_width)

    tqdm_cls = rsgispy.get_tqdm()
    total_blocks = math.ceil(height / blk_sz) * math.ceil(width / blk_w)
    with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
        for row_off in range(0, height, blk_sz):
            win_h = min(blk_sz, height - row_off)
            for col_off in range(0, width, blk_w):
                win_w = min(blk_w, width - col_off)
                yield Window(col_off, row_off, win_w, win_h)
                pbar.update(1)


def iter_image_blocks(
    datasets: Dict[str, DatasetReader],
    norm_defs: Sequence[BandDefn],
    grid: IntersectionGrid,
    block_size: Optional[int] = None,
    read_dtype: Any = np.float64,
    use_pbar: bool = True,
) -> Iterator[Tuple[Window, Dict[str, np.ndarray]]]:
    """
    Iterate over spatial intersection windows, yielding the output window and a
    dictionary of band arrays for each block.

    :param datasets: Mapping of file path to opened rasterio DatasetReader.
    :param norm_defs: Sequence of BandDefn objects specifying bands to read.
    :param grid: IntersectionGrid describing the output extent and input pixel offsets.
    :param block_size: Window block size in pixels. If ``None``, resolved at runtime
                       via :func:`get_default_block_size`.
    :param read_dtype: NumPy dtype to cast input bands into when reading (Default = float64).
    :param use_pbar: If True, display a tqdm progress bar during iteration (Default = True).
    :return: Iterator yielding ``(out_win, block_vars)``.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        band_defs = [rsgispy.imageprocess.BandDefn("b1", "img.tif", 1)]
        with ExitStack() as stack:
            datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
            grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
            for win, block_vars in rsgispy.imageprocess.iter_image_blocks(
                datasets, band_defs, grid
            ):
                print(win, block_vars["b1"].shape)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    for out_win in iter_dataset_windows(
        grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
    ):
        col_off = int(out_win.col_off)
        row_off = int(out_win.row_off)
        win_w = int(out_win.width)
        win_h = int(out_win.height)

        block_vars: Dict[str, np.ndarray] = {}
        for b_def in norm_defs:
            if b_def.input_img is None or b_def.band_name is None:
                raise rsgispy.RSGISPyException(
                    "BandDefn must define both input_img and band_name."
                )
            base_col, base_row = grid.offsets[b_def.input_img]
            in_win = Window(base_col + col_off, base_row + row_off, win_w, win_h)
            block_vars[b_def.band_name] = datasets[b_def.input_img].read(
                b_def.img_band, window=in_win, out_dtype=read_dtype
            )

        yield out_win, block_vars


def iter_single_image_band_blocks(
    input_img: str,
    block_size: Optional[int] = None,
    read_dtype: Any = np.float64,
    use_pbar: bool = True,
) -> Iterator[Tuple[Window, List[np.ndarray]]]:
    """
    Iterate over blocks of a single raster dataset, yielding ``(out_win, band_arrays)``
    where ``band_arrays`` is a list of 2D band arrays ``[band_1, ..., band_n]`` for
    each block.

    :param input_img: Path to the input image file.
    :param block_size: Window block size in pixels. If ``None``, resolved at runtime
                       via :func:`get_default_block_size`.
    :param read_dtype: NumPy dtype to cast bands into when reading (Default = float64).
    :param use_pbar: If True, display a tqdm progress bar during iteration (Default = True).
    :return: Iterator yielding ``(out_win, [band_1_arr, ..., band_n_arr])`` tuples.

    .. code:: python

        import rsgispy.imageprocess

        for win, bands in rsgispy.imageprocess.iter_single_image_band_blocks("img.tif"):
            print(win, len(bands))
    """
    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
        norm_defs = [
            BandDefn(band_name=f"b{idx + 1}", input_img=input_img, img_band=idx + 1)
            for idx in range(n_bands)
        ]
        datasets = {input_img: src}
        grid = compute_spatial_intersection(datasets)

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            block_size=block_size,
            read_dtype=read_dtype,
            use_pbar=use_pbar,
        ):
            yield out_win, [block_vars[f"b{idx + 1}"] for idx in range(n_bands)]


def process_blocks(
    dst: DatasetWriter,
    datasets: Dict[str, DatasetReader],
    norm_defs: Sequence[BandDefn],
    grid: IntersectionGrid,
    block_func: Callable[[Dict[str, np.ndarray]], Union[np.ndarray, float, int]],
    out_np_dtype: np.dtype,
    block_size: Optional[int] = None,
    read_dtype: Any = np.float64,
    use_pbar: bool = True,
) -> None:
    """
    Read input bands in blocks, apply ``block_func``, and write results to ``dst``.

    :param dst: Destination rasterio DatasetWriter.
    :param datasets: Mapping of file path to opened rasterio DatasetReader.
    :param norm_defs: Sequence of BandDefn objects specifying bands to read.
    :param grid: IntersectionGrid describing the output extent and input pixel offsets.
    :param block_func: Callable accepting ``Dict[str, np.ndarray]`` of band arrays for
                       the current window and returning a 2D/3D array or scalar.
    :param out_np_dtype: Target NumPy dtype for the output raster.
    :param block_size: Window block size in pixels. If ``None``, resolved at runtime
                       via :func:`get_default_block_size`.
    :param read_dtype: NumPy dtype to cast input bands into when reading (Default = float64).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).

    .. code:: python

        from contextlib import ExitStack
        import numpy as np
        import rsgispy.imageprocess

        band_defs = [rsgispy.imageprocess.BandDefn("b1", "in.tif", 1)]
        with ExitStack() as stack:
            datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
            grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out.tif", "GTIFF", "float32", grid, count=1
            )
            rsgispy.imageprocess.process_blocks(
                dst, datasets, band_defs, grid,
                lambda b: b["b1"] * 2.0, np.dtype(np.float32)
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    for out_win, block_vars in iter_image_blocks(
        datasets=datasets,
        norm_defs=norm_defs,
        grid=grid,
        block_size=block_size,
        read_dtype=read_dtype,
        use_pbar=use_pbar,
    ):
        res = block_func(block_vars)
        out_arr = cast_result_block(
            res,
            int(out_win.height),
            int(out_win.width),
            out_np_dtype,
            nodata=dst.nodata,
        )
        if out_arr.ndim == 3:
            dst.write(out_arr, window=out_win)
        else:
            dst.write(out_arr, 1, window=out_win)


def run_window_halo_blocks(
    input_img: str,
    output_img: str,
    win_shape: Tuple[int, int],
    block_fn: Callable[[np.ndarray], np.ndarray],
    *,
    gdalformat: str = "GTIFF",
    out_rio_dtype: str = "float32",
    out_np_dtype: Optional[np.dtype] = None,
    out_count: Optional[int] = None,
    per_band: bool = False,
    read_dtype: Any = np.float64,
    pad_mode: str = "edge",
    fill_value: float = 0.0,
    out_no_data_val: Optional[float] = None,
    full_width_strips: bool = False,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    band_idx: Optional[int] = None,
) -> None:
    """
    Process ``input_img`` block-by-block with a halo neighbourhood border and write
    the transformed raster to ``output_img``.

    :param input_img: Path to the input raster file.
    :param output_img: Path to the output raster file.
    :param win_shape: ``(window_height, window_width)`` defining halo radius
                      ``(kh // 2, kw // 2)``.
    :param block_fn: Callable accepting a halo-padded NumPy array (2D when
                     ``per_band=True`` or ``band_idx`` is set, 3D
                     ``(bands, H, W)`` when ``per_band=False`` and
                     ``band_idx=None``) and returning the core tile output.
    :param gdalformat: Output GDAL/rasterio format name (Default = ``'GTIFF'``).
    :param out_rio_dtype: Rasterio output dtype string (Default = ``'float32'``).
    :param out_np_dtype: Target NumPy dtype for casting (defaults to
                         ``np.dtype(out_rio_dtype)`` when ``None``).
    :param out_count: Number of output bands (defaults to ``src.count``).
    :param per_band: If ``True``, read and pass each band separately as a 2D
                     halo array; if ``False``, read all bands together as a 3D
                     halo cube unless ``band_idx`` is set (Default = ``False``).
    :param read_dtype: NumPy dtype passed to :func:`read_window_with_halo`
                       (Default = ``float64``).
    :param pad_mode: Border padding mode passed to :func:`read_window_with_halo`
                     (Default = ``'edge'``).
    :param fill_value: Fill constant when ``pad_mode='constant'``
                       (Default = ``0.0``).
    :param out_no_data_val: Optional output no-data metadata value.
    :param full_width_strips: If ``True``, iterate in full-width horizontal strips
                              (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.
    :param band_idx: Optional 1-based band index to read alone as a 2D halo
                     array and write as a single-band output (Default = ``None``).

    .. code:: python

        import numpy as np
        import rsgispy.imageprocess as ip

        ip.run_window_halo_blocks(
            "in.tif",
            "out.tif",
            win_shape=(3, 3),
            block_fn=lambda h: h[1:-1, 1:-1],
            per_band=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    target_np_dtype = (
        np.dtype(out_rio_dtype)
        if out_np_dtype is None
        else np.dtype(out_np_dtype)
    )
    pad = (int(win_shape[0]) // 2, int(win_shape[1]) // 2)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if band_idx is not None:
            validate_band_index(band_idx, src.count, input_img)
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=out_count,
            creation_opts=creation_opts,
        )
        if out_no_data_val is not None:
            dst.nodata = out_no_data_val

        blk_width = src.width if full_width_strips else None
        for out_win in iter_dataset_windows(
            src.height, src.width, block_width=blk_width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if band_idx is not None:
                halo = read_window_with_halo(
                    src,
                    int(band_idx),
                    out_win,
                    pad=pad,
                    out_dtype=read_dtype,
                    pad_mode=pad_mode,
                    fill_value=fill_value,
                )
                res = block_fn(halo)
                out_blk = cast_result_block(
                    res,
                    win_h,
                    win_w,
                    target_np_dtype,
                )
                if out_blk.ndim == 2:
                    dst.write(out_blk, 1, window=out_win)
                else:
                    dst.write(out_blk, window=out_win)
            elif per_band:
                for b_idx in range(1, src.count + 1):
                    halo = read_window_with_halo(
                        src,
                        b_idx,
                        out_win,
                        pad=pad,
                        out_dtype=read_dtype,
                        pad_mode=pad_mode,
                        fill_value=fill_value,
                    )
                    res = block_fn(halo)
                    dst.write(
                        cast_result_block(
                            res,
                            win_h,
                            win_w,
                            target_np_dtype,
                        ),
                        b_idx,
                        window=out_win,
                    )
            else:
                halo = read_window_with_halo(
                    src,
                    None,
                    out_win,
                    pad=pad,
                    out_dtype=read_dtype,
                    pad_mode=pad_mode,
                    fill_value=fill_value,
                )
                res = block_fn(halo)
                out_blk = cast_result_block(
                    res,
                    win_h,
                    win_w,
                    target_np_dtype,
                )
                if out_blk.ndim == 2:
                    dst.write(out_blk, 1, window=out_win)
                else:
                    dst.write(out_blk, window=out_win)
