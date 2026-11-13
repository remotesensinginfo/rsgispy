"""
The imageutils.masking module provides raster masking and valid-data mask
generation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    get_default_block_size,
    open_multi_image_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.colourtable import set_img_thematic

__all__ = [
    "ImageBandInfo",
    "mask_img",
    "gen_finite_mask",
    "gen_valid_mask",
    "gen_img_edge_mask",
    "mask_all_band_zero_vals",
    "create_valid_mask",
]


class ImageBandInfo:
    """
    Class storing an image file path, logical name, and optional list of bands.

    :param file_name: Path to the raster image file.
    :param name: Logical name describing the image.
    :param bands: Optional list of 1-based band indices to consider.
    """

    def __init__(
        self,
        file_name: Optional[str] = None,
        name: Optional[str] = None,
        bands: Optional[Sequence[int]] = None,
    ):
        self.file_name = file_name
        self.name = name
        self.bands = list(bands) if bands is not None else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageBandInfo):
            return NotImplemented
        return (
            self.file_name == other.file_name
            and self.name == other.name
            and self.bands == other.bands
        )

    def __repr__(self) -> str:
        return (
            f"ImageBandInfo(file_name={self.file_name!r}, "
            f"name={self.name!r}, bands={self.bands!r})"
        )


def _iter_grid_windows(
    width: int, height: int, block_size: Optional[int] = None
):
    """Yield ``Window`` objects covering ``(width, height)`` in ``block_size`` tiles."""
    bsize = get_default_block_size() if block_size is None else int(block_size)
    if bsize <= 0:
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {bsize}."
        )
    for row_off in range(0, height, bsize):
        win_h = min(bsize, height - row_off)
        for col_off in range(0, width, bsize):
            win_w = min(bsize, width - col_off)
            yield Window(col_off, row_off, win_w, win_h)


def mask_img(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    gdalformat: str,
    datatype: int,
    out_value: float,
    mask_value: Union[float, int, Sequence[Union[float, int]]],
    options: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Mask ``input_img`` using ``in_msk_img``: wherever band 1 of ``in_msk_img``
    matches any value in ``mask_value``, output pixels across all bands are set
    to ``out_value``; otherwise input pixel values are preserved.

    :param input_img: Path to the input image file.
    :param in_msk_img: Path to the mask image file (band 1 is used).
    :param output_img: Path to the output masked image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output image.
    :param out_value: Value written to masked pixels in the output image.
    :param mask_value: Scalar or sequence of mask values in ``in_msk_img`` to mask.
    :param options: Optional GDAL creation options mapping.

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_img(
            "in_img.tif", "msk.tif", "out_img.tif", "GTIFF",
            rsgispy.TYPE_32FLOAT, 0.0, 0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(mask_value, (int, float, np.number)):
        mask_vals = [float(mask_value)]
    else:
        mask_vals = [float(v) for v in mask_value]
    if not mask_vals:
        raise rsgispy.RSGISPyException(
            "mask_value must contain at least one numeric value."
        )

    has_nan_msk = any(math.isnan(v) for v in mask_vals)
    finite_mask_vals = [v for v in mask_vals if not math.isnan(v)]

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    creation_opts = dict(options) if isinstance(options, Mapping) else None

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(
            stack, [input_img, in_msk_img]
        )
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        in_col, in_row = grid.offsets[paths[0]]
        msk_col, msk_row = grid.offsets[paths[1]]

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=in_ds.count,
            output_exists=False,
            creation_opts=creation_opts,
        )
        if in_ds.nodata is not None:
            dst.nodata = in_ds.nodata
        for bidx in range(1, in_ds.count + 1):
            desc = in_ds.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for out_win in _iter_grid_windows(grid.width, grid.height):
            win_w = int(out_win.width)
            win_h = int(out_win.height)
            in_win = Window(
                in_col + int(out_win.col_off),
                in_row + int(out_win.row_off),
                win_w,
                win_h,
            )
            msk_win = Window(
                msk_col + int(out_win.col_off),
                msk_row + int(out_win.row_off),
                win_w,
                win_h,
            )
            data = in_ds.read(window=in_win, out_dtype=np.float64)
            msk_arr = msk_ds.read(1, window=msk_win, out_dtype=np.float64)

            if finite_mask_vals:
                is_masked = np.isin(msk_arr, finite_mask_vals)
            else:
                is_masked = np.zeros(msk_arr.shape, dtype=bool)
            if has_nan_msk:
                is_masked = is_masked | np.isnan(msk_arr)

            data[:, is_masked] = float(out_value)
            out_arr = cast_result_block(data, win_h, win_w, out_np_dtype)
            dst.write(out_arr, window=out_win)


def gen_finite_mask(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Generate a binary thematic mask (``1`` where all bands are finite, ``0`` where
    any band contains ``NaN`` or ``+/-Inf``).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output mask image file (8-bit unsigned).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param options: Optional GDAL creation options mapping.

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_finite_mask("in_img.tif", "finite_msk.tif")
    """
    creation_opts = dict(options) if isinstance(options, Mapping) else None
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )

        for out_win in _iter_grid_windows(grid.width, grid.height):
            arr = in_ds.read(window=out_win, out_dtype=np.float64)
            finite_msk = np.all(np.isfinite(arr), axis=0).astype(np.uint8)
            dst.write(finite_msk, 1, window=out_win)

    set_img_thematic(output_img)


def gen_valid_mask(
    input_imgs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_img: str,
    gdalformat: str = "GTIFF",
    no_data_val: float = 0.0,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Generate a binary thematic valid-pixel mask (``1`` where all bands across all
    input images differ from ``no_data_val``, else ``0``).

    :param input_imgs: Input image path or sequence of image paths.
    :param output_img: Path to the output mask image file (8-bit unsigned).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param no_data_val: Value representing no-data (Default = ``0.0``).
    :param options: Optional GDAL creation options mapping.

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_valid_mask(
            ["img1.tif", "img2.tif"], "valid_msk.tif", no_data_val=0.0
        )
    """
    # pylint: disable=too-many-locals
    if isinstance(input_imgs, (str, os.PathLike)):
        img_list: List[Union[str, os.PathLike]] = [input_imgs]
    else:
        img_list = list(input_imgs)

    nd_is_nan = math.isnan(float(no_data_val))
    creation_opts = dict(options) if isinstance(options, Mapping) else None

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )

        for out_win in _iter_grid_windows(grid.width, grid.height):
            win_w = int(out_win.width)
            win_h = int(out_win.height)
            valid = np.ones((win_h, win_w), dtype=bool)
            for path in paths:
                ds = datasets[path]
                base_col, base_row = grid.offsets[path]
                in_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                arr = ds.read(window=in_win, out_dtype=np.float64)
                if nd_is_nan:
                    valid &= np.all(~np.isnan(arr), axis=0)
                else:
                    valid &= np.all(arr != float(no_data_val), axis=0)
            dst.write(valid.astype(np.uint8), 1, window=out_win)

    set_img_thematic(output_img)


def gen_img_edge_mask(
    input_img: str,
    output_img: str,
    gdalformat: str,
    n_edge_pxls: int,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Generate a binary thematic mask highlighting image border pixels (``1`` within
    ``n_edge_pxls`` of any edge of ``input_img``, ``0`` in the interior).

    :param input_img: Path to the input reference image.
    :param output_img: Path to the output mask image file (8-bit unsigned).
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param n_edge_pxls: Number of edge pixels along each side to mark as ``1``.
    :param options: Optional GDAL creation options mapping.

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_img_edge_mask(
            "in_img.tif", "edge_msk.tif", "GTIFF", 5
        )
    """
    # pylint: disable=too-many-locals
    edge_px = int(n_edge_pxls)
    if edge_px < 0:
        raise rsgispy.RSGISPyException(
            f"n_edge_pxls must be non-negative, got {n_edge_pxls}."
        )

    creation_opts = dict(options) if isinstance(options, Mapping) else None
    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=creation_opts,
        )

        for out_win in _iter_grid_windows(grid.width, grid.height):
            row0 = int(out_win.row_off)
            col0 = int(out_win.col_off)
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            rows = np.arange(row0, row0 + win_h)[:, np.newaxis]
            cols = np.arange(col0, col0 + win_w)[np.newaxis, :]
            edge = (
                (rows < edge_px)
                | (rows >= grid.height - edge_px)
                | (cols < edge_px)
                | (cols >= grid.width - edge_px)
            )
            dst.write(edge.astype(np.uint8), 1, window=out_win)

    set_img_thematic(output_img)


def mask_all_band_zero_vals(
    input_img: str,
    output_img: str,
    gdalformat: str,
    out_val: float = 1.0,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Replace zero/non-positive values in individual bands with ``out_val`` where
    the sum across all bands at that pixel is ``> 0``; pixels where all bands are
    ``<= 0`` remain unchanged.

    :param input_img: Path to the input multi-band image file.
    :param output_img: Path to the output image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param out_val: Replacement value when a pixel has positive band sum but
                    a band value ``<= 0`` (Default = ``1.0``).
    :param options: Optional GDAL creation options mapping.

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_all_band_zero_vals(
            "in_img.tif", "out_img.tif", "GTIFF", out_val=1.0
        )
    """
    # pylint: disable=too-many-locals
    creation_opts = dict(options) if isinstance(options, Mapping) else None
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]
        out_rio_dtype = in_ds.dtypes[0]
        out_np_dtype = np.dtype(out_rio_dtype)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=in_ds.count,
            output_exists=False,
            creation_opts=creation_opts,
        )
        if in_ds.nodata is not None:
            dst.nodata = in_ds.nodata
        for bidx in range(1, in_ds.count + 1):
            desc = in_ds.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for out_win in _iter_grid_windows(grid.width, grid.height):
            win_w = int(out_win.width)
            win_h = int(out_win.height)
            data = in_ds.read(window=out_win, out_dtype=np.float64)
            valid_pxl = np.sum(data, axis=0) > 0.0
            replace_mask = (data <= 0.0) & valid_pxl[np.newaxis, :, :]
            data[replace_mask] = float(out_val)
            out_arr = cast_result_block(data, win_h, win_w, out_np_dtype)
            dst.write(out_arr, window=out_win)


def create_valid_mask(
    img_band_info: Sequence[ImageBandInfo],
    out_msk_file: str,
    gdalformat: str,
    tmp_dir: str = "",
) -> None:
    """
    Create a binary thematic valid-pixel mask (``1`` where all bands of every
    image in ``img_band_info`` differ from their image's no-data value, ``0``
    otherwise).

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects describing
                          input images and optional bands to check.
    :param out_msk_file: Path to the output mask image file.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``).
    :param tmp_dir: Temporary directory path (accepted for API compatibility).

    .. code:: python

        import rsgispy.imageutils.masking

        img_info = [
            rsgispy.imageutils.masking.ImageBandInfo("img1.tif", "img1", [1, 2]),
            rsgispy.imageutils.masking.ImageBandInfo("img2.tif", "img2"),
        ]
        rsgispy.imageutils.masking.create_valid_mask(
            img_info, "valid_msk.tif", "GTIFF"
        )
    """
    # pylint: disable=too-many-locals,too-many-branches
    if not img_band_info:
        raise rsgispy.RSGISPyException(
            "img_band_info must contain at least one ImageBandInfo item."
        )
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)

    img_paths: List[str] = []
    for info in img_band_info:
        if not info.file_name:
            raise rsgispy.RSGISPyException(
                "Each ImageBandInfo object must specify a valid 'file_name'."
            )
        img_paths.append(os.fspath(info.file_name))

    with ExitStack() as stack:
        datasets, resolved_paths = open_multi_image_datasets(stack, img_paths)
        grid = compute_spatial_intersection(datasets)

        band_checks = []
        for info, path in zip(img_band_info, resolved_paths):
            ds = datasets[path]
            bands = (
                [int(b) for b in info.bands]
                if info.bands is not None
                else list(range(1, ds.count + 1))
            )
            if not bands:
                raise rsgispy.RSGISPyException(
                    f"ImageBandInfo for '{path}' specified an empty band list."
                )
            for b_idx in bands:
                if b_idx < 1 or b_idx > ds.count:
                    raise rsgispy.RSGISPyException(
                        f"Band {b_idx} is out of range for '{path}' ({ds.count} bands)."
                    )
            band_checks.append((path, ds, bands))

        dst = open_output_dataset(
            stack=stack,
            output_img=out_msk_file,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            grid=grid,
            count=1,
            output_exists=False,
        )

        for out_win in _iter_grid_windows(grid.width, grid.height):
            win_w = int(out_win.width)
            win_h = int(out_win.height)
            valid = np.ones((win_h, win_w), dtype=bool)
            for path, ds, bands in band_checks:
                base_col, base_row = grid.offsets[path]
                in_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                for b_idx in bands:
                    arr = ds.read(b_idx, window=in_win, out_dtype=np.float64)
                    nd = ds.nodatavals[b_idx - 1]
                    eff_nd = 0.0 if nd is None else float(nd)
                    if math.isnan(eff_nd):
                        valid &= ~np.isnan(arr)
                    else:
                        valid &= arr != eff_nd
            dst.write(valid.astype(np.uint8), 1, window=out_win)

    set_img_thematic(out_msk_file)
