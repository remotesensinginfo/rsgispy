"""
The imageutils.masking module provides raster masking and valid-data mask
generation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    build_valid_cube_mask,
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    iter_dataset_windows,
    normalise_creation_options,
    on_output_close,
    open_multi_image_datasets,
    open_output_dataset,
    pop_stats_on_close,
    rsgis_dtype_from_np_dtype,
    validate_band_index,
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
    Class storing an image file path, logical name, optional list of bands,
    and optional output band names.

    :param file_name: Path to the raster image file.
    :param name: Logical name describing the image.
    :param bands: Optional list of 1-based band indices to consider.
    :param band_names: Optional sequence of output band/feature names matching
                       ``bands`` (or all bands in ``file_name`` when
                       ``bands`` is ``None``).
    """

    def __init__(
        self,
        file_name: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
        bands: Optional[Sequence[int]] = None,
        band_names: Optional[Sequence[str]] = None,
    ):
        self.file_name: Optional[str] = (
            os.fspath(file_name) if file_name is not None else None
        )
        self.name = name
        self.bands: Optional[List[int]] = (
            [int(b) for b in bands] if bands is not None else None
        )
        if band_names is not None:
            names_list = [str(bn) for bn in band_names]
            if any(not bn.strip() for bn in names_list):
                raise rsgispy.RSGISPyException(
                    "band_names must contain non-empty strings."
                )
            if self.bands is not None and len(names_list) != len(self.bands):
                raise rsgispy.RSGISPyException(
                    f"Number of band_names ({len(names_list)}) must match "
                    f"number of bands ({len(self.bands)})."
                )
            self.band_names: Optional[List[str]] = names_list
        else:
            self.band_names = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        out: Dict[str, Any] = {
            "file_name": self.file_name,
            "name": self.name,
            "bands": list(self.bands) if self.bands is not None else None,
        }
        if self.band_names is not None:
            out["band_names"] = list(self.band_names)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageBandInfo":
        """Construct an :class:`ImageBandInfo` from a dictionary."""
        if not isinstance(data, Mapping):
            raise rsgispy.RSGISPyException(
                "ImageBandInfo data must be a dictionary."
            )
        return cls(
            file_name=data.get("file_name"),
            name=data.get("name"),
            bands=data.get("bands"),
            band_names=data.get("band_names"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageBandInfo):
            return NotImplemented
        return (
            self.file_name == other.file_name
            and self.name == other.name
            and self.bands == other.bands
            and self.band_names == other.band_names
        )

    def __repr__(self) -> str:
        if self.band_names is None:
            return (
                f"ImageBandInfo(file_name={self.file_name!r}, "
                f"name={self.name!r}, bands={self.bands!r})"
            )
        return (
            f"ImageBandInfo(file_name={self.file_name!r}, "
            f"name={self.name!r}, bands={self.bands!r}, "
            f"band_names={self.band_names!r})"
        )


def mask_img(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    out_value: float,
    mask_value: Union[float, int, Sequence[Union[float, int]]],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Mask ``input_img`` using ``in_msk_img``: wherever band 1 of ``in_msk_img``
    matches any value in ``mask_value``, output pixels across all bands are set
    to ``out_value``; otherwise input pixel values are preserved.

    :param input_img: Path to the input image file.
    :param in_msk_img: Path to the mask image file (band 1 is used).
    :param output_img: Path to the output masked image file.
    :param out_value: Value written to masked pixels in the output image.
    :param mask_value: Scalar or sequence of mask values in ``in_msk_img`` to mask.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param no_data_val: Optional output no-data value (defaults to ``input_img``
                        no-data value when ``None``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_img(
            "in_img.tif", "msk.tif", "out_img.tif", 0.0, 0,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
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
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(
            stack, [input_img, in_msk_img]
        )
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        in_col, in_row = grid.offsets[paths[0]]
        msk_col, msk_row = grid.offsets[paths[1]]

        eff_dtype = (
            rsgis_dtype_from_np_dtype(in_ds.dtypes[0])
            if datatype is None
            else datatype
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=in_ds.count,
            output_exists=False,
            creation_opts=norm_opts,
        )
        eff_nd = no_data_val if no_data_val is not None else in_ds.nodata
        if eff_nd is not None:
            dst.nodata = eff_nd
        pop_stats_on_close(dst, calc_stats, no_data_val=eff_nd, use_pbar=use_pbar)
        for bidx in range(1, in_ds.count + 1):
            desc = in_ds.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
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
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic mask (``1`` where all bands are finite, ``0`` where
    any band contains ``NaN`` or ``+/-Inf``).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_finite_mask("in_img.tif", "finite_msk.tif")
    """
    # pylint: disable=too-many-arguments,too-many-locals
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        # Runs once the writer has closed (for a COG output, on the
        # temporary GeoTIFF before it is converted).
        on_output_close(dst, set_img_thematic)
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=no_data_val,
            thematic=True,
            use_pbar=use_pbar,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            arr = in_ds.read(window=out_win, out_dtype=np.float64)
            finite_msk = build_valid_cube_mask(arr, axis=0).astype(out_np_dtype)
            dst.write(finite_msk, 1, window=out_win)


def gen_valid_mask(
    input_imgs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic valid-pixel mask (``1`` where all bands across all
    input images are finite and differ from ``no_data_val``, else ``0``).

    :param input_imgs: Input image path or sequence of image paths.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Value representing no-data (Default = ``0.0``; pass ``None``
                        to check only finiteness).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_valid_mask(
            ["img1.tif", "img2.tif"], "valid_msk.tif", no_data_val=0.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(input_imgs, (str, os.PathLike)):
        img_list: List[Union[str, os.PathLike]] = [input_imgs]
    else:
        img_list = list(input_imgs)

    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=norm_opts,
        )
        # Runs once the writer has closed (for a COG output, on the
        # temporary GeoTIFF before it is converted).
        on_output_close(dst, set_img_thematic)
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=no_data_val,
            thematic=True,
            use_pbar=use_pbar,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
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
                valid &= build_valid_cube_mask(
                    arr,
                    no_data_val=float(no_data_val) if no_data_val is not None else None,
                    axis=0,
                )
            dst.write(valid.astype(out_np_dtype), 1, window=out_win)


def gen_img_edge_mask(
    input_img: str,
    output_img: str,
    n_edge_pxls: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic mask highlighting image border pixels (``1`` within
    ``n_edge_pxls`` of any edge of ``input_img``, ``0`` in the interior).

    :param input_img: Path to the input reference image.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param n_edge_pxls: Number of edge pixels along each side to mark as ``1``.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_img_edge_mask(
            "in_img.tif", "edge_msk.tif", 5, gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    edge_px = int(n_edge_pxls)
    if edge_px < 0:
        raise rsgispy.RSGISPyException(
            f"n_edge_pxls must be non-negative, got {n_edge_pxls}."
        )

    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        # Runs once the writer has closed (for a COG output, on the
        # temporary GeoTIFF before it is converted).
        on_output_close(dst, set_img_thematic)
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=no_data_val,
            thematic=True,
            use_pbar=use_pbar,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
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
            dst.write(edge.astype(out_np_dtype), 1, window=out_win)


def mask_all_band_zero_vals(
    input_img: str,
    output_img: str,
    out_val: float = 1.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Replace zero/non-positive values in individual bands with ``out_val`` where
    the sum across all bands at that pixel is ``> 0``; pixels where all bands are
    ``<= 0`` remain unchanged.

    :param input_img: Path to the input multi-band image file.
    :param output_img: Path to the output image file.
    :param out_val: Replacement value when a pixel has positive band sum but
                    a band value ``<= 0`` (Default = ``1.0``).
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param no_data_val: Optional output no-data value (defaults to ``input_img``
                        no-data value when ``None``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_all_band_zero_vals(
            "in_img.tif", "out_img.tif", out_val=1.0, gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    norm_opts = normalise_creation_options(creation_opts)
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img])
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]
        eff_dtype = (
            rsgis_dtype_from_np_dtype(in_ds.dtypes[0])
            if datatype is None
            else datatype
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=in_ds.count,
            output_exists=False,
            creation_opts=norm_opts,
        )
        eff_nd = no_data_val if no_data_val is not None else in_ds.nodata
        if eff_nd is not None:
            dst.nodata = eff_nd
        pop_stats_on_close(dst, calc_stats, no_data_val=eff_nd, use_pbar=use_pbar)
        for bidx in range(1, in_ds.count + 1):
            desc = in_ds.descriptions[bidx - 1]
            if desc:
                dst.set_band_description(bidx, desc)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
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
    output_img: Union[str, os.PathLike],
    tmp_dir: str = "",
    default_no_data_val: Optional[float] = 0.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a binary thematic valid-pixel mask (``1`` where all bands of every
    image in ``img_band_info`` are finite and differ from their image's
    no-data value, ``0`` otherwise).

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects describing
                          input images and optional bands to check.
    :param output_img: Path to the output mask image file.
    :param tmp_dir: Temporary directory path (accepted for API compatibility).
    :param default_no_data_val: Fallback no-data value checked when an input
                                band has no embedded raster no-data value
                                (``None`` in ``ds.nodatavals``; Default = ``0.0``).
                                Pass ``None`` to check only finiteness for bands
                                without embedded no-data metadata.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data metadata value to set on ``output_img``
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.classification.classdata
        import rsgispy.imageutils

        img_info = [
            rsgispy.classification.classdata.ImageBandInfo("img1.tif", "img1", [1, 2]),
            rsgispy.classification.classdata.ImageBandInfo("img2.tif", "img2"),
        ]
        rsgispy.imageutils.create_valid_mask(
            img_info, "valid_msk.tif", gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    if not img_band_info:
        raise rsgispy.RSGISPyException(
            "img_band_info must contain at least one ImageBandInfo item."
        )
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    out_msk_path = os.fspath(output_img)

    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    norm_opts = normalise_creation_options(creation_opts)

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
                validate_band_index(b_idx, ds.count, path)
            band_checks.append((path, ds, bands))

        dst = open_output_dataset(
            stack=stack,
            output_img=out_msk_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            output_exists=False,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        # Runs once the writer has closed (for a COG output, on the
        # temporary GeoTIFF before it is converted).
        on_output_close(dst, set_img_thematic)
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=no_data_val,
            thematic=True,
            use_pbar=use_pbar,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
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
                    eff_nd = (
                        default_no_data_val if nd is None else float(nd)
                    )
                    valid &= build_valid_mask(
                        arr,
                        no_data_val=float(eff_nd) if eff_nd is not None else None,
                    )
            dst.write(valid.astype(out_np_dtype), 1, window=out_win)
