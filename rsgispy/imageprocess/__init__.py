"""
The imageprocess module contains reusable utilities for reading, validating,
and processing raster images in blocks using rasterio.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

import rsgispy

DEFAULT_BLOCK_SIZE = 256


@dataclass
class IntersectionGrid:
    """Container for the spatial intersection grid of input images."""

    width: int
    height: int
    transform: Affine
    crs: Optional[rasterio.crs.CRS]
    offsets: Dict[str, Tuple[int, int]]


class BandDefn:
    """
    Definition of an input image band mapped to a variable name.

    :param band_name: Name of the band variable (e.g., 'b1').
    :param input_img: Path to the input image file.
    :param img_band: 1-based band index within the input image.
    """

    def __init__(
        self,
        band_name: Optional[str] = None,
        input_img: Optional[str] = None,
        img_band: Optional[int] = None,
    ):
        self.band_name = band_name
        self.input_img = input_img
        self.img_band = img_band

    def to_dict(self) -> Dict[str, Union[Optional[str], Optional[int]]]:
        """Return a dictionary representation of the band definition."""
        return {
            "band_name": self.band_name,
            "input_img": self.input_img,
            "img_band": self.img_band,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BandDefn):
            return NotImplemented
        return (
            self.band_name == other.band_name
            and self.input_img == other.input_img
            and self.img_band == other.img_band
        )

    def __repr__(self) -> str:
        return (
            f"BandDefn(band_name={self.band_name!r}, "
            f"input_img={self.input_img!r}, img_band={self.img_band!r})"
        )


def normalise_band_defs(
    band_defs: Sequence[Union[BandDefn, Any, Dict[str, Union[str, int]]]]
) -> List[BandDefn]:
    """
    Validate and normalise a sequence of BandDefn objects (or dicts).

    :param band_defs: Sequence of BandDefn objects or dicts.
    :return: Deduplicated list of validated BandDefn objects.
    """
    if not band_defs:
        raise rsgispy.RSGISPyException(
            "band_defs must be a non-empty list of BandDefn objects."
        )

    normalised: List[BandDefn] = []
    seen_vars: Dict[str, Tuple[str, int]] = {}

    for item in band_defs:
        if isinstance(item, dict):
            b_name = item.get("band_name")
            in_img = item.get("input_img")
            i_band = item.get("img_band")
        else:
            b_name = getattr(item, "band_name", None)
            in_img = getattr(item, "input_img", None)
            i_band = getattr(item, "img_band", None)

        if not b_name or not isinstance(b_name, str):
            raise rsgispy.RSGISPyException(
                "Each BandDefn must specify a valid non-empty string 'band_name'."
            )
        if not in_img or not isinstance(in_img, (str, os.PathLike)):
            raise rsgispy.RSGISPyException(
                f"BandDefn '{b_name}' must specify a valid 'input_img' file path."
            )
        in_img_str = os.fspath(in_img)
        if not os.path.exists(in_img_str):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{in_img_str}'."
            )
        if i_band is None or int(i_band) < 1:
            raise rsgispy.RSGISPyException(
                f"BandDefn '{b_name}' must specify a 1-based 'img_band' >= 1."
            )
        i_band_int = int(i_band)

        if b_name in seen_vars:
            if seen_vars[b_name] != (in_img_str, i_band_int):
                raise rsgispy.RSGISPyException(
                    f"Duplicate band_name '{b_name}' pointing to different inputs."
                )
            continue

        seen_vars[b_name] = (in_img_str, i_band_int)
        normalised.append(
            BandDefn(band_name=b_name, input_img=in_img_str, img_band=i_band_int)
        )

    return normalised


def validate_crs_and_res(
    datasets: Dict[str, DatasetReader],
    first_ds: Optional[DatasetReader] = None,
) -> None:
    """
    Validate that all datasets share the same CRS and pixel resolution.

    :param datasets: Mapping of file path to opened rasterio DatasetReader.
    :param first_ds: Reference dataset (defaults to the first dataset in ``datasets``).
    """
    if not datasets:
        raise rsgispy.RSGISPyException("No datasets provided for validation.")

    if first_ds is None:
        first_ds = next(iter(datasets.values()))

    first_res_x = abs(first_ds.transform.a)
    first_res_y = abs(first_ds.transform.e)

    for path, ds in datasets.items():
        if ds.crs != first_ds.crs:
            raise rsgispy.RSGISPyException(
                f"Input image '{path}' has a different CRS ({ds.crs}) "
                f"from '{first_ds.name}' ({first_ds.crs})."
            )
        res_x = abs(ds.transform.a)
        res_y = abs(ds.transform.e)
        if not (
            math.isclose(res_x, first_res_x, rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(res_y, first_res_y, rel_tol=1e-5, abs_tol=1e-7)
        ):
            raise rsgispy.RSGISPyException(
                f"Input image '{path}' has a different pixel resolution "
                f"({res_x}, {res_y}) from ({first_res_x}, {first_res_y})."
            )


def compute_spatial_intersection(
    datasets: Dict[str, DatasetReader]
) -> IntersectionGrid:
    """
    Compute the spatial intersection grid and per-dataset pixel offsets.

    :param datasets: Mapping of file path to opened rasterio DatasetReader.
    :return: IntersectionGrid describing the output extent and input pixel offsets.
    """
    if not datasets:
        raise rsgispy.RSGISPyException(
            "No datasets provided to compute spatial intersection."
        )

    ds_list = list(datasets.values())
    first_ds = ds_list[0]

    if all(
        ds.width == first_ds.width
        and ds.height == first_ds.height
        and ds.transform == first_ds.transform
        and ds.crs == first_ds.crs
        for ds in ds_list[1:]
    ):
        return IntersectionGrid(
            width=first_ds.width,
            height=first_ds.height,
            transform=first_ds.transform,
            crs=first_ds.crs,
            offsets={path: (0, 0) for path in datasets},
        )

    validate_crs_and_res(datasets, first_ds)

    bounds = (
        max(ds.bounds.left for ds in ds_list),
        max(ds.bounds.bottom for ds in ds_list),
        min(ds.bounds.right for ds in ds_list),
        min(ds.bounds.top for ds in ds_list),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise rsgispy.RSGISPyException(
            "Input images do not have a valid spatial overlap."
        )

    offsets: Dict[str, Tuple[int, int]] = {}
    out_size: Tuple[int, int] = (0, 0)
    out_transform = first_ds.transform

    for idx, (path, ds) in enumerate(datasets.items()):
        win = (
            window_from_bounds(*bounds, transform=ds.transform)
            .round_offsets()
            .round_lengths()
        )
        if int(win.width) <= 0 or int(win.height) <= 0:
            raise rsgispy.RSGISPyException(
                "Input images do not have a valid pixel overlap."
            )
        if idx == 0:
            out_size = (int(win.width), int(win.height))
            out_transform = window_transform(win, ds.transform)
        elif (int(win.width), int(win.height)) != out_size:
            raise rsgispy.RSGISPyException(
                "Overlapping pixel dimensions across input images do not match."
            )
        offsets[path] = (int(win.col_off), int(win.row_off))

    return IntersectionGrid(
        width=out_size[0],
        height=out_size[1],
        transform=out_transform,
        crs=first_ds.crs,
        offsets=offsets,
    )


def cast_result_block(
    res: Union[np.ndarray, float, int],
    win_h: int,
    win_w: int,
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """
    Convert a block calculation result to an array of the target NumPy dtype.

    :param res: Result array or scalar from block calculation.
    :param win_h: Height of the current window in pixels.
    :param win_w: Width of the current window in pixels.
    :param out_np_dtype: Target NumPy dtype for the output raster.
    :return: NumPy array of shape ``(win_h, win_w)`` or ``(bands, win_h, win_w)``.
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if np.ndim(res) == 0:
        arr = np.full((win_h, win_w), res, dtype=np.float64)
    else:
        arr = np.asarray(res)

    with np.errstate(invalid="ignore"):
        if np.issubdtype(out_np_dtype, np.integer):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(out_np_dtype, copy=False)


def open_input_datasets(
    stack: ExitStack, norm_defs: Sequence[BandDefn]
) -> Dict[str, DatasetReader]:
    """
    Open and validate all unique input raster datasets within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param norm_defs: Validated sequence of BandDefn objects.
    :return: Mapping of file path to opened rasterio DatasetReader.
    """
    datasets: Dict[str, DatasetReader] = {}
    for b_def in norm_defs:
        assert b_def.input_img is not None
        assert b_def.img_band is not None
        if b_def.input_img not in datasets:
            try:
                datasets[b_def.input_img] = stack.enter_context(
                    rasterio.open(b_def.input_img, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{b_def.input_img}': {err}"
                ) from err

        ds = datasets[b_def.input_img]
        if b_def.img_band > ds.count:
            raise rsgispy.RSGISPyException(
                f"Requested band {b_def.img_band} exceeds band count ({ds.count}) "
                f"for image '{b_def.input_img}'."
            )
    return datasets


def open_output_dataset(
    stack: ExitStack,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    grid: IntersectionGrid,
    count: int = 1,
    output_exists: bool = False,
) -> DatasetWriter:
    """
    Open an existing output raster for update or create a new output raster.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., 'GTIFF').
    :param out_rio_dtype: Rasterio dtype string (e.g., 'float32').
    :param grid: IntersectionGrid defining output dimensions, transform, and CRS.
    :param count: Number of output bands when creating a new file (Default = 1).
    :param output_exists: If True and the file exists, open in 'r+' mode.
    :return: Opened rasterio DatasetWriter.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if output_exists and os.path.exists(output_img):
        dst = stack.enter_context(rasterio.open(output_img, "r+"))
        if dst.width != grid.width or dst.height != grid.height:
            raise rsgispy.RSGISPyException(
                f"Existing output image '{output_img}' dimensions "
                f"({dst.width}x{dst.height}) do not match calculated "
                f"output dimensions ({grid.width}x{grid.height})."
            )
        return dst

    profile = {
        "driver": gdalformat,
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": out_rio_dtype,
        "crs": grid.crs,
        "transform": grid.transform,
    }
    return stack.enter_context(rasterio.open(output_img, "w", **profile))


def process_blocks(
    dst: DatasetWriter,
    datasets: Dict[str, DatasetReader],
    norm_defs: Sequence[BandDefn],
    grid: IntersectionGrid,
    block_func: Callable[[Dict[str, np.ndarray]], Union[np.ndarray, float, int]],
    out_np_dtype: np.dtype,
    block_size: int = DEFAULT_BLOCK_SIZE,
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
    :param block_size: Window block size in pixels (Default = 256).
    :param read_dtype: NumPy dtype to cast input bands into when reading (Default = float64).
    :param use_pbar: If True, display a tqdm progress bar during processing (Default = True).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    tqdm_cls = rsgispy.get_tqdm()
    total_blocks = math.ceil(grid.height / block_size) * math.ceil(
        grid.width / block_size
    )

    with tqdm_cls(total=total_blocks, disable=not use_pbar) as pbar:
        for row_off in range(0, grid.height, block_size):
            win_h = min(block_size, grid.height - row_off)
            for col_off in range(0, grid.width, block_size):
                win_w = min(block_size, grid.width - col_off)
                out_win = Window(col_off, row_off, win_w, win_h)

                block_vars: Dict[str, np.ndarray] = {}
                for b_def in norm_defs:
                    assert b_def.input_img is not None
                    assert b_def.band_name is not None
                    base_col, base_row = grid.offsets[b_def.input_img]
                    in_win = Window(
                        base_col + col_off, base_row + row_off, win_w, win_h
                    )
                    block_vars[b_def.band_name] = datasets[b_def.input_img].read(
                        b_def.img_band, window=in_win, out_dtype=read_dtype
                    )

                res = block_func(block_vars)
                out_arr = cast_result_block(res, win_h, win_w, out_np_dtype)
                if out_arr.ndim == 3:
                    dst.write(out_arr, window=out_win)
                else:
                    dst.write(out_arr, 1, window=out_win)
                pbar.update(1)
