# pylint: disable=too-many-lines
"""
The imageprocess module contains reusable utilities for reading, validating,
and processing raster images in blocks using rasterio.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
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
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.transform import Affine
from rasterio.warp import reproject as rio_reproject
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

import rsgispy
import rsgispy.tools.geometry

DEFAULT_BLOCK_SIZE = 512  # Match COG
ENV_VAR_BLOCK_SIZE = "RSGISPY_BLOCK_SIZE"
ENV_VAR_RSGISLIB_BLOCK_SIZE = "RSGISLIB_BLOCK_SIZE"
_INT_DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32"}


def get_default_block_size() -> int:
    """
    Return the default processing block size in pixels.

    Checks the ``RSGISPY_BLOCK_SIZE`` (and fallback ``RSGISLIB_BLOCK_SIZE``)
    environment variables at runtime, falling back to :data:`DEFAULT_BLOCK_SIZE`.

    :return: Positive integer block size in pixels.

    .. code:: python

        import rsgispy.imageprocess

        blk_size = rsgispy.imageprocess.get_default_block_size()
    """
    raw_val = os.environ.get(ENV_VAR_BLOCK_SIZE) or os.environ.get(
        ENV_VAR_RSGISLIB_BLOCK_SIZE
    )
    if raw_val is not None and raw_val.strip() != "":
        try:
            block_size = int(raw_val.strip())
        except ValueError as err:
            raise rsgispy.RSGISPyException(
                f"Invalid block size in environment variable: '{raw_val}'. "
                "Must be a positive integer."
            ) from err
        if block_size <= 0:
            raise rsgispy.RSGISPyException(
                f"Invalid block size in environment variable: '{raw_val}'. "
                "Must be a positive integer."
            )
        return block_size

    if not isinstance(DEFAULT_BLOCK_SIZE, int) or isinstance(
        DEFAULT_BLOCK_SIZE, bool
    ) or DEFAULT_BLOCK_SIZE <= 0:
        raise rsgispy.RSGISPyException(
            f"DEFAULT_BLOCK_SIZE must be a positive integer, got {DEFAULT_BLOCK_SIZE!r}."
        )
    return DEFAULT_BLOCK_SIZE


def set_default_block_size(block_size: int) -> None:
    """
    Override the default processing block size at runtime by setting the
    ``RSGISPY_BLOCK_SIZE`` environment variable.

    :param block_size: Positive integer block size in pixels.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.set_default_block_size(256)
    """
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or int(block_size) <= 0
    ):
        raise rsgispy.RSGISPyException(
            f"block_size must be a positive integer, got {block_size!r}."
        )
    os.environ[ENV_VAR_BLOCK_SIZE] = str(int(block_size))


def reset_default_block_size() -> None:
    """
    Clear any runtime block size override from the ``RSGISPY_BLOCK_SIZE`` and
    ``RSGISLIB_BLOCK_SIZE`` environment variables so :data:`DEFAULT_BLOCK_SIZE`
    is used again.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.reset_default_block_size()
    """
    os.environ.pop(ENV_VAR_BLOCK_SIZE, None)
    os.environ.pop(ENV_VAR_RSGISLIB_BLOCK_SIZE, None)


def get_gdal_img_creation_opts(gdalformat: str) -> Dict[str, str]:
    """
    Retrieve image creation options for a given GDAL driver format from
    environment variables ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` or
    ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>``, where options are specified as
    colon-separated ``KEY=VALUE`` pairs.

    :param gdalformat: GDAL format string (e.g., 'GTIFF', 'COG', 'KEA').
    :return: Dictionary of creation option key-value strings.

    .. code:: python

        import rsgispy.imageprocess

        opts = rsgispy.imageprocess.get_gdal_img_creation_opts("GTIFF")
    """
    fmt_upper = gdalformat.upper().strip()
    rsgispy_var = f"RSGISPY_IMG_CRT_OPTS_{fmt_upper}"
    rsgislib_var = f"RSGISLIB_IMG_CRT_OPTS_{fmt_upper}"

    raw_opts = os.environ.get(rsgispy_var) or os.environ.get(rsgislib_var)
    opts_dict: Dict[str, str] = {}
    if raw_opts:
        for item in raw_opts.split(":"):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, val = item.split("=", 1)
                opts_dict[key.strip()] = val.strip()
    return opts_dict


@dataclass
class IntersectionGrid:
    """Container for the spatial intersection grid of input images."""

    width: int
    height: int
    transform: Affine
    crs: Optional[rasterio.crs.CRS]
    offsets: Dict[str, Tuple[int, int]]

    @classmethod
    def from_dataset(
        cls,
        src: DatasetReader,
        offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> "IntersectionGrid":
        """Construct an :class:`IntersectionGrid` matching an opened dataset."""
        return cls(
            width=int(src.width),
            height=int(src.height),
            transform=src.transform,
            crs=src.crs,
            offsets={} if offsets is None else offsets,
        )


def ensure_file_exists(file_path: str, label: str = "Input image") -> None:
    """
    Raise :class:`rsgispy.RSGISPyException` if ``file_path`` does not exist.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.ensure_file_exists("img.tif")
    """
    if not os.path.exists(file_path):
        raise rsgispy.RSGISPyException(
            f"{label} file does not exist: '{file_path}'."
        )


def validate_band_index(img_band: int, n_bands: int, img_path: str) -> None:
    """
    Validate that a 1-based band index is within ``[1, n_bands]``.

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.validate_band_index(img_band=1, n_bands=4, img_path="img.tif")
    """
    if img_band < 1 or img_band > n_bands:
        raise rsgispy.RSGISPyException(
            f"Band {img_band} is out of range for image '{img_path}' "
            f"with {n_bands} band(s)."
        )


def build_valid_mask(
    arr: np.ndarray,
    no_data_val: Optional[float] = None,
) -> np.ndarray:
    """
    Build a boolean validity mask for ``arr``, excluding non-finite floating
    values and ``no_data_val``.

    :param arr: Input NumPy array.
    :param no_data_val: Optional no-data value to exclude (supports ``NaN``).
    :return: Boolean NumPy array of the same shape as ``arr`` where ``True``
             marks valid pixels.

    .. code:: python

        import numpy as np
        import rsgispy.imageprocess

        mask = rsgispy.imageprocess.build_valid_mask(
            np.array([0.0, 1.0, np.nan]), no_data_val=0.0
        )
    """
    if np.issubdtype(arr.dtype, np.floating):
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = np.ones(arr.shape, dtype=bool)
    if no_data_val is not None:
        if isinstance(no_data_val, (float, np.floating)) and math.isnan(
            float(no_data_val)
        ):
            if np.issubdtype(arr.dtype, np.floating):
                valid_mask &= ~np.isnan(arr)
        else:
            valid_mask &= arr != no_data_val
    return valid_mask


def read_raster_band(
    input_img: Union[str, os.PathLike],
    img_band: int = 1,
    no_data_val: Optional[float] = None,
) -> Tuple[np.ndarray, Any, Affine, np.ndarray]:
    """
    Read a single raster band along with its CRS, affine transform, and valid
    pixel mask.

    :param input_img: Path to the input raster file.
    :param img_band: 1-based band index to read (Default = ``1``).
    :param no_data_val: Pixel value treated as no-data (reads band header
                        no-data value when ``None``).
    :return: Tuple ``(band_data, crs, transform, valid_mask)``.

    .. code:: python

        import rsgispy.imageprocess

        data, crs, transform, valid = rsgispy.imageprocess.read_raster_band(
            "img.tif", img_band=1
        )
    """
    in_path = os.fspath(input_img)
    ensure_file_exists(in_path, label="Input image")

    try:
        with rasterio.open(in_path, "r") as src:
            validate_band_index(img_band, src.count, in_path)
            eff_nodata = (
                src.nodatavals[img_band - 1]
                if no_data_val is None
                else no_data_val
            )
            band_data = src.read(img_band)
            img_crs = src.crs
            img_transform = src.transform
    except rsgispy.RSGISPyException:
        raise
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{in_path}': {err}"
        ) from err

    return band_data, img_crs, img_transform, build_valid_mask(band_data, eff_nodata)


def validate_odd_window_size(
    size: int, min_size: int = 3, name: str = "filter_size"
) -> int:
    """
    Validate that ``size`` is an odd integer >= ``min_size``.

    :param size: Candidate window size.
    :param min_size: Minimum permitted odd integer size (Default = ``3``).
    :param name: Parameter name used in validation error messages.
    :return: Validated integer size.

    .. code:: python

        import rsgispy.imageprocess

        win_sz = rsgispy.imageprocess.validate_odd_window_size(5, name="op_size")
    """
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or int(size) < min_size
        or int(size) % 2 == 0
    ):
        raise rsgispy.RSGISPyException(
            f"{name} must be an odd integer >= {min_size}, got {size!r}."
        )
    return int(size)


def bbox_from_dataset(src: DatasetReader) -> Tuple[float, float, float, float]:
    """
    Return ``(MinX, MaxX, MinY, MaxY)`` from an opened rasterio dataset.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess

        with rasterio.open("img.tif", "r") as src:
            bbox = rsgispy.imageprocess.bbox_from_dataset(src)
    """
    bounds = src.bounds
    return (
        float(bounds.left),
        float(bounds.right),
        float(bounds.bottom),
        float(bounds.top),
    )


def resolve_band_pair(
    ref_src: DatasetReader,
    flt_src: DatasetReader,
    ref_bands: Optional[Sequence[int]],
    flt_bands: Optional[Sequence[int]],
    in_ref_img: str,
    in_flt_img: str,
) -> Tuple[List[int], List[int]]:
    """
    Validate and resolve paired 1-based band index lists for two raster
    datasets.

    :param ref_src: Opened reference :class:`rasterio.io.DatasetReader`.
    :param flt_src: Opened comparison/floating :class:`rasterio.io.DatasetReader`.
    :param ref_bands: 1-based band indices for ``ref_src`` (or ``None`` for
                      all matching bands).
    :param flt_bands: 1-based band indices for ``flt_src`` (or ``None`` for
                      all matching bands).
    :param in_ref_img: Reference image file path (used in error messages).
    :param in_flt_img: Comparison image file path (used in error messages).
    :return: Tuple ``(ref_band_list, flt_band_list)`` of validated 1-based
             integer band indices.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess as ip

        with rasterio.open("a.tif") as a_src, rasterio.open("b.tif") as b_src:
            ba, bb = ip.resolve_band_pair(a_src, b_src, [1], [1], "a.tif", "b.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if ref_bands is None:
        r_list = list(range(1, ref_src.count + 1))
    else:
        r_list = [int(b) for b in ref_bands]
    if flt_bands is None:
        f_list = list(range(1, flt_src.count + 1))
    else:
        f_list = [int(b) for b in flt_bands]

    if not r_list or not f_list:
        raise rsgispy.RSGISPyException(
            "At least one band must be selected for both reference and floating images."
        )
    if len(r_list) != len(f_list):
        if ref_bands is None and flt_bands is None:
            n_min = min(len(r_list), len(f_list))
            r_list = r_list[:n_min]
            f_list = f_list[:n_min]
        else:
            raise rsgispy.RSGISPyException(
                f"Number of reference bands ({len(r_list)}) must match floating "
                f"bands ({len(f_list)})."
            )

    for b in r_list:
        validate_band_index(b, ref_src.count, in_ref_img)
    for b in f_list:
        validate_band_index(b, flt_src.count, in_flt_img)
    return r_list, f_list


def grids_compatible(
    ref_src: DatasetReader,
    flt_src: DatasetReader,
) -> bool:
    """
    Return ``True`` if two datasets share the same CRS, non-rotated axes, and
    matching pixel size.

    :param ref_src: First opened :class:`rasterio.io.DatasetReader`.
    :param flt_src: Second opened :class:`rasterio.io.DatasetReader`.
    :return: ``True`` if grids have compatible CRS and pixel spacing.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess as ip

        with rasterio.open("a.tif") as a_src, rasterio.open("b.tif") as b_src:
            compat = ip.grids_compatible(a_src, b_src)
    """
    if ref_src.crs is not None and flt_src.crs is not None:
        if pyproj.CRS.from_user_input(ref_src.crs) != pyproj.CRS.from_user_input(
            flt_src.crs
        ):
            return False
    elif (ref_src.crs is None) != (flt_src.crs is None):
        return False

    t_r = ref_src.transform
    t_f = flt_src.transform
    if abs(t_r.b) > 1e-9 or abs(t_r.d) > 1e-9 or abs(t_f.b) > 1e-9 or abs(t_f.d) > 1e-9:
        return False
    return bool(
        np.isclose(t_r.a, t_f.a, rtol=1e-5, atol=1e-9)
        and np.isclose(t_r.e, t_f.e, rtol=1e-5, atol=1e-9)
    )


def compute_overlap_bbox(
    ref_src: DatasetReader,
    flt_src: DatasetReader,
    in_ref_img: str,
    in_flt_img: str,
) -> Tuple[float, float, float, float]:
    """
    Compute the intersecting bounding box ``(xmin, ymin, xmax, ymax)`` of two
    datasets in ``ref_src`` CRS coordinates.

    :param ref_src: Reference :class:`rasterio.io.DatasetReader`.
    :param flt_src: Second :class:`rasterio.io.DatasetReader`.
    :param in_ref_img: Reference image file path (used in error messages).
    :param in_flt_img: Second image file path (used in error messages).
    :return: Overlap bounding box ``(xmin, ymin, xmax, ymax)``.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess as ip

        with rasterio.open("a.tif") as a_src, rasterio.open("b.tif") as b_src:
            ov = ip.compute_overlap_bbox(a_src, b_src, "a.tif", "b.tif")
    """
    ref_bbox = bbox_from_dataset(ref_src)
    flt_bbox = bbox_from_dataset(flt_src)

    if ref_src.crs is not None and flt_src.crs is not None:
        ref_crs = pyproj.CRS.from_user_input(ref_src.crs)
        flt_crs = pyproj.CRS.from_user_input(flt_src.crs)
        if ref_crs != flt_crs:
            flt_bbox = rsgispy.tools.geometry.reproj_bbox(
                flt_bbox, flt_crs, ref_crs
            )

    xmin = max(ref_bbox[0], flt_bbox[0])
    xmax = min(ref_bbox[1], flt_bbox[1])
    ymin = max(ref_bbox[2], flt_bbox[2])
    ymax = min(ref_bbox[3], flt_bbox[3])
    if xmin >= xmax or ymin >= ymax:
        raise rsgispy.RSGISPyException(
            f"Reference '{in_ref_img}' and floating '{in_flt_img}' do not overlap."
        )
    return (xmin, ymin, xmax, ymax)


def extract_overlap_arrays(
    ref_src: DatasetReader,
    flt_src: DatasetReader,
    ref_bands: Sequence[int],
    flt_bands: Sequence[int],
    in_ref_img: str,
    in_flt_img: str,
    ref_no_data: Optional[float] = None,
    flt_no_data: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Read aligned overlapping ``float64`` image arrays ``(C, H, W)`` for
    ``ref_src`` and ``flt_src`` along with valid-pixel masks and geotransform
    metadata.

    When ``ref_src`` and ``flt_src`` share a compatible CRS and pixel spacing,
    both arrays are read directly from aligned pixel windows without resampling;
    otherwise ``flt_src`` is reprojected on the fly onto the overlapping
    reference grid.

    :param ref_src: Opened reference :class:`rasterio.io.DatasetReader`.
    :param flt_src: Opened second :class:`rasterio.io.DatasetReader`.
    :param ref_bands: Sequence of 1-based band indices for ``ref_src``.
    :param flt_bands: Sequence of 1-based band indices for ``flt_src``.
    :param in_ref_img: Reference image file path (used in error messages).
    :param in_flt_img: Second image file path (used in error messages).
    :param ref_no_data: Optional nodata override for ``ref_src``.
    :param flt_no_data: Optional nodata override for ``flt_src``.
    :return: Dictionary containing ``ref_arr``, ``flt_arr``, ``ref_valid``,
             ``flt_valid``, ``ov_transform``, ``flt_win_transform``,
             ``same_grid``, ``height``, and ``width``.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess as ip

        with rasterio.open("a.tif") as a_src, rasterio.open("b.tif") as b_src:
            ov = ip.extract_overlap_arrays(
                a_src, b_src, [1], [1], "a.tif", "b.tif"
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ov_bbox = compute_overlap_bbox(ref_src, flt_src, in_ref_img, in_flt_img)

    ref_win = window_from_bounds(*ov_bbox, transform=ref_src.transform)
    ref_win = ref_win.round_offsets().round_lengths()
    full_ref_win = Window(0, 0, ref_src.width, ref_src.height)
    try:
        ref_win = ref_win.intersection(full_ref_win)
    except rasterio.errors.WindowError as err:
        raise rsgispy.RSGISPyException(
            f"Overlapping region between '{in_ref_img}' and '{in_flt_img}' is empty."
        ) from err
    if ref_win.width <= 0 or ref_win.height <= 0:
        raise rsgispy.RSGISPyException(
            f"Overlapping region between '{in_ref_img}' and '{in_flt_img}' is empty."
        )
    ov_transform = window_transform(ref_win, ref_src.transform)
    out_h = int(ref_win.height)
    out_w = int(ref_win.width)
    ref_arr = ref_src.read(list(ref_bands), window=ref_win).astype(np.float64)

    if grids_compatible(ref_src, flt_src):
        f_col_f, f_row_f = ~flt_src.transform @ (ov_transform.c, ov_transform.f)
        f_col = int(round(f_col_f))
        f_row = int(round(f_row_f))
        r0 = max(0, -f_row)
        c0 = max(0, -f_col)
        r1 = min(out_h, flt_src.height - f_row)
        c1 = min(out_w, flt_src.width - f_col)
        if r1 <= r0 or c1 <= c0:
            raise rsgispy.RSGISPyException(
                f"Overlapping pixel grid between '{in_ref_img}' and '{in_flt_img}' is empty."
            )
        ref_arr = ref_arr[:, r0:r1, c0:c1]
        flt_win = Window(f_col + c0, f_row + r0, c1 - c0, r1 - r0)
        flt_arr = flt_src.read(list(flt_bands), window=flt_win).astype(np.float64)
        ov_transform = ov_transform @ Affine.translation(c0, r0)
        flt_win_transform = window_transform(flt_win, flt_src.transform)
        same_grid = True
    else:
        flt_arr = np.full((len(flt_bands), out_h, out_w), np.nan, dtype=np.float64)
        for idx, b_idx in enumerate(flt_bands):
            s_nd = flt_no_data if flt_no_data is not None else flt_src.nodatavals[b_idx - 1]
            rio_reproject(
                source=rasterio.band(flt_src, b_idx),
                destination=flt_arr[idx],
                src_transform=flt_src.transform,
                src_crs=flt_src.crs,
                src_nodata=s_nd,
                dst_transform=ov_transform,
                dst_crs=ref_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        flt_win_transform = ov_transform
        same_grid = False

    ref_valid = np.isfinite(ref_arr)
    for idx, b_idx in enumerate(ref_bands):
        nd = ref_no_data if ref_no_data is not None else ref_src.nodatavals[b_idx - 1]
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            ref_valid[idx] &= ref_arr[idx] != nd

    flt_valid = np.isfinite(flt_arr)
    for idx, b_idx in enumerate(flt_bands):
        nd = flt_no_data if flt_no_data is not None else flt_src.nodatavals[b_idx - 1]
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            flt_valid[idx] &= flt_arr[idx] != nd

    return {
        "ref_arr": ref_arr,
        "flt_arr": flt_arr,
        "ref_valid": np.all(ref_valid, axis=0),
        "flt_valid": np.all(flt_valid, axis=0),
        "ov_transform": ov_transform,
        "flt_win_transform": flt_win_transform,
        "same_grid": same_grid,
        "height": int(ref_arr.shape[1]),
        "width": int(ref_arr.shape[2]),
    }


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

    .. code:: python

        import rsgispy.imageprocess

        band_defs = rsgispy.imageprocess.normalise_band_defs(
            [rsgispy.imageprocess.BandDefn("b1", "img.tif", 1)]
        )
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

    .. code:: python

        import rasterio
        import rsgispy.imageprocess

        with rasterio.open("a.tif", "r") as ds_a, rasterio.open("b.tif", "r") as ds_b:
            rsgispy.imageprocess.validate_crs_and_res({"a.tif": ds_a, "b.tif": ds_b})
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

    .. code:: python

        import rasterio
        import rsgispy.imageprocess

        with rasterio.open("a.tif", "r") as ds_a, rasterio.open("b.tif", "r") as ds_b:
            grid = rsgispy.imageprocess.compute_spatial_intersection(
                {"a.tif": ds_a, "b.tif": ds_b}
            )
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
    clamp_int: bool = False,
) -> np.ndarray:
    """
    Convert a block calculation result to an array of the target NumPy dtype.

    :param res: Result array or scalar from block calculation.
    :param win_h: Height of the current window in pixels.
    :param win_w: Width of the current window in pixels.
    :param out_np_dtype: Target NumPy dtype for the output raster.
    :param clamp_int: If ``True`` and ``out_np_dtype`` is an integer dtype,
                      clamp values to the dtype's valid range before casting
                      (Default = ``False``).
    :return: NumPy array of shape ``(win_h, win_w)`` or ``(bands, win_h, win_w)``.

    .. code:: python

        import numpy as np
        import rsgispy.imageprocess

        arr = np.ones((256, 256), dtype=np.float64)
        out = rsgispy.imageprocess.cast_result_block(
            arr, 256, 256, np.dtype(np.float32)
        )
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if np.ndim(res) == 0:
        arr = np.full((win_h, win_w), res, dtype=np.float64)
    else:
        arr = np.asarray(res)

    with np.errstate(invalid="ignore"):
        if np.issubdtype(out_np_dtype, np.integer):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if clamp_int:
                info = np.iinfo(out_np_dtype)
                arr = np.clip(arr, info.min, info.max)
        return arr.astype(out_np_dtype, copy=False)


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


def open_input_datasets(
    stack: ExitStack, norm_defs: Sequence[BandDefn]
) -> Dict[str, DatasetReader]:
    """
    Open and validate all unique input raster datasets within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param norm_defs: Validated sequence of BandDefn objects.
    :return: Mapping of file path to opened rasterio DatasetReader.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        band_defs = [rsgispy.imageprocess.BandDefn("b1", "img.tif", 1)]
        with ExitStack() as stack:
            datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
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


def open_multi_image_datasets(
    stack: ExitStack,
    input_imgs: Sequence[Union[str, os.PathLike]],
) -> Tuple[Dict[str, DatasetReader], List[str]]:
    """
    Open and validate all unique input raster datasets from a sequence of
    image paths within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param input_imgs: Sequence of input raster file paths.
    :return: Tuple of ``(datasets, paths)`` where ``datasets`` maps each unique
             file path to an opened rasterio DatasetReader and ``paths`` is the
             resolved list of file paths preserving the input order.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, paths = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["a.tif", "b.tif"]
            )
    """
    if not input_imgs:
        raise rsgispy.RSGISPyException("input_imgs must not be empty.")
    paths: List[str] = []
    datasets: Dict[str, DatasetReader] = {}
    for img in input_imgs:
        img_path = os.fspath(img)
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img_path}'."
            )
        paths.append(img_path)
        if img_path not in datasets:
            try:
                datasets[img_path] = stack.enter_context(
                    rasterio.open(img_path, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{img_path}': {err}"
                ) from err
    return datasets, paths


def open_output_dataset(
    stack: ExitStack,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    grid: IntersectionGrid,
    count: int = 1,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an existing output raster for update or create a new output raster.

    Automatically applies any format creation options configured via
    :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` (such as
    ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>``) as well as any explicit
    ``creation_opts`` dictionary passed to this function.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., 'GTIFF').
    :param out_rio_dtype: Rasterio dtype string (e.g., 'float32').
    :param grid: IntersectionGrid defining output dimensions, transform, and CRS.
    :param count: Number of output bands when creating a new file (Default = 1).
    :param output_exists: If True and the file exists, open in 'r+' mode.
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``).
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, _ = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["in.tif"]
            )
            grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out.tif", "GTIFF", "float32", grid, count=1
            )
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

    img_creation_opts: Dict[str, Any] = get_gdal_img_creation_opts(gdalformat)
    if creation_opts is not None:
        img_creation_opts.update(creation_opts)

    profile: Dict[str, Any] = {
        "driver": gdalformat,
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": out_rio_dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        **img_creation_opts,
    }
    return stack.enter_context(rasterio.open(output_img, "w", **profile))


def create_output_from_dataset(
    stack: ExitStack,
    src: DatasetReader,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    count: Optional[int] = None,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an output raster matching the spatial extent, transform, and CRS of an
    opened reference dataset ``src``.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param src: Reference opened rasterio DatasetReader.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., ``'GTIFF'``).
    :param out_rio_dtype: Rasterio dtype string (e.g., ``'float32'``).
    :param count: Number of output bands when creating a new file
                  (defaults to ``src.count`` if ``None``).
    :param output_exists: If ``True`` and the file exists, open in ``'r+'`` mode.
    :param creation_opts: Optional mapping of GDAL creation options.
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rasterio
        import rsgispy.imageprocess

        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open("in.tif", "r"))
            dst = rsgispy.imageprocess.create_output_from_dataset(
                stack, src, "out.tif", "GTIFF", "float32", count=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid = IntersectionGrid.from_dataset(src)
    return open_output_dataset(
        stack=stack,
        output_img=output_img,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        grid=grid,
        count=int(src.count) if count is None else int(count),
        output_exists=output_exists,
        creation_opts=creation_opts,
    )


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
            assert b_def.input_img is not None
            assert b_def.band_name is not None
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
            res, int(out_win.height), int(out_win.width), out_np_dtype
        )
        if out_arr.ndim == 3:
            dst.write(out_arr, window=out_win)
        else:
            dst.write(out_arr, 1, window=out_win)


def extract_roi_feature_matrix(
    band_arrays: Sequence[Tuple[np.ndarray, Optional[float]]],
    roi_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a dense ``float64`` matrix of valid pixels at flat ROI indices
    ``roi_idx`` across a sequence of ``(band_array, nodata_val)`` pairs.

    Pixels with non-finite values or matching any band's ``nodata_val`` are
    filtered out cleanly.

    :param band_arrays: Sequence of ``(band_array, nodata_val)`` tuples for a
                        processing block.
    :param roi_idx: 1D flat pixel indices inside the block matching the ROI mask.
    :return: Tuple ``(valid_features, valid_roi_indices)`` where
             ``valid_features`` has shape ``(n_valid_pxls, n_bands)``.

    .. code:: python

        import numpy as np
        import rsgispy.imageprocess as ip

        b1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        roi = np.array([0, 1, 2, 3])
        feats, valid_idx = ip.extract_roi_feature_matrix([(b1, None)], roi)
    """
    n_features = len(band_arrays)
    if roi_idx.size == 0:
        return np.empty((0, n_features), dtype=np.float64), roi_idx

    feat_mat = np.empty((roi_idx.size, n_features), dtype=np.float64)
    valid_mask = np.ones(roi_idx.size, dtype=bool)
    for col_idx, (band_arr, band_nd) in enumerate(band_arrays):
        vals = band_arr.ravel()[roi_idx].astype(np.float64, copy=False)
        valid_col = np.isfinite(vals)
        if band_nd is not None and not (
            isinstance(band_nd, (float, np.floating)) and math.isnan(float(band_nd))
        ):
            valid_col &= vals != float(band_nd)
        feat_mat[:, col_idx] = vals
        valid_mask &= valid_col

    return feat_mat[valid_mask], roi_idx[valid_mask]


def run_window_halo_blocks(
    input_img: str,
    output_img: str,
    win_shape: Tuple[int, int],
    block_fn: Callable[[np.ndarray], np.ndarray],
    gdalformat: str = "GTIFF",
    out_rio_dtype: str = "float32",
    out_np_dtype: Optional[np.dtype] = None,
    out_count: Optional[int] = None,
    per_band: bool = False,
    read_dtype: Any = np.float64,
    pad_mode: str = "edge",
    fill_value: float = 0.0,
    out_nodata: Optional[float] = None,
    clamp_int: bool = True,
    full_width_strips: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Process ``input_img`` block-by-block with a halo neighbourhood border and write
    the transformed raster to ``output_img``.

    :param input_img: Path to the input raster file.
    :param output_img: Path to the output raster file.
    :param win_shape: ``(window_height, window_width)`` defining halo radius
                      ``(kh // 2, kw // 2)``.
    :param block_fn: Callable accepting a halo-padded NumPy array (2D when
                     ``per_band=True``, 3D ``(bands, H, W)`` when
                     ``per_band=False``) and returning the core tile output.
    :param gdalformat: Output GDAL/rasterio format name (Default = ``'GTIFF'``).
    :param out_rio_dtype: Rasterio output dtype string (Default = ``'float32'``).
    :param out_np_dtype: Target NumPy dtype for casting (defaults to
                         ``np.dtype(out_rio_dtype)`` when ``None``).
    :param out_count: Number of output bands (defaults to ``src.count``).
    :param per_band: If ``True``, read and pass each band separately as a 2D
                     halo array; if ``False``, read all bands together as a 3D
                     halo cube (Default = ``False``).
    :param read_dtype: NumPy dtype passed to :func:`read_window_with_halo`
                       (Default = ``float64``).
    :param pad_mode: Border padding mode passed to :func:`read_window_with_halo`
                     (Default = ``'edge'``).
    :param fill_value: Fill constant when ``pad_mode='constant'``
                       (Default = ``0.0``).
    :param out_nodata: Optional output nodata metadata value.
    :param clamp_int: Clamp integer outputs to valid dtype bounds
                      (Default = ``True``).
    :param full_width_strips: If ``True``, iterate in full-width horizontal strips
                              (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

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
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=out_count,
        )
        if out_nodata is not None:
            dst.nodata = out_nodata

        blk_width = src.width if full_width_strips else None
        for out_win in iter_dataset_windows(
            src.height, src.width, block_width=blk_width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            if per_band:
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
                            clamp_int=clamp_int,
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
                dst.write(
                    cast_result_block(
                        res,
                        win_h,
                        win_w,
                        target_np_dtype,
                        clamp_int=clamp_int,
                    ),
                    window=out_win,
                )
