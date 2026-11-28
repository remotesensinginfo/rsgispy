"""
Spatial grid metadata, band definition normalisation, grid intersection,
overlap extraction, and single-band array readers for
:mod:`rsgispy.imageprocess`.
"""

from dataclasses import dataclass
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.warp import reproject as rio_reproject
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

import rsgispy
import rsgispy.tools.geometry
from rsgispy.imageprocess._masks import (
    build_valid_cube_mask,
    build_valid_mask,
    ensure_file_exists,
    validate_band_index,
)


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


#: Largest allowed grid-origin offset, as a fraction of a pixel, between
#: inputs combined pixel by pixel.
GRID_ALIGN_TOL = 1e-6


def _check_grid_alignment(datasets: Dict[str, Any], first_ds: Any) -> None:
    """
    Raise :class:`rsgispy.RSGISPyException` when the pixel grid of any dataset
    is offset from that of ``first_ds`` by a fraction of a pixel (more than
    :data:`GRID_ALIGN_TOL` of the pixel size): the inputs would otherwise be
    combined with a silent sub-pixel shift.
    """
    ref = first_ds.transform
    res_x, res_y = abs(ref.a), abs(ref.e)
    for path, ds in datasets.items():
        dx = (ds.transform.c - ref.c) / res_x
        dy = (ds.transform.f - ref.f) / res_y
        off_x = abs(dx - round(dx))
        off_y = abs(dy - round(dy))
        if off_x > GRID_ALIGN_TOL or off_y > GRID_ALIGN_TOL:
            raise rsgispy.RSGISPyException(
                f"The pixel grid of '{path}' is not aligned with that of the "
                f"first input (offset by {off_x:.6g} x {off_y:.6g} pixels). "
                "Resample it onto the same grid first, e.g. with "
                "rsgispy.imageutils.warp.resample_img_to_match."
            )


def same_pixel_grid(
    ds_a: Any, ds_b: Any, *, missing_crs_matches: bool = False
) -> bool:
    """
    Return ``True`` when two open rasters are on the same pixel grid: the
    same width and height, transforms equal to within ``1e-6`` of the pixel
    size (:data:`GRID_ALIGN_TOL`), and the same CRS.

    The CRS rule: two defined CRSs must be equal. When one or both are
    ``None``, the grids match only if both are ``None``, unless
    ``missing_crs_matches=True``, in which case a missing CRS matches any CRS
    (for inputs that are known to share a CRS but may not record it).

    :param ds_a: First open raster (``rasterio`` dataset).
    :param ds_b: Second open raster.
    :param missing_crs_matches: Treat a missing CRS as matching any CRS
                                (Default = ``False``).
    :return: ``True`` if the grids match.
    """
    if ds_a.width != ds_b.width or ds_a.height != ds_b.height:
        return False
    tol = GRID_ALIGN_TOL * max(abs(ds_a.transform.a), 1e-12)
    if not np.allclose(
        np.asarray(ds_a.transform)[:6], np.asarray(ds_b.transform)[:6],
        rtol=0.0, atol=tol,
    ):
        return False
    if ds_a.crs is None or ds_b.crs is None:
        return missing_crs_matches or (ds_a.crs is None and ds_b.crs is None)
    return ds_a.crs == ds_b.crs


def compute_spatial_intersection(
    datasets: Dict[str, DatasetReader]
) -> IntersectionGrid:
    """
    Compute the spatial intersection grid and per-dataset pixel offsets.

    The inputs must share a CRS, a pixel size and a pixel grid: an
    :class:`rsgispy.RSGISPyException` is raised when their grid origins are
    offset by a fraction of a pixel (more than ``1e-6`` of the pixel size);
    resample them onto one grid first (e.g. with
    :func:`rsgispy.imageutils.warp.resample_img_to_match`).

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
    _check_grid_alignment(datasets, first_ds)

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


def bbox_from_dataset(src: DatasetReader) -> rsgispy.BBox:
    """
    Return a :class:`rsgispy.BBox` ``(MinX, MaxX, MinY, MaxY)`` with optional
    ``epsg`` metadata from an opened rasterio dataset.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess

        with rasterio.open("img.tif", "r") as src:
            bbox = rsgispy.imageprocess.bbox_from_dataset(src)
    """
    bounds = src.bounds
    epsg = src.crs.to_epsg() if src.crs is not None else None
    return rsgispy.BBox(
        float(bounds.left),
        float(bounds.right),
        float(bounds.bottom),
        float(bounds.top),
        epsg=epsg,
    )


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

    ref_nds = [
        ref_no_data if ref_no_data is not None else ref_src.nodatavals[b_idx - 1]
        for b_idx in ref_bands
    ]
    flt_nds = [
        flt_no_data if flt_no_data is not None else flt_src.nodatavals[b_idx - 1]
        for b_idx in flt_bands
    ]

    return {
        "ref_arr": ref_arr,
        "flt_arr": flt_arr,
        "ref_valid": build_valid_cube_mask(ref_arr, ref_nds),
        "flt_valid": build_valid_cube_mask(flt_arr, flt_nds),
        "ov_transform": ov_transform,
        "flt_win_transform": flt_win_transform,
        "same_grid": same_grid,
        "height": int(ref_arr.shape[1]),
        "width": int(ref_arr.shape[2]),
    }
