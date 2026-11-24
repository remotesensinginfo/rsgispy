"""
Input file/band/window validation, validity mask construction, and ROI feature
extraction for :mod:`rsgispy.imageprocess`.
"""

import math
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from rasterio.io import DatasetReader

import rsgispy


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


def build_valid_mask(
    arr: np.ndarray,
    no_data_val: Optional[float] = None,
    msk_arr: Optional[np.ndarray] = None,
    valid_msk_val: Optional[Union[int, float]] = 1,
) -> np.ndarray:
    """
    Build a boolean validity mask for ``arr``, excluding non-finite floating
    values, ``no_data_val``, and pixels outside ``msk_arr == valid_msk_val``.

    :param arr: Input NumPy array.
    :param no_data_val: Optional no-data value to exclude (supports ``NaN``).
    :param msk_arr: Optional external mask array matching ``arr.shape``.
    :param valid_msk_val: Value in ``msk_arr`` marking valid pixels
                          (Default = ``1``).
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
    if msk_arr is not None and valid_msk_val is not None:
        valid_mask &= msk_arr == valid_msk_val
    return valid_mask


def build_valid_cube_mask(
    arr_nd: np.ndarray,
    no_data_val: Optional[Union[float, Sequence[Optional[float]]]] = None,
    msk_arr: Optional[np.ndarray] = None,
    valid_msk_val: Optional[Union[int, float]] = 1,
    axis: int = 0,
) -> np.ndarray:
    """
    Build a reduced boolean validity mask across bands/features along ``axis``
    (default ``axis=0`` for ``(B, H, W)`` raster cubes, or ``axis=1`` for
    ``(N, B)`` feature tables).

    :param arr_nd: Input multi-band NumPy array (e.g. ``(B, H, W)`` or ``(N, B)``).
    :param no_data_val: Optional scalar no-data value or per-band sequence of
                        no-data values.
    :param msk_arr: Optional external mask array matching the reduced spatial shape.
    :param valid_msk_val: Value in ``msk_arr`` marking valid pixels (Default = ``1``).
    :param axis: Axis indexing bands/features to reduce with logical AND
                 (Default = ``0``).
    :return: Reduced boolean validity mask.
    """
    if np.issubdtype(arr_nd.dtype, np.floating):
        valid = np.all(np.isfinite(arr_nd), axis=axis)
    else:
        out_shape = tuple(
            dim for idx, dim in enumerate(arr_nd.shape) if idx != (axis % arr_nd.ndim)
        )
        valid = np.ones(out_shape, dtype=bool)

    if no_data_val is not None:
        if isinstance(no_data_val, Sequence) and not isinstance(
            no_data_val, (str, bytes)
        ):
            n_bands = arr_nd.shape[axis]
            for b_idx in range(min(n_bands, len(no_data_val))):
                nd = no_data_val[b_idx]
                if nd is None or (
                    isinstance(nd, (float, np.floating)) and math.isnan(float(nd))
                ):
                    continue
                slc = [slice(None)] * arr_nd.ndim
                slc[axis] = b_idx
                valid &= arr_nd[tuple(slc)] != nd
        else:
            nd_scalar = float(no_data_val)
            if not math.isnan(nd_scalar):
                valid &= np.all(arr_nd != nd_scalar, axis=axis)

    if msk_arr is not None and valid_msk_val is not None:
        valid &= msk_arr == valid_msk_val
    return valid


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
        feat_mat[:, col_idx] = vals
        valid_mask &= build_valid_mask(vals, band_nd)

    return feat_mat[valid_mask], roi_idx[valid_mask]
