# pylint: disable=too-many-lines
"""
The changedetect.imgchngmthds module provides multi-temporal change detection
algorithms using ``rasterio`` and ``numpy``:

* :func:`image_difference` — per-band difference between base and change images
* :func:`image_ratio` — per-band ratio between base and change images
* :func:`change_vector_analysis` — Change Vector Analysis (CVA)
* :func:`slow_feature_analysis` — Iteratively Reweighted Slow Feature Analysis (ISFA)
* :func:`multivariate_alteration_detection` — Iteratively Reweighted MAD (IR-MAD)

Functions requiring ``scipy`` (:func:`slow_feature_analysis`,
:func:`multivariate_alteration_detection`) raise :class:`rsgispy.RSGISPyException`
if ``scipy`` is not installed.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

import rsgispy
from rsgispy.imageprocess import ensure_file_exists, validate_band_index
from rsgispy.imageutils.arrays import (
    create_img_from_array_ref_img,
    get_img_data_as_arr,
)
from rsgispy.imageutils.checks import check_img_file_comparison
from rsgispy.imageutils.metadata import get_img_band_count, get_img_size
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.tools.stats import standardise_img_data
from rsgispy.tools.thresholding import calc_otsu_threshold

try:
    import scipy.linalg  # pylint: disable=import-error
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

try:
    from sklearn.cluster import KMeans  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    KMeans = None
    HAVE_SKLEARN = False

__all__: List[str] = [
    "image_difference",
    "image_ratio",
    "change_vector_analysis",
    "slow_feature_analysis",
    "multivariate_alteration_detection",
]


def _prepare_change_inputs(
    in_base_img: str,
    in_chng_img: str,
    vld_msk_img: Optional[str],
    vld_msk_val: int,
    img_base_bands: Optional[Sequence[int]],
    img_chng_bands: Optional[Sequence[int]],
    base_img_no_data: float,
    chng_img_no_data: float,
    apply_std: bool,
    min_valid_pxls: int = 0,
) -> Tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate matching base/change rasters, extract valid pixels, and optionally
    standardise each band.

    Returns ``(n_bands, img_size_x, img_size_y, n_pxls, pxl_ids, base_flat, chng_flat)``
    where ``base_flat`` and ``chng_flat`` have shape ``(n_bands, n_valid)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(in_base_img, "Base image")
    ensure_file_exists(in_chng_img, "Change image")

    imgs_match = check_img_file_comparison(
        in_base_img,
        in_chng_img,
        test_n_bands=False,
        test_eql_bbox=True,
        print_errors=True,
    )
    if not imgs_match:
        raise rsgispy.RSGISPyException(
            "Input base and change images do not match."
        )

    if img_base_bands is not None:
        if img_chng_bands is None:
            raise rsgispy.RSGISPyException(
                "img_base_bands is not None but img_chng_bands is None."
            )
        base_count = get_img_band_count(in_base_img)
        for bidx in img_base_bands:
            validate_band_index(int(bidx), base_count, in_base_img)
        n_base_bands = len(img_base_bands)
    else:
        n_base_bands = get_img_band_count(in_base_img)

    if img_chng_bands is not None:
        if img_base_bands is None:
            raise rsgispy.RSGISPyException(
                "img_chng_bands is not None but img_base_bands is None."
            )
        chng_count = get_img_band_count(in_chng_img)
        for bidx in img_chng_bands:
            validate_band_index(int(bidx), chng_count, in_chng_img)
        n_chng_bands = len(img_chng_bands)
    else:
        n_chng_bands = get_img_band_count(in_chng_img)

    if n_base_bands != n_chng_bands:
        raise rsgispy.RSGISPyException(
            "The number of bands specified for the two images do not match."
        )
    n_bands = n_base_bands

    img_size_x, img_size_y = get_img_size(in_base_img)
    n_pxls = img_size_x * img_size_y

    base_arr = get_img_data_as_arr(in_base_img, img_bands=img_base_bands).astype(
        np.float64, copy=False
    )
    chng_arr = get_img_data_as_arr(in_chng_img, img_bands=img_chng_bands).astype(
        np.float64, copy=False
    )

    base_arr = np.nan_to_num(
        base_arr,
        copy=True,
        nan=base_img_no_data,
        posinf=base_img_no_data,
        neginf=base_img_no_data,
    )
    chng_arr = np.nan_to_num(
        chng_arr,
        copy=True,
        nan=chng_img_no_data,
        posinf=chng_img_no_data,
        neginf=chng_img_no_data,
    )

    if vld_msk_img is not None:
        ensure_file_exists(vld_msk_img, "Valid mask image")
        msk_size_x, msk_size_y = get_img_size(vld_msk_img)
        if (msk_size_x, msk_size_y) != (img_size_x, img_size_y):
            raise rsgispy.RSGISPyException(
                "Valid mask image dimensions do not match input images."
            )
        vld_msk_arr = get_img_data_as_arr(vld_msk_img, img_bands=[1])[0]
    else:
        base_vld = np.any(
            (base_arr != base_img_no_data) & np.isfinite(base_arr), axis=0
        )
        chng_vld = np.any(
            (chng_arr != chng_img_no_data) & np.isfinite(chng_arr), axis=0
        )
        vld_msk_arr = np.where(base_vld | chng_vld, int(vld_msk_val), 0)

    vld_flat = vld_msk_arr.reshape(n_pxls) == vld_msk_val
    pxl_ids = np.flatnonzero(vld_flat)
    if pxl_ids.size < min_valid_pxls:
        raise rsgispy.RSGISPyException(
            f"There are fewer than {min_valid_pxls} valid pixels ({pxl_ids.size})."
        )

    base_flat = base_arr.reshape(n_bands, n_pxls)[:, pxl_ids]
    chng_flat = chng_arr.reshape(n_bands, n_pxls)[:, pxl_ids]

    if apply_std and pxl_ids.size > 0:
        base_flat = np.nan_to_num(
            standardise_img_data(base_flat), copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        chng_flat = np.nan_to_num(
            standardise_img_data(chng_flat), copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )

    return n_bands, img_size_x, img_size_y, n_pxls, pxl_ids, base_flat, chng_flat


def _safe_otsu_threshold(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold with safe fallback if the array is constant or
    has zero interquartile range.
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    if float(np.max(finite)) == float(np.min(finite)):
        return float(finite[0])
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    if iqr <= 0.0:
        return float(0.5 * (np.min(finite) + np.max(finite)))
    return float(calc_otsu_threshold(finite))


def _get_binary_change_map(data: np.ndarray, max_iter: int = 1500) -> np.ndarray:
    """
    Apply 2-cluster K-means to segment change magnitude values into ``0``
    (unchanged) and ``1`` (changed).
    """
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, "change_vector_analysis (use_kmeans=True)", "scikit-learn", "stats"
    )
    vec = np.asarray(data, dtype=np.float64).reshape(-1, 1)
    km = KMeans(n_clusters=2, max_iter=max_iter, n_init="auto").fit(vec)
    c0, c1 = float(km.cluster_centers_[0, 0]), float(km.cluster_centers_[1, 0])
    chng_cluster = 0 if c0 > c1 else 1
    return (km.labels_ == chng_cluster).astype(np.uint8)


def image_difference(
    in_base_img: str,
    in_chng_img: str,
    out_diff_img: str,
    vld_msk_img: Optional[str] = None,
    vld_msk_val: int = 1,
    apply_std: bool = False,
    img_base_bands: Optional[Sequence[int]] = None,
    img_chng_bands: Optional[Sequence[int]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    base_img_no_data: float = 0,
    chng_img_no_data: float = 0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the per-band difference (``base - change``) between two co-registered
    images of matching dimensions and spatial extent.

    :param in_base_img: Path to the input base (reference) image file.
    :param in_chng_img: Path to the input change image file.
    :param out_diff_img: Path to the output difference image file.
    :param vld_msk_img: Optional path to a valid-pixel mask image (band 1 used).
    :param vld_msk_val: Pixel value in ``vld_msk_img`` indicating valid pixels
                        (Default = ``1``).
    :param apply_std: If ``True``, standardise each band prior to differencing
                      (Default = ``False``).
    :param img_base_bands: Optional sequence of 1-based bands for ``in_base_img``.
    :param img_chng_bands: Optional sequence of 1-based bands for ``in_chng_img``.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param base_img_no_data: No-data value in ``in_base_img`` (Default = ``0``).
    :param chng_img_no_data: No-data value in ``in_chng_img`` (Default = ``0``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.changedetect.imgchngmthds as icm

        icm.image_difference("t1.tif", "t2.tif", "diff.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    n_bands, img_x, img_y, n_pxls, pxl_ids, base_flat, chng_flat = (
        _prepare_change_inputs(
            in_base_img=in_base_img,
            in_chng_img=in_chng_img,
            vld_msk_img=vld_msk_img,
            vld_msk_val=vld_msk_val,
            img_base_bands=img_base_bands,
            img_chng_bands=img_chng_bands,
            base_img_no_data=base_img_no_data,
            chng_img_no_data=chng_img_no_data,
            apply_std=apply_std,
        )
    )

    fill_val = 0.0 if no_data_val is None else float(no_data_val)
    out_flat = np.full((n_bands, n_pxls), fill_val, dtype=np.float32)
    out_flat[:, pxl_ids] = (base_flat - chng_flat).astype(np.float32, copy=False)
    diff_img_arr = out_flat.reshape((n_bands, img_y, img_x))

    create_img_from_array_ref_img(
        data_arr=diff_img_arr,
        output_img=out_diff_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )
    if calc_stats:
        pop_img_stats(
            out_diff_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def image_ratio(
    in_base_img: str,
    in_chng_img: str,
    out_ratio_img: str,
    vld_msk_img: Optional[str] = None,
    vld_msk_val: int = 1,
    apply_std: bool = False,
    img_base_bands: Optional[Sequence[int]] = None,
    img_chng_bands: Optional[Sequence[int]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    base_img_no_data: float = 0,
    chng_img_no_data: float = 0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the per-band ratio (``base / change``) between two co-registered
    images of matching dimensions and spatial extent.

    :param in_base_img: Path to the input base (reference) image file.
    :param in_chng_img: Path to the input change image file.
    :param out_ratio_img: Path to the output ratio image file.
    :param vld_msk_img: Optional path to a valid-pixel mask image (band 1 used).
    :param vld_msk_val: Pixel value in ``vld_msk_img`` indicating valid pixels
                        (Default = ``1``).
    :param apply_std: If ``True``, standardise each band prior to ratioing
                      (Default = ``False``).
    :param img_base_bands: Optional sequence of 1-based bands for ``in_base_img``.
    :param img_chng_bands: Optional sequence of 1-based bands for ``in_chng_img``.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param base_img_no_data: No-data value in ``in_base_img`` (Default = ``0``).
    :param chng_img_no_data: No-data value in ``in_chng_img`` (Default = ``0``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.changedetect.imgchngmthds as icm

        icm.image_ratio("t1.tif", "t2.tif", "ratio.tif")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    n_bands, img_x, img_y, n_pxls, pxl_ids, base_flat, chng_flat = (
        _prepare_change_inputs(
            in_base_img=in_base_img,
            in_chng_img=in_chng_img,
            vld_msk_img=vld_msk_img,
            vld_msk_val=vld_msk_val,
            img_base_bands=img_base_bands,
            img_chng_bands=img_chng_bands,
            base_img_no_data=base_img_no_data,
            chng_img_no_data=chng_img_no_data,
            apply_std=apply_std,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_vals = base_flat / chng_flat
    np.nan_to_num(ratio_vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    fill_val = 0.0 if no_data_val is None else float(no_data_val)
    out_flat = np.full((n_bands, n_pxls), fill_val, dtype=np.float32)
    out_flat[:, pxl_ids] = ratio_vals.astype(np.float32, copy=False)
    ratio_img_arr = out_flat.reshape((n_bands, img_y, img_x))

    create_img_from_array_ref_img(
        data_arr=ratio_img_arr,
        output_img=out_ratio_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )
    if calc_stats:
        pop_img_stats(
            out_ratio_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def change_vector_analysis(
    in_base_img: str,
    in_chng_img: str,
    out_chng_img: str,
    out_chng_dist_img: str,
    vld_msk_img: Optional[str] = None,
    vld_msk_val: int = 1,
    apply_std: bool = False,
    img_base_bands: Optional[Sequence[int]] = None,
    img_chng_bands: Optional[Sequence[int]] = None,
    use_kmeans: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    base_img_no_data: float = 0,
    chng_img_no_data: float = 0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform Change Vector Analysis (CVA) between two co-registered images.

    Computes the Euclidean magnitude of the multi-band difference vector and
    thresholds change pixels using either Otsu's threshold (default) or 2-class
    K-means clustering (``use_kmeans=True``, requires ``scikit-learn``).

    :param in_base_img: Path to the input base image file.
    :param in_chng_img: Path to the input change image file.
    :param out_chng_img: Path to the output ``uint8`` binary change image file
                         (``1`` = change, ``0`` = no change).
    :param out_chng_dist_img: Path to the output threshold-centred change
                              distance image file.
    :param vld_msk_img: Optional path to a valid-pixel mask image (band 1 used).
    :param vld_msk_val: Pixel value in ``vld_msk_img`` indicating valid pixels
                        (Default = ``1``).
    :param apply_std: If ``True``, standardise each band prior to CVA
                      (Default = ``False``).
    :param img_base_bands: Optional sequence of 1-based bands for ``in_base_img``.
    :param img_chng_bands: Optional sequence of 1-based bands for ``in_chng_img``.
    :param use_kmeans: Use K-means clustering instead of Otsu thresholding for
                       the binary change mask (Default = ``False``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type for
                     ``out_chng_dist_img`` (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional output no-data value for ``out_chng_dist_img``
                        (Default = ``None``).
    :param base_img_no_data: No-data value in ``in_base_img`` (Default = ``0``).
    :param chng_img_no_data: No-data value in ``in_chng_img`` (Default = ``0``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.changedetect.imgchngmthds as icm

        icm.change_vector_analysis(
            "t1.tif", "t2.tif", "cva_bin.tif", "cva_dist.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    if use_kmeans:
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            "change_vector_analysis (use_kmeans=True)",
            "scikit-learn",
            "stats",
        )

    _, img_x, img_y, n_pxls, pxl_ids, base_flat, chng_flat = _prepare_change_inputs(
        in_base_img=in_base_img,
        in_chng_img=in_chng_img,
        vld_msk_img=vld_msk_img,
        vld_msk_val=vld_msk_val,
        img_base_bands=img_base_bands,
        img_chng_bands=img_chng_bands,
        base_img_no_data=base_img_no_data,
        chng_img_no_data=chng_img_no_data,
        apply_std=apply_std,
        min_valid_pxls=50,
    )

    img_diff = base_flat - chng_flat
    l2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    np.nan_to_num(l2_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    otsu_thres = _safe_otsu_threshold(l2_norm)

    fill_val = 0.0 if no_data_val is None else float(no_data_val)
    l2_flat_arr = np.full(n_pxls, fill_val, dtype=np.float32)
    l2_flat_arr[pxl_ids] = (l2_norm - otsu_thres).astype(np.float32, copy=False)
    l2_dist_img = l2_flat_arr.reshape((1, img_y, img_x))

    create_img_from_array_ref_img(
        data_arr=l2_dist_img,
        output_img=out_chng_dist_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )

    chng_bin_flat = np.zeros(n_pxls, dtype=np.uint8)
    if use_kmeans:
        chng_bin_flat[pxl_ids] = _get_binary_change_map(l2_norm)
    else:
        chng_bin_flat[pxl_ids] = (l2_norm > otsu_thres).astype(np.uint8)
    chng_bin_img = chng_bin_flat.reshape((1, img_y, img_x))

    create_img_from_array_ref_img(
        data_arr=chng_bin_img,
        output_img=out_chng_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_8UINT,
        no_data_val=255,
    )

    if calc_stats:
        pop_img_stats(
            out_chng_dist_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_img_stats(
            out_chng_img,
            no_data_val=255,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def slow_feature_analysis(
    in_base_img: str,
    in_chng_img: str,
    out_chng_img: str,
    out_chng_dist_img: str,
    vld_msk_img: Optional[str] = None,
    vld_msk_val: int = 1,
    apply_std: bool = False,
    img_base_bands: Optional[Sequence[int]] = None,
    img_chng_bands: Optional[Sequence[int]] = None,
    sfa_max_iter: int = 50,
    sfa_epsilon: float = 1e-6,
    sfa_norm_trans: bool = False,
    sfa_regular: bool = False,
    use_kmeans: bool = False,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    base_img_no_data: float = 0,
    chng_img_no_data: float = 0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform Iteratively Reweighted Slow Feature Analysis (ISFA) change
    detection between two co-registered multi-band images (Wu et al., 2014).
    Requires at least 3 bands and 50 valid pixels.

    :param in_base_img: Path to the input base image file.
    :param in_chng_img: Path to the input change image file.
    :param out_chng_img: Path to the output ``uint8`` binary change image.
    :param out_chng_dist_img: Path to the output threshold-centred chi-square
                              distance image.
    :param vld_msk_img: Optional path to a valid-pixel mask image (band 1 used).
    :param vld_msk_val: Pixel value in ``vld_msk_img`` indicating valid pixels
                        (Default = ``1``).
    :param apply_std: If ``True``, standardise each band prior to SFA
                      (Default = ``False``).
    :param img_base_bands: Optional sequence of 1-based bands for ``in_base_img``.
    :param img_chng_bands: Optional sequence of 1-based bands for ``in_chng_img``.
    :param sfa_max_iter: Maximum number of IRLS iterations (Default = ``50``).
    :param sfa_epsilon: Convergence threshold on ``sqrt(lambda)``
                        (Default = ``1e-6``).
    :param sfa_norm_trans: If ``True``, normalise the transformation matrix
                           (Default = ``False``).
    :param sfa_regular: If ``True``, regularise the generalized eigenvalue
                        matrix with a smoothness penalty (Default = ``False``).
    :param use_kmeans: Use K-means clustering instead of Otsu thresholding for
                       the binary change mask (Default = ``False``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type for
                     ``out_chng_dist_img`` (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional output no-data value for ``out_chng_dist_img``
                        (Default = ``None``).
    :param base_img_no_data: No-data value in ``in_base_img`` (Default = ``0``).
    :param chng_img_no_data: No-data value in ``in_chng_img`` (Default = ``0``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during
                     iterations (Default = ``True``).

    .. code:: python

        import rsgispy.changedetect.imgchngmthds as icm

        icm.slow_feature_analysis(
            "t1.tif", "t2.tif", "sfa_bin.tif", "sfa_dist.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    rsgispy.check_optional_dep(HAVE_SCIPY, "slow_feature_analysis", "scipy", "stats")
    if use_kmeans:
        rsgispy.check_optional_dep(
            HAVE_SKLEARN, "slow_feature_analysis", "scikit-learn", "stats"
        )
    out_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype

    n_bands, img_x, img_y, n_pxls, pxl_ids, base_flat, chng_flat = (
        _prepare_change_inputs(
            in_base_img=in_base_img,
            in_chng_img=in_chng_img,
            vld_msk_img=vld_msk_img,
            vld_msk_val=vld_msk_val,
            img_base_bands=img_base_bands,
            img_chng_bands=img_chng_bands,
            base_img_no_data=base_img_no_data,
            chng_img_no_data=chng_img_no_data,
            apply_std=apply_std,
            min_valid_pxls=50,
        )
    )
    if n_bands < 3:
        raise rsgispy.RSGISPyException("There must be at least 3 bands.")

    arr_l = np.zeros((n_bands - 2, n_bands), dtype=np.float64)
    for i in range(n_bands - 2):
        arr_l[i, i] = 1.0
        arr_l[i, i + 1] = -2.0
        arr_l[i, i + 2] = 1.0
    arr_omega = arr_l.T @ arr_l
    tr_omega = float(np.trace(arr_omega))

    n_valid = base_flat.shape[1]
    img_x_mat = base_flat
    img_y_mat = chng_flat
    lamb = 100.0 * np.ones((n_bands, 1), dtype=np.float64)
    weight = np.ones((1, n_valid), dtype=np.float64)
    t_stat = np.zeros(n_valid, dtype=np.float64)

    pbar = rsgispy.get_tqdm()(
        range(sfa_max_iter), desc="ISFA", disable=not use_pbar
    )
    for _iter in pbar:
        sum_w = max(float(np.sum(weight)), 1e-12)
        denom = (n_valid - 1) * sum_w / float(n_valid)
        denom = max(denom, 1e-12)

        mean_x = np.sum(weight * img_x_mat, axis=1, keepdims=True) / sum_w
        mean_y = np.sum(weight * img_y_mat, axis=1, keepdims=True) / sum_w
        center_x = img_x_mat - mean_x
        center_y = img_y_mat - mean_y

        var_x = np.sum(weight * np.square(center_x), axis=1, keepdims=True) / denom
        var_y = np.sum(weight * np.square(center_y), axis=1, keepdims=True) / denom
        std_x = np.sqrt(np.maximum(var_x, 1e-12))
        std_y = np.sqrt(np.maximum(var_y, 1e-12))

        norm_x = center_x / std_x
        norm_y = center_y / std_y
        diff_img = norm_x - norm_y
        mat_a = (weight * diff_img) @ diff_img.T / denom
        mat_b = ((weight * norm_x) @ norm_x.T + (weight * norm_y) @ norm_y.T) / (
            2.0 * denom
        )

        if sfa_regular and tr_omega > 0.0:
            penalty = float(np.trace(mat_b)) / tr_omega
            mat_b = mat_b + penalty * arr_omega

        mat_a = 0.5 * (mat_a + mat_a.T)
        mat_b = 0.5 * (mat_b + mat_b.T) + 1e-9 * np.eye(n_bands)

        eigenvalue, eigenvector = scipy.linalg.eigh(mat_a, mat_b)
        eigenvalue = np.maximum(eigenvalue.real, 1e-12)
        idx = eigenvalue.argsort()
        eigenvalue = eigenvalue[idx]
        eigenvector = eigenvector[:, idx].real

        aux = np.maximum(np.abs(eigenvector).max(axis=0, keepdims=True), 1e-12)
        trans_mat = eigenvector / aux

        if sfa_norm_trans:
            diag_vals = np.diag(trans_mat.T @ mat_b @ trans_mat)
            out_std = 1.0 / np.sqrt(np.maximum(diag_vals, 1e-12))
            trans_mat = out_std * trans_mat

        isfa_var = (trans_mat.T @ norm_x) - (trans_mat.T @ norm_y)
        eig_col = eigenvalue.reshape(n_bands, 1)
        var_isfa = np.maximum(
            np.diag(trans_mat.T @ mat_a @ trans_mat).reshape(n_bands, 1),
            1e-12,
        )
        t_stat = np.sum(
            np.square(isfa_var) / var_isfa,
            axis=0,
        )

        threshold = float(np.max(np.abs(np.sqrt(lamb) - np.sqrt(eig_col))))
        lamb = eig_col
        if threshold < sfa_epsilon:
            break
        weight = (1.0 - scipy.stats.chi2.cdf(t_stat, n_bands)).reshape(1, n_valid)

    sqrt_chi2 = np.sqrt(np.maximum(t_stat, 0.0))
    np.nan_to_num(sqrt_chi2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    otsu_thres = _safe_otsu_threshold(sqrt_chi2)

    fill_val = 0.0 if no_data_val is None else float(no_data_val)
    dist_flat = np.full(n_pxls, fill_val, dtype=np.float32)
    dist_flat[pxl_ids] = (sqrt_chi2 - otsu_thres).astype(np.float32, copy=False)
    create_img_from_array_ref_img(
        data_arr=dist_flat.reshape((1, img_y, img_x)),
        output_img=out_chng_dist_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )

    bin_flat = np.zeros(n_pxls, dtype=np.uint8)
    if use_kmeans:
        bin_flat[pxl_ids] = _get_binary_change_map(sqrt_chi2)
    else:
        bin_flat[pxl_ids] = (sqrt_chi2 > otsu_thres).astype(np.uint8)

    create_img_from_array_ref_img(
        data_arr=bin_flat.reshape((1, img_y, img_x)),
        output_img=out_chng_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_8UINT,
        no_data_val=255,
    )

    if calc_stats:
        pop_img_stats(
            out_chng_dist_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_img_stats(
            out_chng_img,
            no_data_val=255,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _covw(
    center_x: np.ndarray, center_y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Compute weighted joint covariance matrix for MAD."""
    n = w.shape[1]
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    sum_w = max(float(w.sum()), 1e-12)
    v = np.concatenate((center_x, center_y), axis=0) * sqrt_w
    return (v @ v.T) / sum_w * (float(n) / max(1.0, float(n - 1)))


def multivariate_alteration_detection(
    in_base_img: str,
    in_chng_img: str,
    out_chng_img: str,
    out_chng_dist_img: str,
    out_chng_mads_img: str,
    vld_msk_img: Optional[str] = None,
    vld_msk_val: int = 1,
    apply_std: bool = False,
    img_base_bands: Optional[Sequence[int]] = None,
    img_chng_bands: Optional[Sequence[int]] = None,
    mad_max_iter: int = 50,
    mad_epsilon: float = 1e-6,
    use_kmeans: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    base_img_no_data: float = 0,
    chng_img_no_data: float = 0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform Iteratively Reweighted Multivariate Alteration Detection
    (IR-MAD) between two co-registered multi-band images. Requires at least
    50 valid pixels.

    :param in_base_img: Path to the input base image file.
    :param in_chng_img: Path to the input change image file.
    :param out_chng_img: Path to the output ``uint8`` binary change image.
    :param out_chng_dist_img: Path to the output square-root chi-square
                              distance image.
    :param out_chng_mads_img: Path to the output MAD variates multi-band image.
    :param vld_msk_img: Optional path to a valid-pixel mask image (band 1 used).
    :param vld_msk_val: Pixel value in ``vld_msk_img`` indicating valid pixels
                        (Default = ``1``).
    :param apply_std: If ``True``, standardise each band prior to MAD
                      (Default = ``False``).
    :param img_base_bands: Optional sequence of 1-based bands for ``in_base_img``.
    :param img_chng_bands: Optional sequence of 1-based bands for ``in_chng_img``.
    :param mad_max_iter: Maximum number of IR-MAD iterations (Default = ``50``).
    :param mad_epsilon: Canonical correlation convergence threshold
                        (Default = ``1e-6``).
    :param use_kmeans: Use K-means clustering instead of Otsu thresholding for
                       the binary change mask (Default = ``True``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type for
                     ``out_chng_dist_img`` and ``out_chng_mads_img``
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional output no-data value for continuous outputs
                        (Default = ``None``).
    :param base_img_no_data: No-data value in ``in_base_img`` (Default = ``0``).
    :param chng_img_no_data: No-data value in ``in_chng_img`` (Default = ``0``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during
                     iterations (Default = ``True``).

    .. code:: python

        import rsgispy.changedetect.imgchngmthds as icm

        icm.multivariate_alteration_detection(
            "t1.tif", "t2.tif", "mad_bin.tif", "mad_dist.tif", "mad_vars.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements
    rsgispy.check_optional_dep(
        HAVE_SCIPY, "multivariate_alteration_detection", "scipy", "stats"
    )
    if use_kmeans:
        rsgispy.check_optional_dep(
            HAVE_SKLEARN,
            "multivariate_alteration_detection",
            "scikit-learn",
            "stats",
        )
    out_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype

    n_bands, img_x, img_y, n_pxls, pxl_ids, base_flat, chng_flat = (
        _prepare_change_inputs(
            in_base_img=in_base_img,
            in_chng_img=in_chng_img,
            vld_msk_img=vld_msk_img,
            vld_msk_val=vld_msk_val,
            img_base_bands=img_base_bands,
            img_chng_bands=img_chng_bands,
            base_img_no_data=base_img_no_data,
            chng_img_no_data=chng_img_no_data,
            apply_std=apply_std,
            min_valid_pxls=50,
        )
    )

    n_valid = base_flat.shape[1]
    img_x_mat = base_flat
    img_y_mat = chng_flat
    weight = np.ones((1, n_valid), dtype=np.float64)
    can_corr = 100.0 * np.ones(n_bands, dtype=np.float64)
    mad_variates = np.zeros((n_bands, n_valid), dtype=np.float64)
    chi_square_dis = np.zeros(n_valid, dtype=np.float64)

    pbar = rsgispy.get_tqdm()(
        range(mad_max_iter), desc="IR-MAD", disable=not use_pbar
    )
    for _ in pbar:
        sum_w = max(float(np.sum(weight)), 1e-12)
        mean_x = np.sum(weight * img_x_mat, axis=1, keepdims=True) / sum_w
        mean_y = np.sum(weight * img_y_mat, axis=1, keepdims=True) / sum_w
        center_x = img_x_mat - mean_x
        center_y = img_y_mat - mean_y

        cov_xy = _covw(center_x, center_y, weight)
        size = cov_xy.shape[0]
        sigma_11 = cov_xy[0:n_bands, 0:n_bands] + 1e-9 * np.eye(n_bands)
        sigma_22 = cov_xy[n_bands:size, n_bands:size] + 1e-9 * np.eye(n_bands)
        sigma_12 = cov_xy[0:n_bands, n_bands:size]
        sigma_21 = sigma_12.T

        inv_s11 = np.linalg.pinv(sigma_11)
        inv_s22 = np.linalg.pinv(sigma_22)
        target_mat = inv_s11 @ sigma_12 @ inv_s22 @ sigma_21
        np.nan_to_num(target_mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        eigenvalue, eigenvector_x = np.linalg.eig(target_mat)
        eigenvalue = np.sqrt(np.maximum(eigenvalue.real, 0.0))
        idx = eigenvalue.argsort()
        eigenvalue = eigenvalue[idx]
        eigenvector_x = eigenvector_x[:, idx].real
        eigenvector_y = inv_s22 @ sigma_21 @ eigenvector_x

        denom_x = np.diag(eigenvector_x.T @ sigma_11 @ eigenvector_x)
        denom_y = np.diag(eigenvector_y.T @ sigma_22 @ eigenvector_y)
        norm_x = 1.0 / np.sqrt(np.maximum(denom_x, 1e-12))
        norm_y = 1.0 / np.sqrt(np.maximum(denom_y, 1e-12))
        eigenvector_x = norm_x * eigenvector_x
        eigenvector_y = norm_y * eigenvector_y
        cross_corr = np.diag(eigenvector_x.T @ sigma_12 @ eigenvector_y)
        signs = np.where(cross_corr < 0.0, -1.0, 1.0)
        eigenvector_y = eigenvector_y * signs

        mad_variates = (eigenvector_x.T @ center_x) - (eigenvector_y.T @ center_y)
        mad_var = np.maximum(2.0 * (1.0 - np.clip(eigenvalue, -1.0, 0.999999)), 1e-12)
        chi_square_dis = np.sum(
            np.square(mad_variates) / mad_var.reshape(n_bands, 1),
            axis=0,
        )

        if float(np.max(np.abs(can_corr - eigenvalue))) < mad_epsilon:
            can_corr = eigenvalue
            break
        can_corr = eigenvalue
        weight = (1.0 - scipy.stats.chi2.cdf(chi_square_dis, n_bands)).reshape(
            1, n_valid
        )

    sqrt_chi2 = np.sqrt(np.maximum(chi_square_dis, 0.0))
    np.nan_to_num(sqrt_chi2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    fill_val = 0.0 if no_data_val is None else float(no_data_val)
    dist_flat = np.full(n_pxls, fill_val, dtype=np.float32)
    dist_flat[pxl_ids] = sqrt_chi2.astype(np.float32, copy=False)
    create_img_from_array_ref_img(
        data_arr=dist_flat.reshape((1, img_y, img_x)),
        output_img=out_chng_dist_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )

    mads_flat = np.full((n_bands, n_pxls), fill_val, dtype=np.float32)
    mads_flat[:, pxl_ids] = mad_variates.astype(np.float32, copy=False)
    create_img_from_array_ref_img(
        data_arr=mads_flat.reshape((n_bands, img_y, img_x)),
        output_img=out_chng_mads_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=out_dtype,
        no_data_val=no_data_val,
    )

    bin_flat = np.zeros(n_pxls, dtype=np.uint8)
    if use_kmeans:
        bin_flat[pxl_ids] = _get_binary_change_map(sqrt_chi2)
    else:
        otsu_thres = _safe_otsu_threshold(sqrt_chi2)
        bin_flat[pxl_ids] = (sqrt_chi2 > otsu_thres).astype(np.uint8)

    create_img_from_array_ref_img(
        data_arr=bin_flat.reshape((1, img_y, img_x)),
        output_img=out_chng_img,
        ref_img=in_base_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_8UINT,
        no_data_val=255,
    )

    if calc_stats:
        pop_img_stats(
            out_chng_dist_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_img_stats(
            out_chng_mads_img,
            no_data_val=no_data_val,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_img_stats(
            out_chng_img,
            no_data_val=255,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
