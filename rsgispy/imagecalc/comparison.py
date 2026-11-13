"""
The imagecalc.comparison module provides image difference, equality checks,
correlation, mutual information, and Earth Mover's Distance utilities using
``rasterio``.

Functions requiring ``scipy`` and/or ``scikit-learn`` are only defined and
exported when those optional dependencies are installed.
"""

from contextlib import ExitStack
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
    validate_band_index,
)

try:
    import scipy.stats  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import sklearn.metrics  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

__all__ = [
    "calc_img_difference",
    "are_imgs_equal",
    "are_img_bands_equal",
]

if HAVE_SCIPY:
    __all__.extend(
        [
            "calc_img_correlation",
            "calc_img_earth_move_dist",
        ]
    )

if HAVE_SKLEARN:
    __all__.append("calc_img_mutual_info")


def _extract_valid_paired_pixels(
    in_a_img: str,
    in_b_img: str,
    img_a_band: int,
    img_b_band: int,
    img_a_no_data: Optional[float],
    img_b_no_data: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract flattened finite, non-nodata paired pixel arrays over the spatial
    intersection of ``in_a_img`` and ``in_b_img``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_a_img, in_b_img])
        ds_a = datasets[in_a_img]
        ds_b = datasets[in_b_img]
        validate_band_index(img_a_band, ds_a.count, in_a_img)
        validate_band_index(img_b_band, ds_b.count, in_b_img)

        nd_a = (
            img_a_no_data
            if img_a_no_data is not None
            else ds_a.nodatavals[img_a_band - 1]
        )
        if nd_a is None:
            raise rsgispy.RSGISPyException(
                "A no data value is not available for image A - please specify."
            )

        nd_b = (
            img_b_no_data
            if img_b_no_data is not None
            else ds_b.nodatavals[img_b_band - 1]
        )
        if nd_b is None:
            raise rsgispy.RSGISPyException(
                "A no data value is not available for image B - please specify."
            )

        grid = compute_spatial_intersection(datasets)
        base_ca, base_ra = grid.offsets[in_a_img]
        base_cb, base_rb = grid.offsets[in_b_img]

        win_a = Window(base_ca, base_ra, grid.width, grid.height)
        win_b = Window(base_cb, base_rb, grid.width, grid.height)

        arr_a = ds_a.read(img_a_band, window=win_a, out_dtype=np.float64).ravel()
        arr_b = ds_b.read(img_b_band, window=win_b, out_dtype=np.float64).ravel()

    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if not np.isnan(nd_a):
        valid &= arr_a != nd_a
    if not np.isnan(nd_b):
        valid &= arr_b != nd_b

    return arr_a[valid], arr_b[valid]


def calc_img_difference(
    in_a_img: str,
    in_b_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate the pixel-wise band difference between two images
    (``in_a_img - in_b_img``) over their spatial intersection.

    :param in_a_img: Path to the first input image file.
    :param in_b_img: Path to the second input image file.
    :param output_img: Path to the output difference image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.comparison

        rsgispy.imagecalc.comparison.calc_img_difference(
            in_a_img="img_a.tif",
            in_b_img="img_b.tif",
            output_img="diff.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_a_img, in_b_img])
        ds_a = datasets[in_a_img]
        ds_b = datasets[in_b_img]
        if ds_a.count != ds_b.count:
            raise rsgispy.RSGISPyException(
                "Input images must have the same number of bands "
                f"({ds_a.count} vs {ds_b.count})."
            )

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=ds_a.count,
            creation_opts=creation_opts,
        )

        base_ca, base_ra = grid.offsets[in_a_img]
        base_cb, base_rb = grid.offsets[in_b_img]
        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            win_a = Window(
                base_ca + int(out_win.col_off),
                base_ra + int(out_win.row_off),
                win_w,
                win_h,
            )
            win_b = Window(
                base_cb + int(out_win.col_off),
                base_rb + int(out_win.row_off),
                win_w,
                win_h,
            )
            blk_a = ds_a.read(window=win_a, out_dtype=np.float64)
            blk_b = ds_b.read(window=win_b, out_dtype=np.float64)
            diff = blk_a - blk_b
            out_bands = [
                cast_result_block(diff[b_idx], win_h, win_w, out_np_dtype)
                for b_idx in range(ds_a.count)
            ]
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def _calc_imgs_equal(
    in_ref_img: str,
    in_cmp_img: str,
    img_ref_band: Optional[int] = None,
    img_cmp_band: Optional[int] = None,
    prop_eql: float = 1.0,
    flt_dif: float = 0.0001,
    use_pbar: bool = True,
) -> Tuple[bool, float]:
    """
    Shared implementation for :func:`are_imgs_equal` (when band arguments are
    ``None``) and :func:`are_img_bands_equal` (when both band indices are given).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_ref_img, label="Reference image")
    ensure_file_exists(in_cmp_img, label="Comparison image")

    n_eq_pxls = 0
    n_pxls = 0
    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_ref_img, in_cmp_img])
        ds_ref = datasets[in_ref_img]
        ds_cmp = datasets[in_cmp_img]
        if img_ref_band is None and img_cmp_band is None:
            if ds_ref.count != ds_cmp.count:
                raise rsgispy.RSGISPyException(
                    "The number of image bands is not the same between the two images."
                )
        else:
            if img_ref_band is not None:
                validate_band_index(img_ref_band, ds_ref.count, in_ref_img)
            if img_cmp_band is not None:
                validate_band_index(img_cmp_band, ds_cmp.count, in_cmp_img)

        grid = compute_spatial_intersection(datasets)
        base_cref, base_rref = grid.offsets[in_ref_img]
        base_ccmp, base_rcmp = grid.offsets[in_cmp_img]

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            win_ref = Window(
                base_cref + int(out_win.col_off),
                base_rref + int(out_win.row_off),
                win_w,
                win_h,
            )
            win_cmp = Window(
                base_ccmp + int(out_win.col_off),
                base_rcmp + int(out_win.row_off),
                win_w,
                win_h,
            )
            ref_arr = (
                ds_ref.read(window=win_ref, out_dtype=np.float64)
                if img_ref_band is None
                else ds_ref.read(img_ref_band, window=win_ref, out_dtype=np.float64)
            ).ravel()
            cmp_arr = (
                ds_cmp.read(window=win_cmp, out_dtype=np.float64)
                if img_cmp_band is None
                else ds_cmp.read(img_cmp_band, window=win_cmp, out_dtype=np.float64)
            ).ravel()

            valid = ~(np.isnan(ref_arr) | np.isnan(cmp_arr))
            ref_v = ref_arr[valid]
            cmp_v = cmp_arr[valid]
            if ref_v.size > 0:
                n_eq_pxls += int(np.count_nonzero(np.abs(ref_v - cmp_v) < flt_dif))
                n_pxls += int(ref_v.size)

    if n_pxls == 0:
        return False, 0.0
    prop_pxls_eq = float(n_eq_pxls) / float(n_pxls)
    return bool(prop_pxls_eq >= prop_eql), prop_pxls_eq


def are_imgs_equal(
    in_ref_img: str,
    in_cmp_img: str,
    prop_eql: float = 1.0,
    flt_dif: float = 0.0001,
    use_pbar: bool = True,
) -> Tuple[bool, float]:
    """
    Check whether two multi-band images have equal pixel values across their
    spatial intersection within floating-point tolerance ``flt_dif``.

    :param in_ref_img: Path to the reference image.
    :param in_cmp_img: Path to the comparison image.
    :param prop_eql: Minimum proportion of matching pixels required to return
                     ``True`` (range ``0`` to ``1``, Default = ``1.0``).
    :param flt_dif: Absolute difference tolerance below which two pixels are
                    considered equal (Default = ``0.0001``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: Tuple ``(is_equal, prop_pixels_equal)``.

    .. code:: python

        import rsgispy.imagecalc.comparison

        is_eq, prop = rsgispy.imagecalc.comparison.are_imgs_equal(
            in_ref_img="img_ref.tif",
            in_cmp_img="img_cmp.tif",
            prop_eql=0.99,
            flt_dif=0.0001,
        )
    """
    return _calc_imgs_equal(
        in_ref_img=in_ref_img,
        in_cmp_img=in_cmp_img,
        prop_eql=prop_eql,
        flt_dif=flt_dif,
        use_pbar=use_pbar,
    )


def are_img_bands_equal(
    in_ref_img: str,
    img_ref_band: int,
    in_cmp_img: str,
    img_cmp_band: int,
    prop_eql: float = 1.0,
    flt_dif: float = 0.0001,
    use_pbar: bool = True,
) -> Tuple[bool, float]:
    """
    Check whether two specific image bands have equal pixel values across their
    spatial intersection within floating-point tolerance ``flt_dif``.

    :param in_ref_img: Path to the reference image.
    :param img_ref_band: 1-based band index in ``in_ref_img``.
    :param in_cmp_img: Path to the comparison image.
    :param img_cmp_band: 1-based band index in ``in_cmp_img``.
    :param prop_eql: Minimum proportion of matching pixels required to return
                     ``True`` (range ``0`` to ``1``, Default = ``1.0``).
    :param flt_dif: Absolute difference tolerance below which two pixels are
                    considered equal (Default = ``0.0001``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: Tuple ``(is_equal, prop_pixels_equal)``.

    .. code:: python

        import rsgispy.imagecalc.comparison

        is_eq, prop = rsgispy.imagecalc.comparison.are_img_bands_equal(
            in_ref_img="img_ref.tif",
            img_ref_band=1,
            in_cmp_img="img_cmp.tif",
            img_cmp_band=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _calc_imgs_equal(
        in_ref_img=in_ref_img,
        in_cmp_img=in_cmp_img,
        img_ref_band=img_ref_band,
        img_cmp_band=img_cmp_band,
        prop_eql=prop_eql,
        flt_dif=flt_dif,
        use_pbar=use_pbar,
    )


if HAVE_SCIPY:

    def calc_img_correlation(
        in_a_img: str,
        in_b_img: str,
        img_a_band: int,
        img_b_band: int,
        img_a_no_data: Optional[float] = None,
        img_b_no_data: Optional[float] = None,
        corr_stat_method: int = rsgispy.STATS_CORR_PEARSONS,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate the correlation coefficient and p-value between two image
        bands over their spatial intersection using SciPy.

        :param in_a_img: Path to input image A.
        :param in_b_img: Path to input image B.
        :param img_a_band: 1-based band index within ``in_a_img``.
        :param img_b_band: 1-based band index within ``in_b_img``.
        :param img_a_no_data: No-data value in image A (``None`` reads header).
        :param img_b_no_data: No-data value in image B (``None`` reads header).
        :param corr_stat_method: Correlation method ``rsgispy.STATS_CORR_*``
                                 (Default = ``rsgispy.STATS_CORR_PEARSONS``).
        :return: Tuple ``(correlation_coefficient, p_value)``, or ``(None, None)``
                 if 5 or fewer valid paired pixels exist.

        .. code:: python

            import rsgispy
            import rsgispy.imagecalc.comparison

            r_val, p_val = rsgispy.imagecalc.comparison.calc_img_correlation(
                in_a_img="img_a.tif",
                in_b_img="img_b.tif",
                img_a_band=1,
                img_b_band=1,
                corr_stat_method=rsgispy.STATS_CORR_PEARSONS,
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        arr_a, arr_b = _extract_valid_paired_pixels(
            in_a_img,
            in_b_img,
            img_a_band,
            img_b_band,
            img_a_no_data,
            img_b_no_data,
        )

        if arr_a.size <= 5 or arr_b.size <= 5:
            return None, None

        if corr_stat_method == rsgispy.STATS_CORR_PEARSONS:
            coeff, p_val = scipy.stats.pearsonr(arr_a, arr_b)
        elif corr_stat_method == rsgispy.STATS_CORR_SPEARMAN:
            coeff, p_val = scipy.stats.spearmanr(arr_a, arr_b)
        elif corr_stat_method == rsgispy.STATS_CORR_KENDALL_TAU:
            coeff, p_val = scipy.stats.kendalltau(arr_a, arr_b)
        elif corr_stat_method == rsgispy.STATS_CORR_POINT_BISERIAL:
            coeff, p_val = scipy.stats.pointbiserialr(arr_a, arr_b)
        else:
            raise rsgispy.RSGISPyException(
                "Do not recognise the correlation method specified."
            )
        return float(coeff), float(p_val)

    def calc_img_earth_move_dist(
        in_a_img: str,
        in_b_img: str,
        img_a_band: int,
        img_b_band: int,
        img_a_no_data: Optional[float] = None,
        img_b_no_data: Optional[float] = None,
        hist_a_bins: Optional[int] = None,
        hist_b_bins: Optional[int] = None,
        use_glb_range: Optional[bool] = None,
    ) -> Optional[float]:
        """
        Calculate the Earth Mover's (Wasserstein) distance between the value
        distributions of two image bands using SciPy.

        :param in_a_img: Path to input image A.
        :param in_b_img: Path to input image B.
        :param img_a_band: 1-based band index within ``in_a_img``.
        :param img_b_band: 1-based band index within ``in_b_img``.
        :param img_a_no_data: No-data value in image A (``None`` reads header).
        :param img_b_no_data: No-data value in image B (``None`` reads header).
        :param hist_a_bins: Number of histogram bins for image A (Default = ``10``).
        :param hist_b_bins: Number of histogram bins for image B (Default = ``10``).
        :param use_glb_range: If ``True``, share a common ``[min, max]`` range
                              across both histograms.
        :return: Earth Mover's distance, or ``None`` if <= 5 valid pixels exist.

        .. code:: python

            import rsgispy.imagecalc.comparison

            emd = rsgispy.imagecalc.comparison.calc_img_earth_move_dist(
                in_a_img="img_a.tif",
                in_b_img="img_b.tif",
                img_a_band=1,
                img_b_band=1,
                hist_a_bins=20,
                hist_b_bins=20,
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        arr_a, arr_b = _extract_valid_paired_pixels(
            in_a_img,
            in_b_img,
            img_a_band,
            img_b_band,
            img_a_no_data,
            img_b_no_data,
        )

        if arr_a.size <= 5 or arr_b.size <= 5:
            return None

        img_range: Optional[Tuple[float, float]] = None
        if use_glb_range:
            img_min = float(min(np.min(arr_a), np.min(arr_b)))
            img_max = float(max(np.max(arr_a), np.max(arr_b)))
            img_range = (img_min, img_max)

        bins_a = 10 if hist_a_bins is None else int(hist_a_bins)
        bins_b = 10 if hist_b_bins is None else int(hist_b_bins)
        hist_a, _ = np.histogram(arr_a, bins=bins_a, range=img_range)
        hist_b, _ = np.histogram(arr_b, bins=bins_b, range=img_range)

        hist_a_prob = hist_a / float(arr_a.size)
        hist_b_prob = hist_b / float(arr_b.size)
        em_val = scipy.stats.wasserstein_distance(hist_a_prob, hist_b_prob)
        return float(em_val)


if HAVE_SKLEARN:

    def calc_img_mutual_info(
        in_a_img: str,
        in_b_img: str,
        img_a_band: int,
        img_b_band: int,
        img_a_no_data: Optional[float] = None,
        img_b_no_data: Optional[float] = None,
        hist_bins: int = 100,
        img_a_min: Optional[float] = None,
        img_a_max: Optional[float] = None,
        img_b_min: Optional[float] = None,
        img_b_max: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate the mutual information score between two image bands over
        their spatial intersection using scikit-learn.

        :param in_a_img: Path to input image A.
        :param in_b_img: Path to input image B.
        :param img_a_band: 1-based band index within ``in_a_img``.
        :param img_b_band: 1-based band index within ``in_b_img``.
        :param img_a_no_data: No-data value in image A (``None`` reads header).
        :param img_b_no_data: No-data value in image B (``None`` reads header).
        :param hist_bins: Number of 2D histogram bins (Default = ``100``).
        :param img_a_min: Optional exclusive minimum threshold for image A.
        :param img_a_max: Optional exclusive maximum threshold for image A.
        :param img_b_min: Optional exclusive minimum threshold for image B.
        :param img_b_max: Optional exclusive maximum threshold for image B.
        :return: Mutual information score, or ``None`` if <= 5 valid pixels exist.

        .. code:: python

            import rsgispy.imagecalc.comparison

            mi_val = rsgispy.imagecalc.comparison.calc_img_mutual_info(
                in_a_img="img_a.tif",
                in_b_img="img_b.tif",
                img_a_band=1,
                img_b_band=1,
                hist_bins=64,
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        arr_a, arr_b = _extract_valid_paired_pixels(
            in_a_img,
            in_b_img,
            img_a_band,
            img_b_band,
            img_a_no_data,
            img_b_no_data,
        )

        if (
            img_a_min is not None
            and img_a_max is not None
            and img_b_min is not None
            and img_b_max is not None
        ):
            msk_a = (arr_a > img_a_min) & (arr_a < img_a_max)
            msk_b = (arr_b > img_b_min) & (arr_b < img_b_max)
            msk = msk_a & msk_b
            arr_a = arr_a[msk]
            arr_b = arr_b[msk]

        if arr_a.size <= 5 or arr_b.size <= 5:
            return None

        c_xy = np.histogram2d(arr_a, arr_b, hist_bins)[0]
        mi_val = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)
        return float(mi_val)
