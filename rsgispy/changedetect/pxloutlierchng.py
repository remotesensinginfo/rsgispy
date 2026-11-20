"""
The changedetect.pxloutlierchng module provides per-class pixel outlier
change detection algorithms using ``rasterio``, ``numpy``, and
:mod:`rsgispy.tools.thresholding`:

* :func:`find_class_pyod_outliers` — general outlier detector wrapper (PyOD API)
* :func:`find_class_otsu_outliers` — Otsu automatic thresholding
* :func:`find_class_li_outliers` — Li's iterative minimum cross-entropy thresholding
* :func:`find_class_kurt_skew_outliers` — skewness and kurtosis tail
  optimisation (requires ``scipy``)
"""

from contextlib import ExitStack
import math
from typing import Any, List, Optional, Sequence

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageutils.colourtable import define_colour_table
from rsgispy.imageutils.metadata import get_img_band_count, get_img_no_data_value
from rsgispy.imageutils.sampling import extract_img_pxl_vals_in_msk
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools.plotting import plot_histogram_threshold
from rsgispy.tools.thresholding import (
    HAVE_SCIPY,
    calc_kurt_skew_threshold,
    calc_li_threshold,
    calc_otsu_threshold,
)

__all__: List[str] = [
    "find_class_pyod_outliers",
    "find_class_otsu_outliers",
    "find_class_li_outliers",
    "find_class_kurt_skew_outliers",
]

_OUTLIER_CLR_LUT = {
    0: [0, 0, 0, 0],
    1: [0, 255, 0, 255],
    2: [255, 0, 0, 255],
}


def _apply_threshold_mask_raster(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    chng_thres: float,
    low_thres: bool,
    img_mask_val: int,
    img_band: int,
    img_val_no_data: Optional[float],
    gdalformat: str,
    datatype: Optional[int],
    calc_stats: bool,
    use_pbar: bool,
) -> None:
    """
    Classify pixels inside ``in_msk_img == img_mask_val`` as ``1`` (non-outlier)
    or ``2`` (outlier) relative to ``chng_thres`` in windowed blocks and
    optionally populate thematic metadata and colour table.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(out_rio_dtype)
    nd_is_nan = img_val_no_data is not None and math.isnan(float(img_val_no_data))

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img, in_msk_img])
        img_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        validate_band_index(img_band, img_ds.count, input_img)

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )

        img_xoff, img_yoff = grid.offsets[paths[0]]
        msk_xoff, msk_yoff = grid.offsets[paths[1]]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            col_off, row_off = int(win.col_off), int(win.row_off)
            win_w, win_h = int(win.width), int(win.height)
            img_win = Window(img_xoff + col_off, img_yoff + row_off, win_w, win_h)
            msk_win = Window(msk_xoff + col_off, msk_yoff + row_off, win_w, win_h)

            val_arr = img_ds.read(img_band, window=img_win)
            msk_arr = msk_ds.read(1, window=msk_win)

            out_arr = np.zeros(val_arr.shape, dtype=out_np_dtype)
            in_msk = msk_arr == img_mask_val
            if img_val_no_data is not None:
                if nd_is_nan:
                    in_msk &= ~np.isnan(val_arr)
                else:
                    in_msk &= val_arr != img_val_no_data

            if np.issubdtype(val_arr.dtype, np.floating):
                in_msk &= np.isfinite(val_arr)

            if low_thres:
                outlier_msk = in_msk & (val_arr < chng_thres)
            else:
                outlier_msk = in_msk & (val_arr > chng_thres)

            out_arr[in_msk] = 1
            out_arr[outlier_msk] = 2
            dst.write(out_arr, 1, window=win)

    if calc_stats:
        if out_rio_dtype in ("uint8", "uint16"):
            pop_thmt_img_stats(
                output_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=True,
                use_pbar=use_pbar,
            )
            define_colour_table(output_img, _OUTLIER_CLR_LUT)
        else:
            pop_img_stats(
                output_img,
                no_data_val=0,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def find_class_pyod_outliers(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    pyod_obj: Any,
    out_scores_img: Optional[str] = None,
    img_mask_val: int = 1,
    img_bands: Optional[Sequence[int]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Detect outliers within a masked class using a PyOD outlier detector object
    (or any estimator implementing ``fit``, ``predict``, and ``predict_proba``).

    Output class values are ``0`` (outside mask / no-data), ``1`` (within mask,
    not an outlier), and ``2`` (within mask, outlier).

    :param input_img: Path to the input image file.
    :param in_msk_img: Path to the mask image defining the class of interest.
    :param output_img: Path to the output classification image.
    :param pyod_obj: Initialised outlier detector instance implementing
                     ``fit(X)``, ``predict(X)``, and (when ``out_scores_img``
                     is given) ``predict_proba(X, method="unify")``.
    :param out_scores_img: Optional path to a ``float32`` outlier probability
                           image.
    :param img_mask_val: Pixel value in ``in_msk_img`` defining the target class
                         (Default = ``1``).
    :param img_bands: Optional sequence of 1-based band indices to analyse
                      (all bands used when ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type for
                     ``output_img`` (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data value for ``input_img`` (inferred
                        from image metadata if ``None``).
    :param calc_stats: If ``True``, compute output image statistics, pyramids,
                       and colour table (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        from pyod.models.iforest import IForest
        import rsgispy.changedetect.pxloutlierchng as poc

        poc.find_class_pyod_outliers(
            "img.tif", "mask.tif", "outliers.tif", IForest()
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    ensure_file_exists(input_img, "Input image")
    ensure_file_exists(in_msk_img, "Mask image")

    if img_bands is not None:
        if not isinstance(img_bands, (list, tuple, np.ndarray)):
            raise rsgispy.RSGISPyException(
                "If provided then img_bands should be a sequence (or None)."
            )
        sel_bands = [int(b) for b in img_bands]
    else:
        n_bands = get_img_band_count(input_img)
        sel_bands = list(range(1, n_bands + 1))

    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
    out_np_dtype = np.dtype(out_rio_dtype)

    img_val_no_data = (
        get_img_no_data_value(input_img) if no_data_val is None else no_data_val
    )
    msk_arr_vals = extract_img_pxl_vals_in_msk(
        input_img,
        sel_bands,
        in_msk_img,
        img_mask_val,
        img_val_no_data,
        use_pbar=use_pbar,
    )
    if msk_arr_vals.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels were found within the specified mask value."
        )

    pyod_obj.fit(msk_arr_vals)
    nd_is_nan = img_val_no_data is not None and math.isnan(float(img_val_no_data))

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img, in_msk_img])
        img_ds = datasets[paths[0]]
        msk_ds = datasets[paths[1]]
        for bidx in sel_bands:
            validate_band_index(bidx, img_ds.count, input_img)

        grid = compute_spatial_intersection(datasets)
        dst_lbls = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        dst_scores = (
            open_output_dataset(
                stack=stack,
                output_img=out_scores_img,
                gdalformat=gdalformat,
                out_rio_dtype="float32",
                grid=grid,
                count=1,
            )
            if out_scores_img is not None
            else None
        )

        img_xoff, img_yoff = grid.offsets[paths[0]]
        msk_xoff, msk_yoff = grid.offsets[paths[1]]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            col_off, row_off = int(win.col_off), int(win.row_off)
            win_w, win_h = int(win.width), int(win.height)
            n_pxls = win_w * win_h

            img_win = Window(img_xoff + col_off, img_yoff + row_off, win_w, win_h)
            msk_win = Window(msk_xoff + col_off, msk_yoff + row_off, win_w, win_h)

            msk_flat = msk_ds.read(1, window=msk_win).ravel()
            out_lbls = np.zeros(n_pxls, dtype=out_np_dtype)
            out_scores = (
                np.zeros(n_pxls, dtype=np.float32) if dst_scores is not None else None
            )

            if np.any(msk_flat == img_mask_val):
                img_blk = img_ds.read(sel_bands, window=img_win)
                feat_mat = np.moveaxis(img_blk, 0, -1).reshape(n_pxls, len(sel_bands))

                vld = msk_flat == img_mask_val
                if img_val_no_data is not None:
                    if nd_is_nan:
                        vld &= ~np.isnan(feat_mat).any(axis=1)
                    else:
                        vld &= (feat_mat != img_val_no_data).all(axis=1)
                if np.issubdtype(feat_mat.dtype, np.floating):
                    vld &= np.all(np.isfinite(feat_mat), axis=1)

                if np.any(vld):
                    vld_feats = feat_mat[vld]
                    pred_lbls = np.asarray(pyod_obj.predict(vld_feats)) + 1
                    out_lbls[vld] = pred_lbls.astype(out_np_dtype, copy=False)
                    if out_scores is not None:
                        try:
                            pred_probs = pyod_obj.predict_proba(
                                vld_feats, method="unify"
                            )
                        except TypeError:
                            pred_probs = pyod_obj.predict_proba(vld_feats)
                        out_scores[vld] = np.asarray(
                            pred_probs[:, 1], dtype=np.float32
                        )

            dst_lbls.write(out_lbls.reshape(win_h, win_w), 1, window=win)
            if dst_scores is not None and out_scores is not None:
                dst_scores.write(out_scores.reshape(win_h, win_w), 1, window=win)

    if calc_stats:
        if out_rio_dtype in ("uint8", "uint16"):
            pop_thmt_img_stats(
                output_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=True,
                use_pbar=use_pbar,
            )
            define_colour_table(output_img, _OUTLIER_CLR_LUT)
        else:
            pop_img_stats(
                output_img,
                no_data_val=0,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )

        if out_scores_img is not None:
            pop_img_stats(
                out_scores_img,
                no_data_val=0,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def find_class_otsu_outliers(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    low_thres: bool,
    img_mask_val: int = 1,
    img_band: int = 1,
    plot_thres_file: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> float:
    """
    Find outliers within a masked class using Otsu automatic thresholding.

    Output pixel values are ``0`` (outside mask / no-data), ``1`` (within mask,
    not outlier), and ``2`` (within mask, outlier).

    :param input_img: Path to the input raster image file.
    :param in_msk_img: Path to the mask image defining the class of interest.
    :param output_img: Path to the output thematic image file.
    :param low_thres: If ``True``, values below the threshold are outliers; if
                      ``False``, values above the threshold are outliers.
    :param img_mask_val: Pixel value in ``in_msk_img`` defining the target class
                         (Default = ``1``).
    :param img_band: 1-based band index within ``input_img`` (Default = ``1``).
    :param plot_thres_file: Optional output image path for a histogram plot
                            showing the computed threshold.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data value for ``input_img`` (inferred
                        from image metadata if ``None``).
    :param calc_stats: If ``True``, compute output image statistics, pyramids,
                       and colour table (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: Computed threshold value.

    .. code:: python

        import rsgispy.changedetect.pxloutlierchng as poc

        thres = poc.find_class_otsu_outliers(
            "img.tif", "mask.tif", "outliers_otsu.tif", low_thres=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img, "Input image")
    ensure_file_exists(in_msk_img, "Mask image")

    img_val_no_data = (
        get_img_no_data_value(input_img, img_band)
        if no_data_val is None
        else no_data_val
    )

    msk_arr_vals = extract_img_pxl_vals_in_msk(
        input_img,
        [img_band],
        in_msk_img,
        img_mask_val,
        img_val_no_data,
        use_pbar=use_pbar,
    )
    if msk_arr_vals.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels were found within the specified mask value."
        )

    vals_1d = msk_arr_vals[:, 0]
    chng_thres = float(calc_otsu_threshold(vals_1d))

    _apply_threshold_mask_raster(
        input_img=input_img,
        in_msk_img=in_msk_img,
        output_img=output_img,
        chng_thres=chng_thres,
        low_thres=low_thres,
        img_mask_val=img_mask_val,
        img_band=img_band,
        img_val_no_data=img_val_no_data,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    if plot_thres_file is not None:
        plot_histogram_threshold(vals_1d, plot_thres_file, threshold=chng_thres)

    return chng_thres


def find_class_li_outliers(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    low_thres: bool,
    tolerance: Optional[float] = None,
    init_thres: Optional[float] = None,
    img_mask_val: int = 1,
    img_band: int = 1,
    plot_thres_file: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> float:
    """
    Find outliers within a masked class using Li's iterative Minimum Cross
    Entropy thresholding.

    Output pixel values are ``0`` (outside mask / no-data), ``1`` (within mask,
    not outlier), and ``2`` (within mask, outlier).

    :param input_img: Path to the input raster image file.
    :param in_msk_img: Path to the mask image defining the class of interest.
    :param output_img: Path to the output thematic image file.
    :param low_thres: If ``True``, values below the threshold are outliers; if
                      ``False``, values above the threshold are outliers.
    :param tolerance: Optional convergence tolerance for Li's iteration.
    :param init_thres: Optional initial threshold estimate.
    :param img_mask_val: Pixel value in ``in_msk_img`` defining the target class
                         (Default = ``1``).
    :param img_band: 1-based band index within ``input_img`` (Default = ``1``).
    :param plot_thres_file: Optional output image path for a histogram plot
                            showing the computed threshold.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data value for ``input_img`` (inferred
                        from image metadata if ``None``).
    :param calc_stats: If ``True``, compute output image statistics, pyramids,
                       and colour table (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: Computed threshold value.

    .. code:: python

        import rsgispy.changedetect.pxloutlierchng as poc

        thres = poc.find_class_li_outliers(
            "img.tif", "mask.tif", "outliers_li.tif", low_thres=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(input_img, "Input image")
    ensure_file_exists(in_msk_img, "Mask image")

    img_val_no_data = (
        get_img_no_data_value(input_img, img_band)
        if no_data_val is None
        else no_data_val
    )

    msk_arr_vals = extract_img_pxl_vals_in_msk(
        input_img,
        [img_band],
        in_msk_img,
        img_mask_val,
        img_val_no_data,
        use_pbar=use_pbar,
    )
    if msk_arr_vals.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels were found within the specified mask value."
        )

    vals_1d = msk_arr_vals[:, 0]
    chng_thres = float(calc_li_threshold(vals_1d, tolerance, init_thres))

    _apply_threshold_mask_raster(
        input_img=input_img,
        in_msk_img=in_msk_img,
        output_img=output_img,
        chng_thres=chng_thres,
        low_thres=low_thres,
        img_mask_val=img_mask_val,
        img_band=img_band,
        img_val_no_data=img_val_no_data,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    if plot_thres_file is not None:
        plot_histogram_threshold(vals_1d, plot_thres_file, threshold=chng_thres)

    return chng_thres


def find_class_kurt_skew_outliers(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    vld_min: float,
    vld_max: float,
    init_thres: float,
    low_thres: bool,
    contamination: float = 10.0,
    only_kurtosis: bool = False,
    img_mask_val: int = 1,
    img_band: int = 1,
    plot_thres_file: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> float:
    """
    Find outliers within a masked class by optimising histogram skewness
    and kurtosis (requires ``scipy``).

    Output pixel values are ``0`` (outside mask / no-data), ``1`` (within mask,
    not outlier), and ``2`` (within mask, outlier).

    :param input_img: Path to the input raster image file.
    :param in_msk_img: Path to the mask image defining the class of interest.
    :param output_img: Path to the output thematic image file.
    :param vld_min: Minimum threshold bound for valid values.
    :param vld_max: Maximum threshold bound for valid values.
    :param init_thres: Initial estimate of the change threshold.
    :param low_thres: If ``True``, values below the threshold are outliers; if
                      ``False``, values above the threshold are outliers.
    :param contamination: Estimated outlier percentage in ``[1, 100]``
                          (Default = ``10.0``).
    :param only_kurtosis: If ``True``, optimise kurtosis alone instead of
                          ``abs(kurtosis) + abs(skew)`` (Default = ``False``).
    :param img_mask_val: Pixel value in ``in_msk_img`` defining the target class
                         (Default = ``1``).
    :param img_band: 1-based band index within ``input_img`` (Default = ``1``).
    :param plot_thres_file: Optional output image path for a histogram plot
                            showing the computed threshold.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data value for ``input_img`` (inferred
                        from image metadata if ``None``).
    :param calc_stats: If ``True``, compute output image statistics, pyramids,
                       and colour table (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: Computed threshold value.

    .. code:: python

        import rsgispy.changedetect.pxloutlierchng as poc

        thres = poc.find_class_kurt_skew_outliers(
            "img.tif",
            "mask.tif",
            "outliers_ks.tif",
            vld_min=0.0,
            vld_max=0.8,
            init_thres=0.3,
            low_thres=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rsgispy.check_optional_dep(
        HAVE_SCIPY, "find_class_kurt_skew_outliers", "scipy", "stats"
    )
    ensure_file_exists(input_img, "Input image")
    ensure_file_exists(in_msk_img, "Mask image")

    img_val_no_data = (
        get_img_no_data_value(input_img, img_band)
        if no_data_val is None
        else no_data_val
    )

    msk_arr_vals = extract_img_pxl_vals_in_msk(
        input_img,
        [img_band],
        in_msk_img,
        img_mask_val,
        img_val_no_data,
        use_pbar=use_pbar,
    )
    if msk_arr_vals.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels were found within the specified mask value."
        )

    vals_1d = msk_arr_vals[:, 0]
    chng_thres = float(
        calc_kurt_skew_threshold(
            vals_1d,
            vld_max,
            vld_min,
            init_thres,
            low_thres,
            contamination,
            only_kurtosis,
        )
    )

    _apply_threshold_mask_raster(
        input_img=input_img,
        in_msk_img=in_msk_img,
        output_img=output_img,
        chng_thres=chng_thres,
        low_thres=low_thres,
        img_mask_val=img_mask_val,
        img_band=img_band,
        img_val_no_data=img_val_no_data,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    if plot_thres_file is not None:
        plot_histogram_threshold(vals_1d, plot_thres_file, threshold=chng_thres)

    return chng_thres
