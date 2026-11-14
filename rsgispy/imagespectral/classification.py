"""
The imagespectral.classification module provides spectral similarity mapping
and classification functions using reference endmember spectra, including
Spectral Angle Mapper (SAM) and Spectral Correlation Mapper (SCM).
"""

from contextlib import ExitStack
from typing import List, Optional

import numpy as np
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imagespectral._helpers import open_spectral_inputs, valid_cube_mask
from rsgispy.imagespectral.endmembers import (
    EndmemberInput,
    prepare_endmembers_array,
)
from rsgispy.imageutils.metadata import set_band_names
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats

__all__: List[str] = [
    "spectral_angle_mapper",
    "spectral_correlation_mapper",
]


def _run_spectral_similarity_mapper(
    input_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    mode: str,
    out_class_img: Optional[str] = None,
    out_summary_img: Optional[str] = None,
    summary_band_name: str = "score",
    gdalformat: str = "GTIFF",
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    degrees: bool = False,
    score_threshold: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Shared windowed runner for SAM and SCM spectral similarity mapping."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(input_img, "Input image")
    em_names, n_em, n_em_bands, em_arr = prepare_endmembers_array(
        endmembers, gain=endmembers_gain, file_format=file_format
    )
    if n_em == 0:
        raise rsgispy.RSGISPyException(
            "Endmembers table must contain at least one class column."
        )

    is_scm = mode == "scm"
    if is_scm:
        if n_em_bands < 2:
            raise rsgispy.RSGISPyException(
                "Spectral Correlation Mapper requires at least 2 spectral bands."
            )
        if score_threshold is not None and (
            float(score_threshold) < -1.0 or float(score_threshold) > 1.0
        ):
            raise rsgispy.RSGISPyException(
                f"min_corr must be within [-1.0, 1.0], got {score_threshold}."
            )
        em_work = em_arr - np.mean(em_arr, axis=1, keepdims=True)
        em_norms = np.linalg.norm(em_work, axis=1, keepdims=True)
        if np.any(em_norms == 0.0) or not np.all(np.isfinite(em_norms)):
            raise rsgispy.RSGISPyException(
                "All endmember spectra must be finite and have non-zero "
                "variance across bands."
            )
    else:
        em_work = em_arr
        em_norms = np.linalg.norm(em_work, axis=1, keepdims=True)
        if np.any(em_norms == 0.0) or not np.all(np.isfinite(em_norms)):
            raise rsgispy.RSGISPyException(
                "All endmember spectra must be finite and have non-zero norm."
            )
    em_unit = em_work / em_norms  # (n_em, n_bands)

    out_float_nd = -9999.0
    cls_dtype = "uint8" if n_em <= 255 else "uint16"
    cls_np_dtype = np.uint8 if n_em <= 255 else np.uint16

    with ExitStack() as stack:
        src, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        if int(src.count) != n_em_bands:
            raise rsgispy.RSGISPyException(
                f"Input image band count ({src.count}) does not match "
                f"endmember band count ({n_em_bands})."
            )

        eff_nd = no_data_val if no_data_val is not None else src.nodata

        dst_scores = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=n_em,
        )
        dst_scores.nodata = out_float_nd

        dst_summary = None
        if out_summary_img is not None:
            dst_summary = open_output_dataset(
                stack=stack,
                output_img=out_summary_img,
                gdalformat=gdalformat,
                out_rio_dtype="float32",
                grid=grid,
                count=1,
            )
            dst_summary.nodata = out_float_nd

        dst_cls = None
        if out_class_img is not None:
            dst_cls = open_output_dataset(
                stack=stack,
                output_img=out_class_img,
                gdalformat=gdalformat,
                out_rio_dtype=cls_dtype,
                grid=grid,
                count=1,
            )
            dst_cls.nodata = 0

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            n_pxls = win_h * win_w
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = src.read(window=img_win, out_dtype=np.float64)
            msk_blk = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_blk = msk_ds.read(1, window=msk_win)
            vld = valid_cube_mask(blk, eff_nd, msk_blk, valid_msk_val)

            vld_flat = vld.ravel()
            scores_out = np.full(
                (n_pxls, n_em), out_float_nd, dtype=np.float32
            )
            summary_out = np.full(n_pxls, out_float_nd, dtype=np.float32)
            cls_out = np.zeros(n_pxls, dtype=cls_np_dtype)

            if np.any(vld_flat):
                pxls = np.moveaxis(blk, 0, -1).reshape(-1, n_em_bands)[
                    vld_flat
                ]
                if img_gain > 1:
                    pxls = pxls / float(img_gain)
                if is_scm:
                    pxls_work = pxls - np.mean(pxls, axis=1, keepdims=True)
                else:
                    pxls_work = pxls
                norms = np.linalg.norm(pxls_work, axis=1)
                non_zero = norms > 0.0
                if np.any(non_zero):
                    nz_unit = pxls_work[non_zero] / norms[non_zero, None]
                    cos_sim = np.clip(nz_unit @ em_unit.T, -1.0, 1.0)
                    if is_scm:
                        metric_vals = cos_sim
                        best_idx = np.argmax(metric_vals, axis=1)
                    else:
                        metric_vals = np.arccos(cos_sim)
                        if degrees:
                            metric_vals = np.degrees(metric_vals)
                        best_idx = np.argmin(metric_vals, axis=1)

                    nz_indices = np.flatnonzero(vld_flat)[non_zero]
                    scores_out[nz_indices] = metric_vals.astype(np.float32)
                    best_score = metric_vals[
                        np.arange(metric_vals.shape[0]), best_idx
                    ]
                    summary_out[nz_indices] = best_score.astype(np.float32)

                    cls_vals = (best_idx + 1).astype(cls_np_dtype)
                    if score_threshold is not None:
                        keep = (
                            best_score >= float(score_threshold)
                            if is_scm
                            else best_score <= float(score_threshold)
                        )
                        cls_vals = np.where(keep, cls_vals, 0).astype(
                            cls_np_dtype
                        )
                    cls_out[nz_indices] = cls_vals

            out_cube = np.moveaxis(
                scores_out.reshape((win_h, win_w, n_em)), -1, 0
            )
            dst_scores.write(out_cube, window=dst_win)
            if dst_summary is not None:
                dst_summary.write(
                    summary_out.reshape((1, win_h, win_w)), window=dst_win
                )
            if dst_cls is not None:
                dst_cls.write(
                    cls_out.reshape((1, win_h, win_w)), window=dst_win
                )

    set_band_names(output_img, em_names, feedback=False)
    if out_summary_img is not None:
        set_band_names(out_summary_img, [summary_band_name], feedback=False)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_float_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        if out_summary_img is not None:
            pop_img_stats(
                out_summary_img,
                no_data_val=out_float_nd,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )
        if out_class_img is not None:
            pop_thmt_img_stats(
                out_class_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=True,
                use_pbar=use_pbar,
            )


def spectral_angle_mapper(
    input_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    out_class_img: Optional[str] = None,
    out_min_angle_img: Optional[str] = None,
    gdalformat: str = "GTIFF",
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    degrees: bool = False,
    max_angle: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute Spectral Angle Mapper (SAM; Kruse et al., 1993) spectral angles
    between every pixel spectrum in ``input_img`` and reference endmembers,
    processed window-by-window without loading full images into memory.

    For a pixel spectrum :math:`x` and endmember spectrum :math:`e`, the
    spectral angle is
    :math:`\\theta = \\arccos(\\langle x, e \\rangle / (\\|x\\|\\|e\\|))`,
    providing illumination- and gain-invariant similarity scoring in
    ``[0, pi]`` radians (or ``[0, 180]`` degrees when ``degrees=True``).

    :param input_img: Path to the input multi-band reflectance/radiance image.
    :param endmembers: :class:`pandas.DataFrame` (bands as rows, endmember class
                       names as columns) or path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param output_img: Path to the output ``float32`` multi-band spectral angle
                       raster (one band per endmember class).
    :param out_class_img: Optional output thematic classification raster path
                          assigning each pixel the 1-based index of the
                          smallest spectral angle (``0`` for nodata/zero-norm
                          or when exceeding ``max_angle``).
    :param out_min_angle_img: Optional single-band ``float32`` output raster
                              recording the minimum spectral angle per pixel.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param img_gain: Gain divisor applied to ``input_img`` when ``> 1``
                     (Default = ``1``).
    :param endmembers_gain: Gain divisor applied to ``endmembers`` when ``> 1``
                            (Default = ``1``).
    :param file_format: Optional table format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param degrees: If ``True``, output angles (and interpret ``max_angle``) in
                    degrees rather than radians (Default = ``False``).
    :param max_angle: Optional maximum spectral angle threshold for
                      ``out_class_img``; pixels whose minimum angle exceeds
                      ``max_angle`` are assigned class ``0`` (unclassified).
    :param valid_msk_img: Optional single-band valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute statistics and pyramids on output
                       rasters (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.classification as sc

        sc.spectral_angle_mapper(
            "refl.tif",
            "endmembers.csv",
            "sam_angles.tif",
            out_class_img="sam_class.tif",
            out_min_angle_img="sam_min_angle.tif",
            degrees=True,
            max_angle=15.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_spectral_similarity_mapper(
        input_img=input_img,
        endmembers=endmembers,
        output_img=output_img,
        mode="sam",
        out_class_img=out_class_img,
        out_summary_img=out_min_angle_img,
        summary_band_name="min_angle",
        gdalformat=gdalformat,
        img_gain=img_gain,
        endmembers_gain=endmembers_gain,
        file_format=file_format,
        degrees=degrees,
        score_threshold=max_angle,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def spectral_correlation_mapper(
    input_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    out_class_img: Optional[str] = None,
    out_max_corr_img: Optional[str] = None,
    gdalformat: str = "GTIFF",
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    min_corr: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute Spectral Correlation Mapper (SCM; de Carvalho & Meneses, 2000)
    Pearson correlation scores between every pixel spectrum in ``input_img``
    and reference endmembers, processed window-by-window.

    Unlike :func:`spectral_angle_mapper`, SCM centres each spectrum by its
    band mean before computing the cosine similarity:
    :math:`r = \\langle x - \\bar{x}, e - \\bar{e} \\rangle / `
    :math:`(\\|x - \\bar{x}\\|\\|e - \\bar{e}\\|) \\in [-1, 1]`.
    This makes SCM invariant to both positive multiplicative scaling and
    additive illumination/offset shifts across bands (higher correlation values
    indicate closer spectral match).

    :param input_img: Path to the input multi-band reflectance/radiance image
                      (requires at least 2 spectral bands).
    :param endmembers: :class:`pandas.DataFrame` (bands as rows, endmember class
                       names as columns) or path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param output_img: Path to the output ``float32`` multi-band SCM correlation
                       raster (values in ``[-1, 1]``, one band per endmember
                       class).
    :param out_class_img: Optional output thematic classification raster path
                          assigning each pixel the 1-based index of the
                          highest SCM correlation (``0`` for nodata/zero-variance
                          spectra or when below ``min_corr``).
    :param out_max_corr_img: Optional single-band ``float32`` output raster
                             recording the maximum SCM correlation per pixel.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param img_gain: Gain divisor applied to ``input_img`` when ``> 1``
                     (Default = ``1``).
    :param endmembers_gain: Gain divisor applied to ``endmembers`` when ``> 1``
                            (Default = ``1``).
    :param file_format: Optional table format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param min_corr: Optional minimum correlation threshold in ``[-1.0, 1.0]``
                     for ``out_class_img``; pixels whose maximum correlation is
                     strictly below ``min_corr`` are assigned class ``0``
                     (unclassified).
    :param valid_msk_img: Optional single-band valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute statistics and pyramids on output
                       rasters (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.classification as sc

        sc.spectral_correlation_mapper(
            "refl.tif",
            "endmembers.csv",
            "scm_corr.tif",
            out_class_img="scm_class.tif",
            out_max_corr_img="scm_max_corr.tif",
            min_corr=0.85,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_spectral_similarity_mapper(
        input_img=input_img,
        endmembers=endmembers,
        output_img=output_img,
        mode="scm",
        out_class_img=out_class_img,
        out_summary_img=out_max_corr_img,
        summary_band_name="max_corr",
        gdalformat=gdalformat,
        img_gain=img_gain,
        endmembers_gain=endmembers_gain,
        file_format=file_format,
        degrees=False,
        score_threshold=min_corr,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
