# pylint: disable=too-many-lines
"""
The imagespectral.transforms module provides spectral feature and radiometric
transformations including ZCA noise whitening, windowed/incremental Principal
Component Analysis (PCA), Minimum Noise Fraction (MNF), Tasseled Cap
transformations, and streaming histogram matching.
"""

from contextlib import ExitStack
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.imagespectral._helpers import (
    open_spectral_inputs,
    resolve_eff_nodata,
    resolve_selected_bands,
    valid_cube_mask,
)
from rsgispy.imageutils.blank import create_copy_img
from rsgispy.imageutils.metadata import get_img_no_data_value, set_band_names
from rsgispy.imageutils.stats import pop_img_stats

__all__: List[str] = [
    "whiten_image",
    "calc_pca",
    "calc_mnf",
    "get_tasseled_cap_sensors",
    "calc_tasseled_cap",
    "histogram_matching",
]

_TASSELED_CAP_PRESETS: Dict[str, Dict[str, Any]] = {
    "landsat5_tm": {
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Crist (1985), Remote Sensing of Environment 17(3), 301-306 (reflectance)
        "coeffs": np.array(
            [
                [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
                [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
                [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109],
            ],
            dtype=np.float64,
        ),
    },
    "landsat7_etm": {
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Huang et al. (2002), Int. J. Remote Sens. 23(8), 1741-1748 (TOA reflectance)
        "coeffs": np.array(
            [
                [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596],
                [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630],
                [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388],
            ],
            dtype=np.float64,
        ),
    },
    "landsat8_oli": {
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Baig et al. (2014), Remote Sensing Letters 5(5), 423-431 (TOA reflectance)
        "coeffs": np.array(
            [
                [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
            ],
            dtype=np.float64,
        ),
    },
    "landsat9_oli": {
        "bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Baig et al. (2014), Remote Sensing Letters 5(5), 423-431 (OLI-2 compatible)
        "coeffs": np.array(
            [
                [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
            ],
            dtype=np.float64,
        ),
    },
    "sentinel2_msi": {
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Shi & Xu (2019), IEEE JSTARS 12(10), 4038-4048 (TOA reflectance)
        "coeffs": np.array(
            [
                [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949],
                [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856],
                [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308],
            ],
            dtype=np.float64,
        ),
    },
    "modis_nbar": {
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "components": ["Brightness", "Greenness", "Wetness"],
        # Lobser & Cohen (2007), Int. J. Remote Sens. 28(22), 5079-5098 (NBAR)
        "coeffs": np.array(
            [
                [0.4395, 0.5945, 0.2460, 0.3918, 0.3506, 0.2136, 0.2678],
                [-0.4064, 0.5129, -0.2744, -0.2893, 0.4882, -0.0036, -0.4169],
                [0.1147, 0.2489, 0.2408, 0.3132, -0.3122, -0.6416, -0.5087],
            ],
            dtype=np.float64,
        ),
    },
}


def whiten_image(
    input_img: str,
    valid_msk_img: str,
    valid_msk_val: int,
    output_img: str,
    gdalformat: str = "GTIFF",
    use_pbar: bool = True,
) -> None:
    """
    Apply ZCA noise whitening to ``input_img`` over pixels matching
    ``valid_msk_val`` in ``valid_msk_img`` so valid spectra have unit variance
    and zero band-to-band correlation.

    :param input_img: Path to the input multi-band image file.
    :param valid_msk_img: Path to the single-band valid-region mask image file.
    :param valid_msk_val: Mask value identifying valid pixels.
    :param output_img: Path to the output float32 whitened image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        st.whiten_image(
            "input_img.tif", "valid_msk.tif", 1, "whitened_img.tif", "GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    ensure_file_exists(valid_msk_img, label="Valid mask image")

    with rasterio.open(valid_msk_img, "r") as msk_src:
        if msk_src.count != 1:
            raise rsgispy.RSGISPyException(
                "Valid mask only expected to have a single band."
            )
        x_msk_size, y_msk_size = msk_src.width, msk_src.height
        vld_flat = msk_src.read(1).ravel()

    with rasterio.open(input_img, "r") as img_src:
        n_bands = img_src.count
        x_size, y_size = img_src.width, img_src.height
        if x_msk_size != x_size or y_msk_size != y_size:
            raise rsgispy.RSGISPyException(
                "Mask and input image dimensions do not match."
            )
        img_arr = img_src.read(out_dtype=np.float32)

    valid_mask = vld_flat == valid_msk_val
    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        raise rsgispy.RSGISPyException(
            "At least 2 valid pixels are required to compute the covariance matrix."
        )

    img_flat = np.moveaxis(img_arr, 0, 2).reshape(-1, n_bands)
    valid_data = img_flat[valid_mask].astype(np.float64)
    centred = valid_data - valid_data.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / (n_valid - 1)
    _, s_vals, vh_mat = np.linalg.svd(cov)
    safe_s = np.where(s_vals > 1e-12, s_vals, 1.0)
    inv_sqrt = np.where(s_vals > 1e-12, safe_s**-0.5, 0.0)
    whiten_mat = vh_mat.T @ np.diag(inv_sqrt) @ vh_mat
    whitened = (centred @ whiten_mat).astype(np.float32)

    create_copy_img(
        input_img,
        output_img,
        n_bands,
        0,
        gdalformat,
        rsgispy.TYPE_32FLOAT,
        use_pbar=use_pbar,
    )
    out_flat = np.zeros((vld_flat.shape[0], n_bands), dtype=np.float32)
    out_flat[valid_mask] = whitened
    out_cube = np.moveaxis(
        out_flat.reshape((y_size, x_size, n_bands)), 2, 0
    )
    with rasterio.open(output_img, "r+") as dst:
        dst.write(out_cube)


def _update_streaming_moments(
    count_a: int,
    mean_a: Optional[np.ndarray],
    m2_a: Optional[np.ndarray],
    batch: np.ndarray,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Update exact running mean and centred scatter matrix ``M2`` (Chan et al.)."""
    n_b = int(batch.shape[0])
    if n_b == 0:
        return count_a, mean_a, m2_a
    mean_b = np.mean(batch, axis=0)
    centred_b = batch - mean_b
    m2_b = centred_b.T @ centred_b
    if count_a == 0 or mean_a is None or m2_a is None:
        return n_b, mean_b, m2_b
    delta = mean_b - mean_a
    n_total = count_a + n_b
    mean_new = mean_a + delta * (float(n_b) / float(n_total))
    m2_new = (
        m2_a
        + m2_b
        + np.outer(delta, delta) * (float(count_a) * float(n_b) / float(n_total))
    )
    return n_total, mean_new, m2_new


def _canonicalise_eigenvector_signs(vecs: np.ndarray) -> np.ndarray:
    """Orient each column vector so its largest-magnitude element is positive."""
    out = vecs.copy()
    for col in range(out.shape[1]):
        max_idx = int(np.argmax(np.abs(out[:, col])))
        if out[max_idx, col] < 0.0:
            out[:, col] *= -1.0
    return out


def _stream_linear_projection(
    input_img: str,
    output_img: str,
    band_indices: Sequence[int],
    proj_matrix: np.ndarray,
    mean_vec: Optional[np.ndarray],
    scale_vec: Optional[np.ndarray],
    out_band_names: Sequence[str],
    eff_in_nd: Optional[float],
    eff_out_nd: float,
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    gdalformat: str,
    datatype: int,
    calc_stats: bool,
    use_pbar: bool,
) -> None:
    """Apply ``((X - mean) / scale) @ proj_matrix`` across raster windows."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    n_in_bands = len(band_indices)
    n_out_bands = int(proj_matrix.shape[1])

    with ExitStack() as stack:
        img_ds, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_out_bands,
            output_exists=False,
        )
        dst.nodata = eff_out_nd

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = img_ds.read(
                band_indices, window=img_win, out_dtype=np.float64
            )
            msk_blk: Optional[np.ndarray] = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_blk = msk_ds.read(1, window=msk_win)

            valid = valid_cube_mask(blk, eff_in_nd, msk_blk, valid_msk_val)
            valid_flat = valid.ravel()
            out_flat = np.full(
                (win_h * win_w, n_out_bands), eff_out_nd, dtype=np.float64
            )
            if np.any(valid_flat):
                pxls = np.moveaxis(blk, 0, 2).reshape(-1, n_in_bands)[
                    valid_flat
                ]
                if mean_vec is not None:
                    pxls = pxls - mean_vec
                if scale_vec is not None:
                    pxls = pxls / scale_vec
                out_flat[valid_flat] = pxls @ proj_matrix

            out_blk = np.moveaxis(
                out_flat.reshape((win_h, win_w, n_out_bands)), 2, 0
            )
            dst.write(
                cast_result_block(out_blk, win_h, win_w, out_np_dtype),
                window=dst_win,
            )

    set_band_names(output_img, list(out_band_names), feedback=False)
    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=eff_out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def calc_pca(
    input_img: str,
    output_img: str,
    n_components: Optional[int] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    bands: Optional[Sequence[int]] = None,
    standardise: bool = False,
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    out_stats_json: Optional[str] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Perform two-pass windowed/incremental Principal Component Analysis (PCA)
    on a multi-band raster image without loading the full scene into memory.

    Pass 1 accumulates exact per-band means and sample covariance across raster
    windows via Chan's parallel batch moment update over finite, non-nodata
    pixels; Pass 2 projects pixels onto the top ``n_components`` eigenvectors
    sorted in descending order of explained variance.

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output principal components raster file.
    :param n_components: Number of principal components to retain (defaults to
                         all selected input bands when ``None``).
    :param valid_msk_img: Optional path to a valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param bands: Optional sequence of 1-based band indices from ``input_img``
                  to include in the PCA (defaults to all bands when ``None``).
    :param standardise: If ``True``, decompose the correlation matrix
                        (zero mean and unit variance per band) rather than the
                        unstandardised covariance matrix (Default = ``False``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param out_no_data_val: Output nodata value (defaults to ``no_data_val`` if
                            set, else ``-9999.0`` for float outputs or ``0``
                            for integer outputs).
    :param out_stats_json: Optional JSON file path to save eigenvalues,
                           explained variance ratios, means, scales, and
                           component loadings.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output raster
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: Dictionary containing ``n_valid_pixels``, ``n_components``,
             ``standardised``, ``mean``, ``scale``, ``eigenvalues``,
             ``explained_variance_ratio``, ``cumulative_variance_ratio``, and
             ``loadings``.

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        stats = st.calc_pca(
            "sentinel2.tif",
            "pca_3bands.tif",
            n_components=3,
            standardise=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    eff_in_nd, eff_out_nd = resolve_eff_nodata(
        input_img, no_data_val, out_no_data_val, out_np_dtype
    )

    n_valid = 0
    mean_vec: Optional[np.ndarray] = None
    m2_mat: Optional[np.ndarray] = None

    with ExitStack() as stack:
        img_ds, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        band_indices = resolve_selected_bands(img_ds, bands, input_img)
        n_sel = len(band_indices)
        n_comp = n_sel if n_components is None else int(n_components)
        if n_comp < 1 or n_comp > n_sel:
            raise rsgispy.RSGISPyException(
                f"n_components ({n_comp}) must be between 1 and {n_sel}."
            )

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = img_ds.read(
                band_indices, window=img_win, out_dtype=np.float64
            )
            msk_blk: Optional[np.ndarray] = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_blk = msk_ds.read(1, window=msk_win)

            valid = valid_cube_mask(blk, eff_in_nd, msk_blk, valid_msk_val)
            valid_flat = valid.ravel()
            if np.any(valid_flat):
                batch = np.moveaxis(blk, 0, 2).reshape(-1, n_sel)[valid_flat]
                n_valid, mean_vec, m2_mat = _update_streaming_moments(
                    n_valid, mean_vec, m2_mat, batch
                )

    if n_valid < 2 or mean_vec is None or m2_mat is None:
        raise rsgispy.RSGISPyException(
            "At least 2 valid pixels are required to compute PCA."
        )

    cov_mat = 0.5 * ((m2_mat + m2_mat.T) / float(n_valid - 1))
    if standardise:
        raw_std = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))
        scale_vec = np.where(raw_std > 1e-12, raw_std, 1.0)
        decomp_mat = cov_mat / np.outer(scale_vec, scale_vec)
        for idx in range(n_sel):
            if raw_std[idx] <= 1e-12:
                decomp_mat[idx, :] = 0.0
                decomp_mat[:, idx] = 0.0
    else:
        scale_vec = np.ones(n_sel, dtype=np.float64)
        decomp_mat = cov_mat

    eigvals, eigvecs = np.linalg.eigh(0.5 * (decomp_mat + decomp_mat.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = _canonicalise_eigenvector_signs(eigvecs[:, order])

    proj_mat = eigvecs[:, :n_comp]
    sel_eigvals = eigvals[:n_comp]
    total_var = float(np.sum(eigvals))
    if total_var > 0.0:
        ev_ratio = sel_eigvals / total_var
    else:
        ev_ratio = np.zeros(n_comp, dtype=np.float64)
    cum_ratio = np.cumsum(ev_ratio)

    band_names = [f"PC_{idx + 1}" for idx in range(n_comp)]
    _stream_linear_projection(
        input_img=input_img,
        output_img=output_img,
        band_indices=band_indices,
        proj_matrix=proj_mat,
        mean_vec=mean_vec,
        scale_vec=scale_vec if standardise else None,
        out_band_names=band_names,
        eff_in_nd=eff_in_nd,
        eff_out_nd=eff_out_nd,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    stats_info: Dict[str, Any] = {
        "n_valid_pixels": int(n_valid),
        "n_components": int(n_comp),
        "standardised": bool(standardise),
        "mean": mean_vec.tolist(),
        "scale": scale_vec.tolist(),
        "eigenvalues": sel_eigvals.tolist(),
        "explained_variance_ratio": ev_ratio.tolist(),
        "cumulative_variance_ratio": cum_ratio.tolist(),
        "loadings": proj_mat.tolist(),
    }
    if out_stats_json is not None:
        out_dir = os.path.dirname(os.path.abspath(out_stats_json))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_stats_json, "w", encoding="utf-8") as handle:
            json.dump(stats_info, handle, indent=2)
    return stats_info


def calc_mnf(
    input_img: str,
    output_img: str,
    n_components: Optional[int] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    bands: Optional[Sequence[int]] = None,
    noise_direction: str = "both",
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    out_stats_json: Optional[str] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Compute the Minimum Noise Fraction (MNF) forward transformation
    (Green et al., 1988) in two streaming raster passes.

    Pass 1 estimates the data covariance matrix and the spatial noise
    covariance matrix using nearest-neighbour pixel shift differences over
    valid adjacent pixels (including exact window boundaries); Pass 2 applies
    noise whitening followed by eigenvector rotation ordered from highest to
    lowest signal-to-noise ratio (``MNF_1``, ..., ``MNF_k``).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output MNF component raster file.
    :param n_components: Number of MNF components to retain (defaults to all
                         selected input bands when ``None``).
    :param valid_msk_img: Optional path to a valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param bands: Optional sequence of 1-based band indices from ``input_img``
                  to include (defaults to all bands when ``None``).
    :param noise_direction: Spatial difference direction for noise covariance
                            estimation: ``'both'`` (horizontal and vertical),
                            ``'horizontal'``, or ``'vertical'``
                            (Default = ``'both'``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param out_no_data_val: Output nodata value (defaults to ``no_data_val`` if
                            set, else ``-9999.0`` for float outputs or ``0``
                            for integer outputs).
    :param out_stats_json: Optional JSON file path to save MNF eigenvalues,
                           estimated component SNRs, means, and transform matrix.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output raster
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: Dictionary containing ``n_valid_pixels``, ``n_noise_pairs``,
             ``n_components``, ``noise_direction``, ``mean``,
             ``mnf_eigenvalues``, ``snr_estimates``, and ``transform_matrix``.

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        mnf_info = st.calc_mnf(
            "hyperspectral.tif",
            "mnf_reduced.tif",
            n_components=10,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches
    dir_key = noise_direction.strip().lower()
    if dir_key not in ("both", "horizontal", "vertical"):
        raise rsgispy.RSGISPyException(
            f"Unsupported noise_direction '{noise_direction}'. Expected "
            "'both', 'horizontal', or 'vertical'."
        )

    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    eff_in_nd, eff_out_nd = resolve_eff_nodata(
        input_img, no_data_val, out_no_data_val, out_np_dtype
    )

    n_valid = 0
    mean_vec: Optional[np.ndarray] = None
    m2_mat: Optional[np.ndarray] = None
    n_noise_pairs = 0
    noise_ss: Optional[np.ndarray] = None

    with ExitStack() as stack:
        img_ds, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        band_indices = resolve_selected_bands(img_ds, bands, input_img)
        n_sel = len(band_indices)
        n_comp = n_sel if n_components is None else int(n_components)
        if n_comp < 1 or n_comp > n_sel:
            raise rsgispy.RSGISPyException(
                f"n_components ({n_comp}) must be between 1 and {n_sel}."
            )
        noise_ss = np.zeros((n_sel, n_sel), dtype=np.float64)

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            # Extend by up to 1 pixel right/down inside the grid to capture exact
            # window-boundary differences without artificial border padding.
            ext_w = min(win_w + 1, grid.width - col_off)
            ext_h = min(win_h + 1, grid.height - row_off)
            ext_img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, ext_w, ext_h
            )
            ext_blk = img_ds.read(
                band_indices, window=ext_img_win, out_dtype=np.float64
            )
            ext_msk: Optional[np.ndarray] = None
            if msk_ds is not None and msk_off is not None:
                ext_msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, ext_w, ext_h
                )
                ext_msk = msk_ds.read(1, window=ext_msk_win)

            ext_valid = valid_cube_mask(
                ext_blk, eff_in_nd, ext_msk, valid_msk_val
            )
            core_blk = ext_blk[:, :win_h, :win_w]
            core_valid_flat = ext_valid[:win_h, :win_w].ravel()
            if np.any(core_valid_flat):
                batch = np.moveaxis(core_blk, 0, 2).reshape(-1, n_sel)[
                    core_valid_flat
                ]
                n_valid, mean_vec, m2_mat = _update_streaming_moments(
                    n_valid, mean_vec, m2_mat, batch
                )

            if dir_key in ("both", "horizontal") and ext_w > 1:
                h_pairs = (
                    ext_valid[:win_h, 1:ext_w] & ext_valid[:win_h, : ext_w - 1]
                ).ravel()
                if np.any(h_pairs):
                    dh = (
                        ext_blk[:, :win_h, 1:ext_w]
                        - ext_blk[:, :win_h, : ext_w - 1]
                    )
                    dh_flat = np.moveaxis(dh, 0, 2).reshape(-1, n_sel)[h_pairs]
                    noise_ss += 0.5 * (dh_flat.T @ dh_flat)
                    n_noise_pairs += int(dh_flat.shape[0])

            if dir_key in ("both", "vertical") and ext_h > 1:
                v_pairs = (
                    ext_valid[1:ext_h, :win_w] & ext_valid[: ext_h - 1, :win_w]
                ).ravel()
                if np.any(v_pairs):
                    dv = (
                        ext_blk[:, 1:ext_h, :win_w]
                        - ext_blk[:, : ext_h - 1, :win_w]
                    )
                    dv_flat = np.moveaxis(dv, 0, 2).reshape(-1, n_sel)[v_pairs]
                    noise_ss += 0.5 * (dv_flat.T @ dv_flat)
                    n_noise_pairs += int(dv_flat.shape[0])

    if n_valid < 2 or mean_vec is None or m2_mat is None:
        raise rsgispy.RSGISPyException(
            "At least 2 valid pixels are required to compute MNF."
        )
    if n_noise_pairs < 1 or noise_ss is None:
        raise rsgispy.RSGISPyException(
            "At least 1 adjacent valid pixel pair is required to estimate noise."
        )

    data_cov = 0.5 * ((m2_mat + m2_mat.T) / float(n_valid - 1))
    noise_cov = 0.5 * ((noise_ss + noise_ss.T) / float(n_noise_pairs))

    noise_vals, noise_vecs = np.linalg.eigh(noise_cov)
    max_nv = float(np.max(noise_vals))
    if max_nv <= 0.0:
        raise rsgispy.RSGISPyException(
            "Estimated spatial noise covariance matrix is identically zero; "
            "cannot compute noise-whitening transformation."
        )
    reg_eps = max(max_nv * 1e-10, 1e-15)
    safe_nv = np.maximum(noise_vals, reg_eps)
    whiten_mat = noise_vecs @ np.diag(safe_nv**-0.5)

    whitened_cov = whiten_mat.T @ data_cov @ whiten_mat
    whitened_cov = 0.5 * (whitened_cov + whitened_cov.T)
    mnf_vals, rot_vecs = np.linalg.eigh(whitened_cov)
    order = np.argsort(mnf_vals)[::-1]
    mnf_vals = np.maximum(mnf_vals[order], 0.0)
    rot_vecs = rot_vecs[:, order]

    full_proj = _canonicalise_eigenvector_signs(whiten_mat @ rot_vecs)
    proj_mat = full_proj[:, :n_comp]
    sel_vals = mnf_vals[:n_comp]
    snr_est = np.maximum(sel_vals - 1.0, 0.0)

    band_names = [f"MNF_{idx + 1}" for idx in range(n_comp)]
    _stream_linear_projection(
        input_img=input_img,
        output_img=output_img,
        band_indices=band_indices,
        proj_matrix=proj_mat,
        mean_vec=mean_vec,
        scale_vec=None,
        out_band_names=band_names,
        eff_in_nd=eff_in_nd,
        eff_out_nd=eff_out_nd,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )

    mnf_info: Dict[str, Any] = {
        "n_valid_pixels": int(n_valid),
        "n_noise_pairs": int(n_noise_pairs),
        "n_components": int(n_comp),
        "noise_direction": dir_key,
        "mean": mean_vec.tolist(),
        "mnf_eigenvalues": sel_vals.tolist(),
        "snr_estimates": snr_est.tolist(),
        "transform_matrix": proj_mat.tolist(),
    }
    if out_stats_json is not None:
        out_dir = os.path.dirname(os.path.abspath(out_stats_json))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_stats_json, "w", encoding="utf-8") as handle:
            json.dump(mnf_info, handle, indent=2)
    return mnf_info


def get_tasseled_cap_sensors() -> List[str]:
    """
    Return the sorted list of built-in Tasseled Cap sensor preset names
    accepted by :func:`calc_tasseled_cap`.

    :return: Sorted list of supported sensor preset keys.

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        sensors = st.get_tasseled_cap_sensors()
    """
    return sorted(_TASSELED_CAP_PRESETS.keys())


def calc_tasseled_cap(
    input_img: str,
    output_img: str,
    sensor: Optional[str] = "landsat8_oli",
    bands: Optional[Sequence[int]] = None,
    coeffs: Optional[
        Union[pd.DataFrame, Sequence[Sequence[float]], np.ndarray]
    ] = None,
    component_names: Optional[Sequence[str]] = None,
    gain: float = 1.0,
    offset: float = 0.0,
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute the orthogonal Tasseled Cap (Kauth–Thomas) transformation over
    raster windows for supported optical sensors or a custom coefficient matrix.

    Built-in sensor presets (band ordering expectations listed below):

    * ``'landsat5_tm'`` — Crist (1985), *Remote Sensing of Environment* 17(3),
      301–306 (reflectance; bands: Blue, Green, Red, NIR, SWIR1, SWIR2).
    * ``'landsat7_etm'`` — Huang et al. (2002), *Int. J. Remote Sens.* 23(8),
      1741–1748 (TOA reflectance; bands: Blue, Green, Red, NIR, SWIR1, SWIR2).
    * ``'landsat8_oli'`` / ``'landsat9_oli'`` — Baig et al. (2014),
      *Remote Sensing Letters* 5(5), 423–431 (TOA reflectance; OLI bands B2–B7:
      Blue, Green, Red, NIR, SWIR1, SWIR2).
    * ``'sentinel2_msi'`` — Shi & Xu (2019), *IEEE JSTARS* 12(10), 4038–4048
      (reflectance; bands: B2 Blue, B3 Green, B4 Red, B8 NIR, B11 SWIR1,
      B12 SWIR2).
    * ``'modis_nbar'`` — Lobser & Cohen (2007), *Int. J. Remote Sens.* 28(22),
      5079–5098 (NBAR reflectance; MODIS bands B1 Red, B2 NIR1, B3 Blue,
      B4 Green, B5 NIR2, B6 SWIR1, B7 SWIR2).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output Tasseled Cap raster file.
    :param sensor: Built-in sensor preset name (see
                   :func:`get_tasseled_cap_sensors`). Ignored when ``coeffs`` is
                   supplied (Default = ``'landsat8_oli'``).
    :param bands: Optional 1-based band indices from ``input_img`` ordered to
                  match the sensor preset or coefficient columns/rows
                  (defaults to all image bands when ``None``).
    :param coeffs: Optional custom Tasseled Cap coefficients overriding
                   ``sensor``: either a :class:`pandas.DataFrame` of shape
                   ``(n_bands, n_components)`` (columns = component names) or a
                   2D array/sequence of shape ``(n_components, n_bands)``.
    :param component_names: Optional sequence of output band names (defaults to
                            DataFrame columns, preset names
                            ``['Brightness', 'Greenness', 'Wetness']``, or
                            ``['TC_1', ...]``).
    :param gain: Optional scale divisor applied before projection as
                 ``(pixel - offset) / gain`` (Default = ``1.0``).
    :param offset: Optional additive offset subtracted before dividing by
                   ``gain`` (Default = ``0.0``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param out_no_data_val: Output nodata value (defaults to ``no_data_val`` if
                            set, else ``-9999.0`` for float outputs or ``0``
                            for integer outputs).
    :param valid_msk_img: Optional path to a valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant for the output raster
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        st.calc_tasseled_cap(
            "s2_6band.tif",
            "s2_tct.tif",
            sensor="sentinel2_msi",
            gain=10000.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches
    if gain == 0.0:
        raise rsgispy.RSGISPyException("gain must not be zero.")

    if coeffs is not None:
        if isinstance(coeffs, pd.DataFrame):
            if coeffs.empty:
                raise rsgispy.RSGISPyException(
                    "coeffs DataFrame must not be empty."
                )
            proj_mat = coeffs.to_numpy(dtype=np.float64)
            default_names = [str(col) for col in coeffs.columns]
            req_n_bands, n_comp = proj_mat.shape
        else:
            raw_arr = np.asarray(coeffs, dtype=np.float64)
            if raw_arr.ndim != 2 or raw_arr.shape[0] < 1 or raw_arr.shape[1] < 1:
                raise rsgispy.RSGISPyException(
                    "coeffs array must be 2D with shape (n_components, n_bands)."
                )
            n_comp, req_n_bands = raw_arr.shape
            proj_mat = raw_arr.T
            default_names = [f"TC_{idx + 1}" for idx in range(n_comp)]
    else:
        if sensor is None:
            raise rsgispy.RSGISPyException(
                "Provide either a supported sensor name or custom coeffs."
            )
        sensor_key = sensor.strip().lower()
        if sensor_key not in _TASSELED_CAP_PRESETS:
            raise rsgispy.RSGISPyException(
                f"Unknown Tasseled Cap sensor '{sensor}'. Supported presets: "
                f"{get_tasseled_cap_sensors()}."
            )
        preset = _TASSELED_CAP_PRESETS[sensor_key]
        raw_preset = preset["coeffs"]
        n_comp, req_n_bands = raw_preset.shape
        proj_mat = raw_preset.T
        default_names = list(preset["components"])

    if component_names is not None:
        out_names = [str(name) for name in component_names]
        if len(out_names) != n_comp:
            raise rsgispy.RSGISPyException(
                f"Length of component_names ({len(out_names)}) does not match "
                f"component count ({n_comp})."
            )
    else:
        out_names = default_names

    ensure_file_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        band_indices = resolve_selected_bands(src, bands, input_img)

    if len(band_indices) != req_n_bands:
        raise rsgispy.RSGISPyException(
            f"Expected {req_n_bands} input bands for Tasseled Cap transform, "
            f"got {len(band_indices)}."
        )

    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    eff_in_nd, eff_out_nd = resolve_eff_nodata(
        input_img, no_data_val, out_no_data_val, out_np_dtype
    )

    mean_vec = (
        np.full(req_n_bands, float(offset), dtype=np.float64)
        if offset != 0.0
        else None
    )
    scale_vec = (
        np.full(req_n_bands, float(gain), dtype=np.float64)
        if gain != 1.0
        else None
    )

    _stream_linear_projection(
        input_img=input_img,
        output_img=output_img,
        band_indices=band_indices,
        proj_matrix=proj_mat,
        mean_vec=mean_vec,
        scale_vec=scale_vec,
        out_band_names=out_names,
        eff_in_nd=eff_in_nd,
        eff_out_nd=eff_out_nd,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        gdalformat=gdalformat,
        datatype=datatype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def _compute_streaming_band_cdfs(
    img_path: str,
    band_indices: Sequence[int],
    n_bins: int,
    no_data_val: Optional[float],
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    use_pbar: bool,
) -> Tuple[List[float], List[float], List[np.ndarray], List[np.ndarray]]:
    """
    Compute per-band ``(vmin, vmax, bin_edges, cdf)`` over finite, valid pixels
    in two streaming passes.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_bands = len(band_indices)
    vmins = [np.inf] * n_bands
    vmaxs = [-np.inf] * n_bands
    counts = [0] * n_bands

    with ExitStack() as stack:
        img_ds, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, img_path, valid_msk_img
        )
        band_nds = [
            no_data_val
            if no_data_val is not None
            else img_ds.nodatavals[b_idx - 1]
            for b_idx in band_indices
        ]

        # Pass 1: exact streaming min / max per band
        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = img_ds.read(
                band_indices, window=img_win, out_dtype=np.float64
            )
            msk_valid: Optional[np.ndarray] = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_valid = msk_ds.read(1, window=msk_win) == valid_msk_val

            for idx in range(n_bands):
                arr = blk[idx]
                vld = np.isfinite(arr)
                nd = band_nds[idx]
                if nd is not None and math.isfinite(float(nd)):
                    vld &= arr != float(nd)
                if msk_valid is not None:
                    vld &= msk_valid
                if np.any(vld):
                    vals = arr[vld]
                    vmins[idx] = min(vmins[idx], float(np.min(vals)))
                    vmaxs[idx] = max(vmaxs[idx], float(np.max(vals)))
                    counts[idx] += int(vals.size)

        for idx, cnt in enumerate(counts):
            if cnt == 0:
                raise rsgispy.RSGISPyException(
                    f"Band {band_indices[idx]} of '{img_path}' has no valid pixels "
                    "for histogram matching."
                )

        edges_list = [
            np.linspace(vmins[idx], vmaxs[idx], n_bins + 1, dtype=np.float64)
            if vmaxs[idx] > vmins[idx]
            else np.array([vmins[idx], vmaxs[idx]], dtype=np.float64)
            for idx in range(n_bands)
        ]
        hists = [
            np.zeros(n_bins, dtype=np.int64)
            if vmaxs[idx] > vmins[idx]
            else np.zeros(1, dtype=np.int64)
            for idx in range(n_bands)
        ]

        # Pass 2: accumulate per-band histograms across windows
        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = img_ds.read(
                band_indices, window=img_win, out_dtype=np.float64
            )
            msk_valid = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_valid = msk_ds.read(1, window=msk_win) == valid_msk_val

            for idx in range(n_bands):
                if vmaxs[idx] <= vmins[idx]:
                    continue
                arr = blk[idx]
                vld = np.isfinite(arr)
                nd = band_nds[idx]
                if nd is not None and math.isfinite(float(nd)):
                    vld &= arr != float(nd)
                if msk_valid is not None:
                    vld &= msk_valid
                if np.any(vld):
                    cnt_hist, _ = np.histogram(arr[vld], bins=edges_list[idx])
                    hists[idx] += cnt_hist

    cdfs: List[np.ndarray] = []
    for idx in range(n_bands):
        if vmaxs[idx] <= vmins[idx]:
            cdfs.append(np.array([0.0, 1.0], dtype=np.float64))
        else:
            cum = np.cumsum(hists[idx], dtype=np.float64) / float(counts[idx])
            cdfs.append(np.concatenate(([0.0], cum)))
    return vmins, vmaxs, edges_list, cdfs


def histogram_matching(
    input_img: str,
    ref_img: str,
    output_img: str,
    bands: Optional[Sequence[int]] = None,
    ref_bands: Optional[Sequence[int]] = None,
    n_bins: int = 4096,
    no_data_val: Optional[float] = None,
    ref_no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    ref_valid_msk_img: Optional[str] = None,
    ref_valid_msk_val: int = 1,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Match the per-band empirical cumulative distribution function (CDF) of
    ``input_img`` to that of ``ref_img`` using windowed streaming histograms.

    Useful for radiometric normalisation across multi-temporal scenes or mosaic
    seam balancing. Both images are processed in windows without loading full
    rasters into memory, and need not share the same spatial dimensions or CRS.

    :param input_img: Path to the source raster image file to adjust.
    :param ref_img: Path to the reference raster image providing target CDFs.
    :param output_img: Path to the histogram-matched output raster file.
    :param bands: Optional 1-based band indices from ``input_img`` to match
                  (defaults to all bands when ``None``).
    :param ref_bands: Optional 1-based band indices from ``ref_img`` matching
                      the length of ``bands`` (defaults to ``bands`` or all
                      bands of ``ref_img`` when ``None``).
    :param n_bins: Number of histogram bins used for streaming CDF estimation
                   (Default = ``4096``).
    :param no_data_val: Explicit nodata value for ``input_img`` overriding
                        metadata (Default = ``None``).
    :param ref_no_data_val: Explicit nodata value for ``ref_img`` overriding
                            metadata (Default = ``None``).
    :param out_no_data_val: Output nodata value (defaults to ``input_img``
                            nodata value when ``None``).
    :param valid_msk_img: Optional valid-region mask image for ``input_img``.
    :param valid_msk_val: Mask value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param ref_valid_msk_img: Optional valid-region mask image for ``ref_img``.
    :param ref_valid_msk_val: Mask value identifying valid pixels in
                              ``ref_valid_msk_img`` (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output raster
                     (defaults to matching ``input_img`` dtype when ``None``).
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.transforms as st

        st.histogram_matching(
            "slave_scene.tif",
            "master_scene.tif",
            "matched_scene.tif",
            n_bins=4096,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches
    if int(n_bins) < 2:
        raise rsgispy.RSGISPyException("n_bins must be at least 2.")

    ensure_file_exists(input_img)
    ensure_file_exists(ref_img, label="Reference image")

    with rasterio.open(input_img, "r") as in_src:
        in_band_list = resolve_selected_bands(in_src, bands, input_img)
        in_native_rio_dtype = str(in_src.dtypes[in_band_list[0] - 1])
        in_descriptions = [in_src.descriptions[b - 1] for b in in_band_list]
        in_band_nds = [
            no_data_val if no_data_val is not None else in_src.nodatavals[b - 1]
            for b in in_band_list
        ]

    with rasterio.open(ref_img, "r") as ref_src:
        if ref_bands is not None:
            ref_band_list = resolve_selected_bands(ref_src, ref_bands, ref_img)
        elif bands is not None:
            ref_band_list = resolve_selected_bands(ref_src, bands, ref_img)
        else:
            ref_band_list = resolve_selected_bands(ref_src, None, ref_img)

    if len(in_band_list) != len(ref_band_list):
        raise rsgispy.RSGISPyException(
            f"Number of selected input bands ({len(in_band_list)}) does not "
            f"match reference bands ({len(ref_band_list)})."
        )

    in_mins, in_maxs, in_edges, in_cdfs = _compute_streaming_band_cdfs(
        input_img,
        in_band_list,
        int(n_bins),
        no_data_val,
        valid_msk_img,
        valid_msk_val,
        use_pbar,
    )
    ref_mins, ref_maxs, ref_edges, ref_cdfs = _compute_streaming_band_cdfs(
        ref_img,
        ref_band_list,
        int(n_bins),
        ref_no_data_val,
        ref_valid_msk_img,
        ref_valid_msk_val,
        use_pbar,
    )

    if datatype is None:
        out_rio_dtype = in_native_rio_dtype
        out_np_dtype = np.dtype(out_rio_dtype)
    else:
        out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    eff_in_nd = (
        no_data_val if no_data_val is not None else get_img_no_data_value(input_img)
    )
    eff_out_nd: Optional[float]
    if out_no_data_val is not None:
        eff_out_nd = float(out_no_data_val)
    elif eff_in_nd is not None and math.isfinite(float(eff_in_nd)):
        eff_out_nd = float(eff_in_nd)
    else:
        eff_out_nd = None

    fill_val = (
        eff_out_nd
        if eff_out_nd is not None
        else (np.nan if np.issubdtype(out_np_dtype, np.floating) else 0.0)
    )
    n_bands = len(in_band_list)

    with ExitStack() as stack:
        img_ds, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            output_exists=False,
        )
        if eff_out_nd is not None:
            dst.nodata = eff_out_nd
        for idx, desc in enumerate(in_descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            img_win = Window(
                img_off[0] + col_off, img_off[1] + row_off, win_w, win_h
            )
            blk = img_ds.read(
                in_band_list, window=img_win, out_dtype=np.float64
            )
            msk_valid: Optional[np.ndarray] = None
            if msk_ds is not None and msk_off is not None:
                msk_win = Window(
                    msk_off[0] + col_off, msk_off[1] + row_off, win_w, win_h
                )
                msk_valid = msk_ds.read(1, window=msk_win) == valid_msk_val

            out_blk = np.full(
                (n_bands, win_h, win_w), fill_val, dtype=np.float64
            )
            for idx in range(n_bands):
                arr = blk[idx]
                vld = np.isfinite(arr)
                nd = in_band_nds[idx]
                if nd is not None and math.isfinite(float(nd)):
                    vld &= arr != float(nd)
                if msk_valid is not None:
                    vld &= msk_valid
                if np.any(vld):
                    vals = arr[vld]
                    if (
                        in_maxs[idx] <= in_mins[idx]
                        or ref_maxs[idx] <= ref_mins[idx]
                    ):
                        mapped = np.full(
                            vals.shape, ref_mins[idx], dtype=np.float64
                        )
                    else:
                        quantiles = np.interp(
                            vals, in_edges[idx], in_cdfs[idx]
                        )
                        mapped = np.interp(
                            quantiles, ref_cdfs[idx], ref_edges[idx]
                        )
                    out_blk[idx][vld] = mapped

            if np.issubdtype(out_np_dtype, np.integer):
                out_blk = np.rint(out_blk)
            dst.write(
                cast_result_block(
                    out_blk, win_h, win_w, out_np_dtype, clamp_int=True
                ),
                window=dst_win,
            )

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=eff_out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
