# pylint: disable=too-many-lines
"""
The imagespectral.unmixing module provides linear spectral unmixing solvers,
abundance rescaling, reflectance prediction, RMSE/RMSPE evaluation, and
multi-endmember model summarisation using ``rasterio``, ``numpy``, and
``pandas``.

**Recommended unmixing functions:**

* :func:`spec_unmix_ucls` — vectorised unconstrained least squares (core
  ``numpy`` only)
* :func:`spec_unmix_nnls` — batched active-set Lawson–Hanson non-negative
  least squares (requires ``scipy``)
* :func:`spec_unmix_fcls` — batched affine-nullspace fully constrained least
  squares (non-negative and sum-to-one, requires ``scipy``)

Functions wrapping ``pysptools``, ``cvxopt``, and ``pymcr`` are also retained
for compatibility and raise :class:`rsgispy.RSGISPyException` with
installation instructions if their optional dependencies are not installed.
"""

from contextlib import ExitStack
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    fits_dtype_range,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
    resolve_out_nodata,
)
from rsgispy.imagespectral.endmembers import (
    EndmemberInput,
    ImageEndmemberInfo,
    coerce_endmembers_df,
    prepare_endmembers_array,
)
from rsgispy.imageutils.metadata import set_band_names
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats

try:
    import scipy.optimize

    HAVE_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    HAVE_SCIPY = False

try:
    import pysptools.abundance_maps.amaps as _spts_amaps  # pylint: disable=import-error

    HAVE_PYSPTOOLS = True
except ImportError:
    _spts_amaps = None
    HAVE_PYSPTOOLS = False

try:
    import cvxopt as _cvxopt  # noqa: F401  # pylint: disable=import-error,unused-import

    HAVE_CVXOPT = True
except ImportError:
    HAVE_CVXOPT = False

try:
    from pymcr.constraints import (  # pylint: disable=import-error
        ConstraintNonneg,
        ConstraintNorm,
    )
    from pymcr.mcr import McrAR  # pylint: disable=import-error
    from pymcr.regressors import NNLS as _PyMcrNNLS  # pylint: disable=import-error

    HAVE_PYMCR = True
except ImportError:
    ConstraintNonneg = None
    ConstraintNorm = None
    McrAR = None
    _PyMcrNNLS = None
    HAVE_PYMCR = False

__all__: List[str] = [
    "exhcon_linear_spec_unmix",
    "spec_unmix_ucls",
    "spec_unmix_nnls",
    "spec_unmix_fcls",
    "spec_unmix_spts_ucls",
    "spec_unmix_spts_nnls",
    "spec_unmix_spts_fcls",
    "spec_unmix_pymcr_nnls",
    "spec_unmix_pymcr_fcls",
    "rescale_unmixing_results",
    "predict_refl_linear_unmixing",
    "calc_unmixing_rmse_residual_err",
    "summarise_multi_endmember_linear_unmixing",
]


def _check_out_no_data_val(
    out_no_data_val: Optional[float], out_np_dtype: np.dtype = np.dtype("float32")
) -> float:
    """
    Resolve and validate the output no-data value of an unmixing output.

    ``None`` gives ``-9999`` (or, for integer data types that cannot hold it,
    the type's maximum for unsigned and minimum for signed types). An explicit
    value must be representable by ``out_np_dtype`` (``NaN`` is allowed for
    floating-point outputs).
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if out_no_data_val is None:
        return float(
            resolve_out_nodata(
                out_np_dtype,
                None,
                None,
                default_float=-9999.0,
                default_int=-9999.0,
                clamp_int=True,
                unsigned_clamp="max",
            )
        )
    nd_val = float(out_no_data_val)
    if math.isnan(nd_val) and np.issubdtype(out_np_dtype, np.floating):
        return nd_val
    if not fits_dtype_range(nd_val, out_np_dtype):
        raise rsgispy.RSGISPyException(
            f"out_no_data_val ({out_no_data_val}) cannot be represented by the "
            f"output data type '{out_np_dtype.name}'."
        )
    return nd_val


def _valid_pxls(flat: np.ndarray, nodatavals: Sequence[Optional[float]]) -> np.ndarray:
    """
    Return a boolean mask of the rows of a ``(n_pxls, n_bands)`` array that
    are finite in every band and not equal to that band's no-data value.
    """
    vld = np.all(np.isfinite(flat), axis=1)
    for b_idx, b_nd in enumerate(nodatavals[: flat.shape[1]]):
        if b_nd is not None and math.isfinite(float(b_nd)):
            vld &= flat[:, b_idx] != float(b_nd)
    return vld


def _build_simplex_weights(n_endmembers: int, step_res: float) -> np.ndarray:
    """
    Generate all non-negative endmember weight combinations summing to 1.0
    on a simplex grid of resolution ``step_res``.
    """
    if step_res <= 0.0 or step_res > 1.0:
        raise rsgispy.RSGISPyException(
            f"step_res must be in (0, 1], got {step_res}."
        )

    n_steps = int(round(1.0 / step_res))
    if math.isclose(n_steps * step_res, 1.0, rel_tol=1e-6, abs_tol=1e-7):
        rows: List[Tuple[int, ...]] = []

        def _gen_int(pos: int, remaining: int, cur: List[int]) -> None:
            if pos == n_endmembers - 1:
                rows.append(tuple(cur + [remaining]))
                return
            for val in range(remaining + 1):
                cur.append(val)
                _gen_int(pos + 1, remaining - val, cur)
                cur.pop()

        _gen_int(0, n_steps, [])
        return np.asarray(rows, dtype=np.float64) / float(n_steps)

    grid_vals: List[List[float]] = []
    steps = np.arange(0.0, 1.0 + 1e-9, step_res)

    def _gen_flt(pos: int, rem: float, cur: List[float]) -> None:
        if pos == n_endmembers - 1:
            if -1e-6 <= rem <= 1.0 + 1e-6:
                grid_vals.append(cur + [max(0.0, rem)])
            return
        for v in steps:
            if v <= rem + 1e-6:
                cur.append(float(v))
                _gen_flt(pos + 1, rem - float(v), cur)
                cur.pop()

    _gen_flt(0, 1.0, [])
    if not grid_vals:
        raise rsgispy.RSGISPyException(
            f"No valid simplex weights generated for step_res={step_res}."
        )
    return np.asarray(grid_vals, dtype=np.float64)


def exhcon_linear_spec_unmix(
    input_img: str,
    output_img: str,
    endmembers: Optional[EndmemberInput] = None,
    step_res: float = 0.1,
    gain: float = 1.0,
    offset: float = 0.0,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    out_no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> None:
    """
    Perform exhaustive constrained linear spectral unmixing of ``input_img``.

    Both pixel spectra and endmember spectra are L2-normalised prior to
    searching all non-negative abundance combinations summing to 1.0 at grid
    step ``step_res``. The output raster contains ``m + 1`` bands named after
    the endmember columns followed by ``RMSE`` (scaled via
    ``offset + gain * value``). Pixels that cannot be unmixed (a non-finite
    value or the input image's no-data value in any band, or all bands zero,
    so the spectrum has no direction) are written as ``out_no_data_val`` in
    every band, which is also recorded as the output no-data value.

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output unmixed raster image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param step_res: Search step resolution in ``(0, 1]`` (Default = ``0.1``).
    :param gain: Multiplicative gain applied to output values
                 (Default = ``1.0``).
    :param offset: Additive offset applied to output values (Default = ``0.0``).
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param out_no_data_val: Output no-data value for pixels that are not
                            unmixed (Default = ``None``: ``-9999``, or for
                            integer data types that cannot hold it the type's
                            maximum (unsigned) or minimum (signed) value). It
                            must be representable by ``datatype``.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagespectral.unmixing as su

        su.exhcon_linear_spec_unmix(
            input_img="s2_refl.tif",
            output_img="unmixed.tif",
            endmembers="endmembers.csv",
            step_res=0.1,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-statements
    ensure_file_exists(input_img, "Input image")
    if endmembers is None:
        raise rsgispy.RSGISPyException("endmembers must be provided.")

    em_names, m, n_bands, em_arr = prepare_endmembers_array(
        endmembers, file_format=file_format
    )
    em_norms = np.linalg.norm(em_arr, axis=1, keepdims=True)
    if np.any(em_norms == 0):
        raise rsgispy.RSGISPyException(
            "One or more endmember spectra have zero norm."
        )
    em_normed = em_arr / em_norms

    cand_weights = _build_simplex_weights(m, float(step_res))  # (C, m)
    cand_spectra = cand_weights @ em_normed  # (C, n_bands)
    cand_sq_norms = np.sum(cand_spectra * cand_spectra, axis=1)  # (C,)
    cand_spectra_t = cand_spectra.T  # (n_bands, C)

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    out_nd = _check_out_no_data_val(out_no_data_val, out_np_dtype)
    n_out_bands = m + 1
    chunk_size = 8192

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.count != n_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src.count} bands, but "
                f"endmembers expect {n_bands} bands."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=n_out_bands,
        )
        dst.nodata = out_nd
        for b_idx, b_name in enumerate(em_names + ["RMSE"], start=1):
            dst.set_band_description(b_idx, b_name)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(window=win).astype(np.float64, copy=False)
            flat = np.moveaxis(blk, 0, -1).reshape(-1, n_bands)

            valid = _valid_pxls(flat, src.nodatavals)
            norms = np.zeros(flat.shape[0], dtype=np.float64)
            norms[valid] = np.linalg.norm(flat[valid], axis=1)
            valid &= norms > 0.0
            out_flat = np.full(
                (flat.shape[0], n_out_bands), out_nd, dtype=np.float64
            )

            if np.any(valid):
                v_idx = np.flatnonzero(valid)
                pxl_normed = flat[v_idx] / norms[v_idx, np.newaxis]
                pxl_sq = np.sum(pxl_normed * pxl_normed, axis=1)

                n_valid = v_idx.size
                for start in range(0, n_valid, chunk_size):
                    end = min(start + chunk_size, n_valid)
                    sub_p = pxl_normed[start:end]
                    sub_sq = (
                        pxl_sq[start:end, np.newaxis]
                        + cand_sq_norms[np.newaxis, :]
                    )
                    sq_dists = sub_sq - 2.0 * (sub_p @ cand_spectra_t)
                    best_c = np.argmin(sq_dists, axis=1)
                    min_sq = np.maximum(
                        0.0, sq_dists[np.arange(end - start), best_c]
                    )
                    rmse = np.sqrt(min_sq / float(n_bands))

                    out_idx = v_idx[start:end]
                    out_flat[out_idx, :m] = offset + (
                        gain * cand_weights[best_c]
                    )
                    out_flat[out_idx, m] = offset + (gain * rmse)

            out_blk = np.moveaxis(
                out_flat.reshape((win_h, win_w, n_out_bands)), -1, 0
            )
            dst.write(
                cast_result_block(
                    out_blk, win_h, win_w, out_np_dtype, clamp_int=False
                ),
                window=win,
            )


def _run_windowed_unmixer(
    input_img: str,
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    output_img: str,
    endmembers: EndmemberInput,
    solver_fn,
    gdalformat: str = "GTIFF",
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    out_no_data_val: float = -9999.0,
) -> None:
    """
    Shared windowed linear spectral unmixing runner.

    Pixels outside the valid mask, with a non-finite value, the input image's
    no-data value or a value ``<= 0`` in any band are not unmixed and are
    written as ``out_no_data_val`` in every band.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    out_nd = _check_out_no_data_val(out_no_data_val)
    em_names, m, _, endmembers_arr = prepare_endmembers_array(
        endmembers, gain=gain, weight=weight, file_format=file_format
    )

    in_paths = (
        [input_img] if valid_msk_img is None else [input_img, valid_msk_img]
    )
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, in_paths)
        src_img = datasets[paths[0]]
        src_msk = datasets[paths[1]] if valid_msk_img is not None else None
        grid = compute_spatial_intersection(datasets)

        expected_bands = endmembers_arr.shape[1] - (
            1 if weight is not None else 0
        )
        if src_img.count != expected_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src_img.count} bands, expected "
                f"{expected_bands}."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=m,
        )
        dst.nodata = out_nd
        for b_idx, b_name in enumerate(em_names, start=1):
            dst.set_band_description(b_idx, b_name)

        img_xoff, img_yoff = grid.offsets[paths[0]]
        msk_xoff, msk_yoff = (
            grid.offsets[paths[1]] if valid_msk_img is not None else (0, 0)
        )

        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(win.height)
            win_w = int(win.width)
            img_win = Window(
                int(win.col_off) + img_xoff,
                int(win.row_off) + img_yoff,
                win_w,
                win_h,
            )

            out_flat = np.full((win_h * win_w, m), out_nd, dtype=np.float32)
            if src_msk is not None:
                msk_win = Window(
                    int(win.col_off) + msk_xoff,
                    int(win.row_off) + msk_yoff,
                    win_w,
                    win_h,
                )
                msk_vld = (
                    src_msk.read(1, window=msk_win).ravel() == valid_msk_val
                )
            else:
                msk_vld = np.ones(win_h * win_w, dtype=bool)

            if np.any(msk_vld):
                img_blk = src_img.read(window=img_win).astype(
                    np.float64, copy=False
                )
                img_flat = np.moveaxis(img_blk, 0, -1).reshape(
                    -1, src_img.count
                )

                vld = (
                    msk_vld
                    & _valid_pxls(img_flat, src_img.nodatavals)
                    & np.all(img_flat > 0.0, axis=1)
                )
                if np.any(vld):
                    vld_pxls = img_flat[vld]
                    if gain > 1:
                        vld_pxls = vld_pxls / float(gain)
                    if weight is not None:
                        w_col = np.full(
                            (vld_pxls.shape[0], 1),
                            float(weight),
                            dtype=np.float64,
                        )
                        vld_pxls = np.hstack([vld_pxls, w_col])

                    abundances = solver_fn(vld_pxls, endmembers_arr)
                    out_flat[vld] = abundances.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, m)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _numpy_ucls_solver(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """Vectorised unconstrained least squares solver using ``numpy.linalg.lstsq``."""
    abundances, _, _, _ = np.linalg.lstsq(
        endmembers_arr.T, vld_pxls.T, rcond=None
    )
    return abundances.T


def spec_unmix_ucls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    **[Recommended]** Perform unconstrained least-squares (UCLS) linear
    spectral unmixing using vectorised ``numpy.linalg.lstsq``.

    Requires only core ``numpy`` (no optional unmixing packages needed).

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file
                          (if ``None``, all finite positive pixels are unmixed).
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels
                          (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_ucls(
            "img.tif", "unmix_ucls.tif", "endmembers.csv", "mask.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_numpy_ucls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def _scipy_nnls_loop(vld_pxls: np.ndarray, end_t: np.ndarray) -> np.ndarray:
    """Fallback row-wise Lawson–Hanson NNLS solver via ``scipy.optimize.nnls``."""
    n_pxls = vld_pxls.shape[0]
    m = end_t.shape[1]
    out = np.empty((n_pxls, m), dtype=np.float64)
    for idx in range(n_pxls):
        out[idx] = scipy.optimize.nnls(end_t, vld_pxls[idx])[0]
    return out


def _scipy_nnls_solver(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """
    Batched active-set (Lawson–Hanson) NNLS solver using NumPy/SciPy.

    Solves batches sharing the same passive variable set in a single linear
    least-squares call, falling back to ``scipy.optimize.nnls`` for high
    endmember counts (``m > 20``) or degenerate active sets.
    """
    # pylint: disable=too-many-locals
    rsgispy.check_optional_dep(HAVE_SCIPY, "spec_unmix_nnls", "scipy", "unmixing")
    end_t = np.ascontiguousarray(endmembers_arr.T, dtype=np.float64)
    n_pxls = vld_pxls.shape[0]
    m = endmembers_arr.shape[0]
    if n_pxls == 0 or m == 0:
        return np.empty((n_pxls, m), dtype=np.float64)
    if m > 20:
        return _scipy_nnls_loop(vld_pxls, end_t)

    x0 = np.linalg.lstsq(end_t, vld_pxls.T, rcond=None)[0].T
    out = np.empty((n_pxls, m), dtype=np.float64)
    unconstrained_ok = np.all(x0 >= -1e-12, axis=1)
    if np.any(unconstrained_ok):
        out[unconstrained_ok] = np.clip(x0[unconstrained_ok], 0.0, None)

    rem_idx = np.flatnonzero(~unconstrained_ok)
    if rem_idx.size == 0:
        return out

    y_rem = np.ascontiguousarray(vld_pxls[rem_idx], dtype=np.float64)
    n_rem = y_rem.shape[0]
    x_mat = np.zeros((n_rem, m), dtype=np.float64)
    p_mask = np.zeros((n_rem, m), dtype=bool)
    active = np.ones(n_rem, dtype=bool)

    e_norm = max(float(np.linalg.norm(end_t)), 1e-12)
    y_norms = np.maximum(np.linalg.norm(y_rem, axis=1), 1e-12)
    weights = np.int64(1) << np.arange(m, dtype=np.int64)

    for _ in range(max(3 * m, 10)):
        act_idx = np.flatnonzero(active)
        if act_idx.size == 0:
            break
        grad = (x_mat[act_idx] @ end_t.T - y_rem[act_idx]) @ end_t
        masked_grad = np.where(~p_mask[act_idx], grad, np.inf)
        best_col = np.argmin(masked_grad, axis=1)
        min_g = masked_grad[np.arange(act_idx.size), best_col]
        tol_act = (
            1e-11
            * e_norm
            * np.maximum(
                e_norm * np.linalg.norm(x_mat[act_idx], axis=1), y_norms[act_idx]
            )
        )
        conv = min_g >= -tol_act
        if np.any(conv):
            active[act_idx[conv]] = False
        need_add = act_idx[~conv]
        if need_add.size == 0:
            break
        p_mask[need_add, best_col[~conv]] = True

        for _ in range(m + 2):
            keys = p_mask[need_add] @ weights
            s_mat = np.zeros((need_add.size, m), dtype=np.float64)
            for key in np.unique(keys):
                if key == 0:
                    continue
                row_sel = np.flatnonzero(keys == key)
                cols = np.flatnonzero((key & weights) != 0)
                s_sub = np.linalg.lstsq(
                    end_t[:, cols], y_rem[need_add[row_sel]].T, rcond=None
                )[0].T
                s_mat[np.ix_(row_sel, cols)] = s_sub

            neg_mask = np.any((s_mat <= -1e-12) & p_mask[need_add], axis=1)
            if np.any(~neg_mask):
                x_mat[need_add[~neg_mask]] = np.clip(s_mat[~neg_mask], 0.0, None)
            if not np.any(neg_mask):
                break

            fix = need_add[neg_mask]
            s_fix = s_mat[neg_mask]
            x_fix = x_mat[fix]
            valid_step = p_mask[fix] & (s_fix <= 0.0) & (x_fix > s_fix)
            denom = np.where(valid_step, x_fix - s_fix, 1.0)
            ratios = np.where(valid_step, x_fix / denom, np.inf)
            alpha = np.min(ratios, axis=1, keepdims=True)
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            x_new = x_fix + alpha * (s_fix - x_fix)
            x_new[x_new < 1e-12] = 0.0
            x_mat[fix] = x_new
            p_mask[fix] &= x_new > 0.0
            need_add = fix

    if np.any(active):
        rem_act = np.flatnonzero(active)
        x_mat[rem_act] = _scipy_nnls_loop(y_rem[rem_act], end_t)

    out[rem_idx] = np.clip(x_mat, 0.0, None)
    return out


def _solve_simplex_eq_subset(
    end_t: np.ndarray,
    y_mat: np.ndarray,
    cols: np.ndarray,
    null_basis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve equality-constrained least squares (sum = 1) on passive columns."""
    k = int(cols.size)
    if k == 1:
        return np.ones((y_mat.shape[0], 1), dtype=np.float64)
    if null_basis is None:
        q_full, _ = np.linalg.qr(
            np.ones((k, 1), dtype=np.float64), mode="complete"
        )
        null_basis = q_full[:, 1:]
    e_sub = end_t[:, cols]
    x0 = np.full(k, 1.0 / float(k), dtype=np.float64)
    rhs = y_mat.T - (e_sub @ x0)[:, None]
    w_coef = np.linalg.lstsq(e_sub @ null_basis, rhs, rcond=None)[0]
    return (x0[:, None] + null_basis @ w_coef).T


def _scipy_fcls_heinz_chang_loop(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """Fallback row-wise Heinz & Chang (2001) FCLS via ``scipy.optimize.nnls``."""
    m = endmembers_arr.shape[0]
    n_pxls = vld_pxls.shape[0]
    scale = max(float(np.max(np.abs(endmembers_arr))), 1e-12)
    delta = 1e-5 / scale
    end_aug = np.vstack(
        [delta * endmembers_arr.T, np.ones((1, m), dtype=np.float64)]
    )
    pxls_aug = np.hstack(
        [delta * vld_pxls, np.ones((n_pxls, 1), dtype=np.float64)]
    )
    out = np.empty((n_pxls, m), dtype=np.float64)
    for idx in range(n_pxls):
        out[idx] = scipy.optimize.nnls(end_aug, pxls_aug[idx])[0]
    out = np.clip(out, 0.0, None)
    row_sums = np.sum(out, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(row_sums > 0.0, out / row_sums, 0.0)
    return out


def _scipy_fcls_solver(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """
    Batched affine-nullspace active-set FCLS solver (non-negative, sum-to-one).

    Minimises :math:`\\|E^T x - y\\|_2` directly on the simplex
    :math:`x \\ge 0, \\mathbf{1}^T x = 1` across pixel batches sharing the same
    active set, with scale-invariant KKT checks and ``scipy.optimize.nnls``
    fallback for high endmember counts (``m > 20``) or degenerate pixels.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    rsgispy.check_optional_dep(HAVE_SCIPY, "spec_unmix_fcls", "scipy", "unmixing")
    end_t = np.ascontiguousarray(endmembers_arr.T, dtype=np.float64)
    n_pxls = vld_pxls.shape[0]
    m = endmembers_arr.shape[0]
    if n_pxls == 0 or m == 0:
        return np.empty((n_pxls, m), dtype=np.float64)
    if m == 1:
        return np.ones((n_pxls, 1), dtype=np.float64)
    if m > 20:
        return _scipy_fcls_heinz_chang_loop(vld_pxls, endmembers_arr)

    null_cache: Dict[int, np.ndarray] = {}

    def _get_null(k_size: int) -> np.ndarray:
        if k_size not in null_cache:
            q_full, _ = np.linalg.qr(
                np.ones((k_size, 1), dtype=np.float64), mode="complete"
            )
            null_cache[k_size] = q_full[:, 1:]
        return null_cache[k_size]

    all_cols = np.arange(m, dtype=np.int64)
    x_full = _solve_simplex_eq_subset(
        end_t, vld_pxls, all_cols, null_basis=_get_null(m)
    )
    out = np.empty((n_pxls, m), dtype=np.float64)
    ok_full = np.all(x_full >= -1e-12, axis=1)
    if np.any(ok_full):
        xf = np.clip(x_full[ok_full], 0.0, None)
        xf_sum = np.sum(xf, axis=1, keepdims=True)
        out[ok_full] = np.where(xf_sum > 0.0, xf / xf_sum, 0.0)

    rem_idx = np.flatnonzero(~ok_full)
    if rem_idx.size == 0:
        return out

    y_rem = np.ascontiguousarray(vld_pxls[rem_idx], dtype=np.float64)
    n_rem = y_rem.shape[0]
    diffs = y_rem[:, None, :] - endmembers_arr[None, :, :]
    best_v = np.argmin(np.sum(diffs * diffs, axis=2), axis=1)

    x_mat = np.zeros((n_rem, m), dtype=np.float64)
    p_mask = np.zeros((n_rem, m), dtype=bool)
    x_mat[np.arange(n_rem), best_v] = 1.0
    p_mask[np.arange(n_rem), best_v] = True
    active = np.ones(n_rem, dtype=bool)

    e_norm = max(float(np.linalg.norm(end_t)), 1e-12)
    y_norms = np.maximum(np.linalg.norm(y_rem, axis=1), 1e-12)
    weights = np.int64(1) << np.arange(m, dtype=np.int64)

    for _ in range(max(3 * m, 10)):
        act_idx = np.flatnonzero(active)
        if act_idx.size == 0:
            break
        grad = (x_mat[act_idx] @ end_t.T - y_rem[act_idx]) @ end_t
        p_act = p_mask[act_idx]
        p_counts = np.maximum(np.sum(p_act, axis=1), 1)
        mu = np.sum(grad * p_act, axis=1) / p_counts
        reduced = grad - mu[:, None]
        masked_red = np.where(~p_act, reduced, np.inf)
        best_col = np.argmin(masked_red, axis=1)
        min_red = masked_red[np.arange(act_idx.size), best_col]
        tol_act = 1e-11 * e_norm * np.maximum(e_norm, y_norms[act_idx])
        conv = min_red >= -tol_act
        if np.any(conv):
            active[act_idx[conv]] = False
        need_add = act_idx[~conv]
        if need_add.size == 0:
            break
        p_mask[need_add, best_col[~conv]] = True

        for _ in range(m + 2):
            keys = p_mask[need_add] @ weights
            s_mat = np.zeros((need_add.size, m), dtype=np.float64)
            for key in np.unique(keys):
                if key == 0:
                    continue
                row_sel = np.flatnonzero(keys == key)
                cols = np.flatnonzero((key & weights) != 0)
                s_sub = _solve_simplex_eq_subset(
                    end_t,
                    y_rem[need_add[row_sel]],
                    cols,
                    null_basis=_get_null(int(cols.size))
                    if cols.size > 1
                    else None,
                )
                s_mat[np.ix_(row_sel, cols)] = s_sub

            neg_mask = np.any((s_mat <= -1e-12) & p_mask[need_add], axis=1)
            if np.any(~neg_mask):
                s_ok = np.clip(s_mat[~neg_mask], 0.0, None)
                s_sum = np.sum(s_ok, axis=1, keepdims=True)
                s_ok = np.where(s_sum > 0.0, s_ok / s_sum, 0.0)
                x_mat[need_add[~neg_mask]] = s_ok
            if not np.any(neg_mask):
                break

            fix = need_add[neg_mask]
            s_fix = s_mat[neg_mask]
            x_fix = x_mat[fix]
            valid_step = p_mask[fix] & (s_fix <= 0.0) & (x_fix > s_fix)
            denom = np.where(valid_step, x_fix - s_fix, 1.0)
            ratios = np.where(valid_step, x_fix / denom, np.inf)
            alpha = np.min(ratios, axis=1, keepdims=True)
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            x_new = x_fix + alpha * (s_fix - x_fix)
            x_new[x_new < 1e-12] = 0.0
            row_s = np.sum(x_new, axis=1, keepdims=True)
            x_new = np.where(row_s > 0.0, x_new / row_s, x_fix)
            x_mat[fix] = x_new
            p_mask[fix] &= x_new > 0.0
            no_passive = ~np.any(p_mask[fix], axis=1)
            if np.any(no_passive):
                empty_rows = fix[no_passive]
                p_mask[empty_rows, np.argmax(x_mat[empty_rows], axis=1)] = True
            need_add = fix

    if np.any(active):
        rem_act = np.flatnonzero(active)
        x_mat[rem_act] = _scipy_fcls_heinz_chang_loop(
            y_rem[rem_act], endmembers_arr
        )

    x_mat = np.clip(x_mat, 0.0, None)
    row_sums = np.sum(x_mat, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_mat = np.where(row_sums > 0.0, x_mat / row_sums, 0.0)
    out[rem_idx] = x_mat
    return out


def spec_unmix_nnls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    **[Recommended]** Perform non-negative least-squares (NNLS) linear
    spectral unmixing using a batched active-set solver backed by
    ``scipy.optimize.nnls``.

    .. tip::

       This NumPy/SciPy function is the recommended implementation for NNLS
       unmixing in ``rsgispy``. Unlike ``pysptools.abundance_maps.amaps.NNLS``,
       it solves :math:`\\min_{x \\ge 0} \\| E^T x - y \\|_2` directly
       without squaring the system via normal equations, and requires only
       ``scipy`` (no ``pysptools`` or ``pyMCR``).

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file
                          (if ``None``, all finite positive pixels are
                          unmixed).
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                          pixels (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during
                     processing (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_nnls(
            "img.tif", "unmix_nnls.tif", "endmembers.csv", "mask.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_SCIPY, "spec_unmix_nnls", "scipy", "unmixing")
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_scipy_nnls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def spec_unmix_fcls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    **[Recommended]** Perform fully constrained least-squares (FCLS:
    non-negative and sum-to-one) linear spectral unmixing using a
    batched affine-nullspace active-set solver with ``scipy.optimize.nnls``
    fallback.

    .. tip::

       This NumPy/SciPy function is the recommended implementation for FCLS
       unmixing in ``rsgispy``. It avoids the compiled ``cvxopt`` dependency
       required by :func:`spec_unmix_spts_fcls` while delivering high
       accuracy and substantially faster runtime.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file
                          (if ``None``, all finite positive pixels are
                          unmixed).
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                          pixels (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during
                     processing (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_fcls(
            "img.tif", "unmix_fcls.tif", "endmembers.csv", "mask.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(HAVE_SCIPY, "spec_unmix_fcls", "scipy", "unmixing")
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_scipy_fcls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=None,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def spec_unmix_spts_ucls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform unconstrained least-squares (UCLS) spectral unmixing using
    ``pysptools``.

    .. note::

       Prefer the recommended pure-NumPy :func:`spec_unmix_ucls` function
       for general workflows; this function is retained for ``pysptools``
       compatibility.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file.
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels
                          (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_spts_ucls(
            "img.tif", "unmix_ucls.tif", "endmembers.csv", "mask.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_PYSPTOOLS, "spec_unmix_spts_ucls", "pysptools", "unmixing"
    )
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_spts_amaps.UCLS,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def spec_unmix_spts_nnls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform non-negative least-squares (NNLS) spectral unmixing using
    ``pysptools``.

    .. note::

       Prefer the recommended SciPy :func:`spec_unmix_nnls` function for
       general workflows; ``spec_unmix_nnls`` solves the direct NNLS
       problem without squaring normal equations.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file.
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels
                          (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_spts_nnls(
            "img.tif", "unmix_nnls.tif", "endmembers.csv", "mask.tif", 1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_PYSPTOOLS, "spec_unmix_spts_nnls", "pysptools", "unmixing"
    )
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_spts_amaps.NNLS,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def spec_unmix_spts_fcls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform fully constrained least-squares (FCLS: non-negative and
    sum-to-one) spectral unmixing using ``pysptools`` and ``cvxopt``.

    .. note::

       Prefer the recommended SciPy :func:`spec_unmix_fcls` function
       for general workflows; it does not require ``cvxopt`` or
       ``pysptools`` and runs substantially faster.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file.
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                          pixels (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param file_format: Optional format override when ``endmembers`` is
                        a file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and
                       pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during
                     processing (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_spts_fcls(
            "img.tif",
            "unmix_fcls.tif",
            "endmembers.csv",
            valid_msk_img="mask.tif",
            valid_msk_val=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_PYSPTOOLS, "spec_unmix_spts_fcls", "pysptools", "unmixing"
    )
    rsgispy.check_optional_dep(
        HAVE_CVXOPT, "spec_unmix_spts_fcls", "cvxopt", "unmixing"
    )
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_spts_amaps.FCLS,
        gdalformat=gdalformat,
        gain=gain,
        weight=None,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def _pymcr_nnls_solver(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """Fit PyMCR NNLS model and return abundance matrix."""
    mcrar = McrAR(
        max_iter=100,
        st_regr=_PyMcrNNLS(),
        c_regr=_PyMcrNNLS(),
        c_constraints=[ConstraintNonneg()],
    )
    mcrar.fit(vld_pxls, ST=endmembers_arr)
    return mcrar.C_opt_


def _pymcr_fcls_solver(
    vld_pxls: np.ndarray, endmembers_arr: np.ndarray
) -> np.ndarray:
    """Fit PyMCR FCLS model and return abundance matrix."""
    mcrar = McrAR(
        max_iter=200,
        st_regr=_PyMcrNNLS(),
        c_regr=_PyMcrNNLS(),
        c_constraints=[ConstraintNonneg(), ConstraintNorm()],
    )
    mcrar.fit(vld_pxls, ST=endmembers_arr)
    return mcrar.C_opt_


def spec_unmix_pymcr_nnls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    weight: Optional[float] = None,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform non-negative least-squares (NNLS) spectral unmixing using
    ``pymcr``.

    .. note::

       Prefer the recommended SciPy :func:`spec_unmix_nnls` function for
       general workflows.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file.
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels
                          (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_pymcr_nnls(
            "img.tif",
            "unmix_pymcr_nnls.tif",
            "endmembers.csv",
            valid_msk_img="mask.tif",
            valid_msk_val=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_PYMCR, "spec_unmix_pymcr_nnls", "pyMCR", "unmixing"
    )
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_pymcr_nnls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def spec_unmix_pymcr_fcls(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    gain: float = 1,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform fully constrained least-squares (FCLS: non-negative and
    sum-to-one) spectral unmixing using ``pymcr``.

    .. note::

       Prefer the recommended SciPy :func:`spec_unmix_fcls` function for
       general workflows.

    Pixels are not unmixed, and are written as ``out_no_data_val`` in every
    band, when they are outside ``valid_msk_img``, or have a non-finite value,
    the input image's no-data value or a value ``<= 0`` in any band. Values
    ``<= 0`` are not valid reflectance: they are fill values (0 is a common
    no-data value) or come from noise and atmospheric over-correction of dark
    surfaces (e.g. water in the SWIR), and would give unreliable fractions.
    Clip valid pixels to a small positive value first if they must be unmixed.
    ``out_no_data_val`` is distinct from ``0``, which is a valid fraction.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param valid_msk_img: Optional path to the valid pixel mask image file.
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels
                          (Default = ``1``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param file_format: Optional format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value written to every band for
                            pixels that are not unmixed, and recorded as the
                            output no-data value (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.spec_unmix_pymcr_fcls(
            "img.tif",
            "unmix_pymcr_fcls.tif",
            "endmembers.csv",
            valid_msk_img="mask.tif",
            valid_msk_val=1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(
        HAVE_PYMCR, "spec_unmix_pymcr_fcls", "pyMCR", "unmixing"
    )
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers=endmembers,
        solver_fn=_pymcr_fcls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=None,
        file_format=file_format,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        out_no_data_val=out_no_data_val,
    )


def rescale_unmixing_results(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Clamp negative abundances to zero and normalise each pixel's abundances
    so they sum to ``1.0`` across bands.

    Pixels that are no data in ``input_img`` (its no-data value or a
    non-finite value in any band) and pixels with no positive abundance,
    which cannot be normalised, are written as ``out_no_data_val``.

    :param input_img: Path to the input unmixed abundance image file.
    :param output_img: Path to the output rescaled abundance image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value, also recorded in the output
                            (Default = ``-9999.0``; ``0`` is a valid
                            abundance).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.rescale_unmixing_results("unmix_ucls.tif", "unmix_rescaled.tif")
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    out_nd = _check_out_no_data_val(out_no_data_val)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=src.count,
        )
        dst.nodata = out_nd

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(window=win).astype(np.float64, copy=True)
            flat = np.moveaxis(blk, 0, -1).reshape(-1, src.count)
            vld = _valid_pxls(flat, src.nodatavals)
            flat = np.where(flat < 0.0, 0.0, flat)
            band_sum = np.sum(flat, axis=1)
            vld &= np.isfinite(band_sum) & (band_sum > 0.0)
            out_flat = np.full(flat.shape, out_nd, dtype=np.float32)
            out_flat[vld] = flat[vld] / band_sum[vld, np.newaxis]
            out_blk = np.moveaxis(
                out_flat.reshape((win_h, win_w, src.count)), -1, 0
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def predict_refl_linear_unmixing(
    in_unmix_coefs_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Predict per-band reflectance values from unmixed abundance coefficients and
    an endmembers :class:`pandas.DataFrame` (or file path).

    Pixels that are no data in ``in_unmix_coefs_img`` (its no-data value or a
    non-finite value in any coefficient band, e.g. pixels the unmixing
    skipped) are written as ``out_no_data_val``.

    :param in_unmix_coefs_img: Path to the input abundance coefficients image.
    :param output_img: Path to the output predicted reflectance image.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value, also recorded in the output
                            (Default = ``-9999.0``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.predict_refl_linear_unmixing(
            "unmix.tif", "pred_refl.tif", "endmembers.csv"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_unmix_coefs_img, "Input unmixing coefficients image")
    out_nd = _check_out_no_data_val(out_no_data_val)
    em_df = coerce_endmembers_df(endmembers, file_format=file_format)
    endmembers_arr = em_df.to_numpy(dtype=np.float64).T
    m, n_bands = endmembers_arr.shape
    band_names = [str(idx) for idx in em_df.index]

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(in_unmix_coefs_img, "r"))
        if src.count < m:
            raise rsgispy.RSGISPyException(
                f"Unmixing image '{in_unmix_coefs_img}' has {src.count} bands, "
                f"but endmembers define {m}."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=n_bands,
        )
        dst.nodata = out_nd
        for b_idx, b_name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, b_name)

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(list(range(1, m + 1)), window=win).astype(
                np.float64, copy=False
            )
            coefs_flat = np.moveaxis(blk, 0, -1).reshape(-1, m)
            vld = _valid_pxls(coefs_flat, src.nodatavals)
            pred_flat = np.full(
                (coefs_flat.shape[0], n_bands), out_nd, dtype=np.float32
            )
            pred_flat[vld] = coefs_flat[vld] @ endmembers_arr
            out_blk = np.moveaxis(
                pred_flat.reshape((win_h, win_w, n_bands)), -1, 0
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def calc_unmixing_rmse_residual_err(
    input_img: str,
    input_unmix_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    file_format: Optional[str] = None,
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = -9999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate linear unmixing prediction RMSE, RMSPE, and mean absolute residual
    error against the original reference image.

    Output bands are: Band 1 ``RMSE``, Band 2 ``RMSPE`` (percentage), and
    Band 3 ``Residual``. Pixels that are zero in every band or no data in
    ``input_img``, or no data in ``input_unmix_img`` (e.g. pixels the
    unmixing skipped), are written as ``out_no_data_val``.

    :param input_img: Path to the original reference reflectance image file.
    :param input_unmix_img: Path to the unmixed abundance coefficients image.
    :param output_img: Path to the 3-band output error image file.
    :param endmembers: :class:`pandas.DataFrame` (columns = class names,
                       rows = bands) or file path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param file_format: Optional format override when ``endmembers`` is a file
                        path (Default = ``None``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: Output no-data value, also recorded in the output
                            (Default = ``-9999.0``; ``0`` is a valid error).
    :param calc_stats: If ``True``, assign band names and calculate image
                       statistics and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.unmixing as su

        su.calc_unmixing_rmse_residual_err(
            "s2_refl.tif", "unmix.tif", "unmix_err.tif", "endmembers.csv"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_nd = _check_out_no_data_val(out_no_data_val)
    _, m, n_bands, endmembers_arr = prepare_endmembers_array(
        endmembers, file_format=file_format
    )

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(
            stack, [input_img, input_unmix_img]
        )
        src_orig = datasets[paths[0]]
        src_unmix = datasets[paths[1]]
        grid = compute_spatial_intersection(datasets)

        if src_orig.count != n_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src_orig.count} bands, "
                f"expected {n_bands}."
            )
        if src_unmix.count < m:
            raise rsgispy.RSGISPyException(
                f"Unmixed image '{input_unmix_img}' has {src_unmix.count} bands, "
                f"expected at least {m}."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=3,
        )
        dst.nodata = out_nd

        orig_xoff, orig_yoff = grid.offsets[paths[0]]
        unmix_xoff, unmix_yoff = grid.offsets[paths[1]]

        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(win.height)
            win_w = int(win.width)
            orig_win = Window(
                int(win.col_off) + orig_xoff,
                int(win.row_off) + orig_yoff,
                win_w,
                win_h,
            )
            unmix_win = Window(
                int(win.col_off) + unmix_xoff,
                int(win.row_off) + unmix_yoff,
                win_w,
                win_h,
            )

            orig_blk = src_orig.read(window=orig_win).astype(
                np.float64, copy=False
            )
            unmix_blk = src_unmix.read(
                list(range(1, m + 1)), window=unmix_win
            ).astype(np.float64, copy=False)

            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, n_bands)
            unmix_flat = np.moveaxis(unmix_blk, 0, -1).reshape(-1, m)

            vld = (
                np.logical_not(np.all(orig_flat == 0.0, axis=1))
                & _valid_pxls(orig_flat, src_orig.nodatavals)
                & _valid_pxls(unmix_flat, src_unmix.nodatavals)
            )
            out_flat = np.full((orig_flat.shape[0], 3), out_nd, dtype=np.float32)

            if np.any(vld):
                v_orig = orig_flat[vld]
                v_coef = unmix_flat[vld]
                pred_refl = v_coef @ endmembers_arr
                diff = v_orig - pred_refl

                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)

                rmse = np.sqrt(np.sum(diff * diff, axis=1) / float(n_bands))
                rmspe = (
                    np.sqrt(np.sum(norm_diff * norm_diff, axis=1) / float(n_bands))
                    * 100.0
                )
                residual = np.sum(np.abs(diff), axis=1) / float(n_bands)

                err_stack = np.stack([rmse, rmspe, residual], axis=-1)
                err_stack = np.nan_to_num(
                    err_stack, nan=0.0, posinf=0.0, neginf=0.0
                )
                out_flat[vld] = err_stack.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, 3)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        set_band_names(output_img, ["RMSE", "RMSPE", "Residual"], feedback=False)
        pop_img_stats(
            output_img,
            no_data_val=out_nd,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def summarise_multi_endmember_linear_unmixing(
    input_img: str,
    out_unmix_img: str,
    out_ref_img: str,
    in_unmixed_datasets: Sequence[ImageEndmemberInfo],
    tmp_dir: str = "tmp",
    *,
    gdalformat: str = "GTIFF",
    out_no_data_val: float = 999.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Merge multiple linear spectral unmixing results by selecting the model with
    the lowest Root Mean Square Percentage Error (RMSPE) at each pixel.

    A model is not considered at pixels where its abundances are no data (its
    no-data value or a non-finite value in any band). Pixels with no usable
    model, or that are zero in every band or no data in ``input_img``, are
    written as ``out_no_data_val`` in ``out_unmix_img`` and ``0`` in
    ``out_ref_img``.

    :param input_img: Path to the original input reflectance image file.
    :param out_unmix_img: Path to the combined output abundance image file.
    :param out_ref_img: Path to the 1-based reference index image indicating
                        which candidate dataset achieved the lowest RMSPE per
                        pixel (``0`` for invalid pixels).
    :param in_unmixed_datasets: Sequence of
                                :class:`rsgispy.imagespectral.endmembers.ImageEndmemberInfo`
                                objects describing each candidate unmixing run.
    :param tmp_dir: Retained for API compatibility (Default = ``'tmp'``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param out_no_data_val: No-data value of ``out_unmix_img``
                            (Default = ``999.0``, as in earlier versions; pass
                            ``-9999.0`` to match the unmixing functions).
    :param calc_stats: If ``True``, set band names and compute statistics and
                       pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.endmembers as em
        import rsgispy.imagespectral.unmixing as su

        info_a = em.ImageEndmemberInfo("unmix_a.tif", "em_a.csv")
        info_b = em.ImageEndmemberInfo("unmix_b.tif", "em_b.csv")
        su.summarise_multi_endmember_linear_unmixing(
            "s2_refl.tif",
            "unmix_best.tif",
            "unmix_ref.tif",
            [info_a, info_b],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    _ = tmp_dir
    unmix_nodata = _check_out_no_data_val(out_no_data_val)
    if (
        not isinstance(in_unmixed_datasets, (list, tuple))
        or not in_unmixed_datasets
    ):
        raise rsgispy.RSGISPyException(
            "in_unmixed_datasets must be a non-empty list of ImageEndmemberInfo."
        )

    endmember_names: List[str] = []
    em_infos: List[Tuple[int, int, np.ndarray]] = []
    resolved_ds_names: List[List[str]] = []
    img_paths = [input_img]

    for ds_info in in_unmixed_datasets:
        if not isinstance(ds_info, ImageEndmemberInfo):
            raise rsgispy.RSGISPyException(
                "in_unmixed_datasets must contain ImageEndmemberInfo instances."
            )
        em_names, em_m, em_n, em_arr = prepare_endmembers_array(
            ds_info.endmembers, file_format=ds_info.file_format
        )
        if ds_info.endmember_names is not None:
            if not isinstance(ds_info.endmember_names, (list, tuple)):
                raise rsgispy.RSGISPyException(
                    "unmixed_dataset.endmember_names must be a list or tuple."
                )
            if em_m != len(ds_info.endmember_names):
                raise rsgispy.RSGISPyException(
                    f"Endmember count ({em_m}) does not match "
                    f"endmember_names length ({len(ds_info.endmember_names)})."
                )
            ds_names = [str(name) for name in ds_info.endmember_names]
        else:
            ds_names = em_names

        resolved_ds_names.append(ds_names)
        em_infos.append((em_m, em_n, em_arr))
        img_paths.append(ds_info.in_unmix_img)
        for name in ds_names:
            if name not in endmember_names:
                endmember_names.append(name)

    name_to_band: Dict[str, int] = {
        name: idx for idx, name in enumerate(endmember_names)
    }
    n_out_bands = len(endmember_names)
    n_models = len(in_unmixed_datasets)
    ref_dtype = "uint8" if n_models <= 254 else "uint16"

    with ExitStack() as stack:
        datasets, resolved_paths = open_multi_image_datasets(stack, img_paths)
        src_orig = datasets[resolved_paths[0]]
        for ds_idx, ds_info in enumerate(in_unmixed_datasets):
            u_ds = datasets[resolved_paths[ds_idx + 1]]
            em_m, em_n, _ = em_infos[ds_idx]
            ds_names = resolved_ds_names[ds_idx]
            if u_ds.count != len(ds_names):
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmember names ({len(ds_names)})."
                )
            if em_m != u_ds.count:
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmember count ({em_m})."
                )
            if em_n != src_orig.count:
                raise rsgispy.RSGISPyException(
                    f"Input image '{input_img}' band count ({src_orig.count}) "
                    f"does not match endmember band count ({em_n})."
                )

        grid = compute_spatial_intersection(datasets)
        dst_unmix = open_output_dataset(
            stack=stack,
            output_img=out_unmix_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=n_out_bands,
        )
        dst_unmix.nodata = unmix_nodata
        for b_idx, b_name in enumerate(endmember_names, start=1):
            dst_unmix.set_band_description(b_idx, b_name)

        dst_ref = open_output_dataset(
            stack=stack,
            output_img=out_ref_img,
            gdalformat=gdalformat,
            out_rio_dtype=ref_dtype,
            grid=grid,
            count=1,
        )
        dst_ref.nodata = 0

        orig_xoff, orig_yoff = grid.offsets[resolved_paths[0]]
        model_band_maps = [
            [name_to_band[name] for name in ds_names]
            for ds_names in resolved_ds_names
        ]

        for win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(win.height)
            win_w = int(win.width)
            n_pxls = win_h * win_w

            orig_win = Window(
                int(win.col_off) + orig_xoff,
                int(win.row_off) + orig_yoff,
                win_w,
                win_h,
            )
            orig_blk = src_orig.read(window=orig_win).astype(
                np.float64, copy=False
            )
            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, src_orig.count)
            vld = np.logical_not(np.all(orig_flat == 0.0, axis=1)) & _valid_pxls(
                orig_flat, src_orig.nodatavals
            )

            best_rmspe = np.full(n_pxls, np.inf, dtype=np.float64)
            best_ref = np.zeros(n_pxls, dtype=np.int32)
            best_abund = np.full(
                (n_pxls, n_out_bands), unmix_nodata, dtype=np.float32
            )

            if np.any(vld):
                v_orig = orig_flat[vld]
                for ds_idx in range(n_models):
                    u_path = resolved_paths[ds_idx + 1]
                    u_ds = datasets[u_path]
                    u_xoff, u_yoff = grid.offsets[u_path]
                    u_win = Window(
                        int(win.col_off) + u_xoff,
                        int(win.row_off) + u_yoff,
                        win_w,
                        win_h,
                    )
                    u_blk = u_ds.read(window=u_win).astype(
                        np.float64, copy=False
                    )
                    u_flat = np.moveaxis(u_blk, 0, -1).reshape(-1, u_ds.count)
                    v_coef = u_flat[vld]

                    _, n_bands_m, em_arr = em_infos[ds_idx]
                    pred_refl = v_coef @ em_arr
                    diff = v_orig - pred_refl
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)
                    rmspe = (
                        np.sqrt(
                            np.sum(norm_diff * norm_diff, axis=1)
                            / float(n_bands_m)
                        )
                        * 100.0
                    )
                    coef_ok = _valid_pxls(v_coef, u_ds.nodatavals)
                    rmspe = np.where(
                        coef_ok & np.isfinite(rmspe), rmspe, np.inf
                    )

                    v_idx = np.flatnonzero(vld)
                    better = rmspe < best_rmspe[v_idx]
                    if np.any(better):
                        win_pxls = v_idx[better]
                        best_rmspe[win_pxls] = rmspe[better]
                        best_ref[win_pxls] = ds_idx + 1
                        best_abund[win_pxls, :] = 0.0
                        for local_b, global_b in enumerate(
                            model_band_maps[ds_idx]
                        ):
                            best_abund[win_pxls, global_b] = v_coef[
                                better, local_b
                            ]

            out_unmix_blk = np.moveaxis(
                best_abund.reshape((win_h, win_w, n_out_bands)), -1, 0
            )
            dst_unmix.write(out_unmix_blk, window=win)
            dst_ref.write(
                best_ref.astype(ref_dtype).reshape((1, win_h, win_w)), window=win
            )

    if calc_stats:
        set_band_names(out_unmix_img, endmember_names, feedback=False)
        pop_img_stats(
            out_unmix_img,
            no_data_val=unmix_nodata,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
        pop_thmt_img_stats(
            out_ref_img,
            add_clr_tab=True,
            calc_pyramids=True,
            ignore_zero=True,
            use_pbar=use_pbar,
        )
