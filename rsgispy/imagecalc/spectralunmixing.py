# pylint: disable=too-many-lines
"""
The imagecalc.spectralunmixing module provides linear spectral unmixing,
endmember extraction/comparison, pixel purity index (PPI), error evaluation,
and multi-endmember unmixing summary utilities using ``rasterio`` and ``numpy``.

**Recommended unmixing functions:**

* :func:`spec_unmix_ucls` — vectorised unconstrained least squares (core
  ``numpy`` only)
* :func:`spec_unmix_nnls` — direct Lawson–Hanson non-negative least squares
  (requires ``scipy``)
* :func:`spec_unmix_fcls` — scale-invariant fully constrained least squares
  (non-negative and sum-to-one, requires ``scipy``)

Functions wrapping ``pysptools``, ``cvxopt``, and ``pymcr`` are also retained
and exported when those optional dependencies are installed.
"""

from contextlib import ExitStack
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas
import numpy as np
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import rasterio.windows

import rsgispy
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.metadata import set_band_names
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools.geometry import do_bboxes_intersect, reproj_bbox
from rsgispy.vectorprocess import get_gdf_bbox, read_vec_gdf, resolve_vector_crs

try:
    import scipy.optimize

    HAVE_SCIPY = True
except ImportError:
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
    HAVE_PYMCR = False

__all__: List[str] = [
    "ImageEndmemberInfo",
    "extract_avg_endmembers",
    "read_endmembers_mtxt",
    "write_endmembers_mtxt",
    "plot_endmembers",
    "are_endmembers_equal",
    "spec_unmix_ucls",
    "exhcon_linear_spec_unmix",
    "rescale_unmixing_results",
    "predict_refl_linear_unmixing",
    "calc_unmixing_rmse_residual_err",
    "summarise_multi_endmember_linear_unmixing",
    "calc_ppi",
]

if HAVE_SCIPY:
    __all__.extend(["spec_unmix_nnls", "spec_unmix_fcls"])

if HAVE_PYSPTOOLS:
    __all__.extend(["spec_unmix_spts_ucls", "spec_unmix_spts_nnls"])
    if HAVE_CVXOPT:
        __all__.append("spec_unmix_spts_fcls")

if HAVE_PYMCR:
    __all__.extend(["spec_unmix_pymcr_nnls", "spec_unmix_pymcr_fcls"])


class ImageEndmemberInfo:
    """
    Container associating an unmixed abundance image with the endmember matrix
    file and ordered list of endmember names used to produce it.

    :param in_unmix_img: Path to the linear unmixed abundance image file.
    :param endmembers_file: Path to the ``.mtxt`` endmembers file.
    :param endmember_names: Ordered sequence of endmember names matching the
                            bands of ``in_unmix_img``.

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        info = su.ImageEndmemberInfo(
            in_unmix_img="unmix.tif",
            endmembers_file="endmembers.mtxt",
            endmember_names=["Soil", "Veg", "Water"],
        )
    """

    def __init__(
        self,
        in_unmix_img: str,
        endmembers_file: str,
        endmember_names: List[str],
    ) -> None:
        self.in_unmix_img = in_unmix_img
        self.endmembers_file = endmembers_file
        self.endmember_names = endmember_names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageEndmemberInfo):
            return NotImplemented
        return (
            self.in_unmix_img == other.in_unmix_img
            and self.endmembers_file == other.endmembers_file
            and self.endmember_names == other.endmember_names
        )

    def __repr__(self) -> str:
        return (
            f"ImageEndmemberInfo(in_unmix_img={self.in_unmix_img!r}, "
            f"endmembers_file={self.endmembers_file!r}, "
            f"endmember_names={self.endmember_names!r})"
        )


def read_endmembers_mtxt(
    endmembers_file: str,
    gain: float = 1,
    weight: Optional[float] = None,
) -> Tuple[int, int, np.ndarray]:
    """
    Read an endmembers matrix file in RSGISLib ``.mtxt`` format.

    :param endmembers_file: Path to the input ``.mtxt`` file.
    :param gain: Optional gain divisor applied to endmember spectra
                 (``endmembers / gain`` when ``gain > 1``, Default = ``1``).
    :param weight: Optional constraint weight appended as an extra column to
                   enforce sum-to-one optimisation (Scarth et al., 2010).
    :return: Tuple ``(m, n, end_mem_arr)`` where ``m`` is the number of
             endmembers, ``n`` is the number of spectral bands (plus 1 if
             ``weight`` is provided), and ``end_mem_arr`` has shape ``(m, n)``.

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        m, n, em_arr = su.read_endmembers_mtxt("endmembers.mtxt")
    """
    ensure_file_exists(endmembers_file, "Endmembers file")
    with open(endmembers_file, "r", encoding="utf-8") as handle:
        data_lst = [line.strip() for line in handle if line.strip() != ""]

    if len(data_lst) < 3:
        raise rsgispy.RSGISPyException(
            f"Invalid endmembers file format in '{endmembers_file}'."
        )

    m = int(data_lst[0].split("=")[-1])
    n = int(data_lst[1].split("=")[-1])
    raw_vals = [val.strip() for val in data_lst[2].split(",") if val.strip() != ""]
    if len(raw_vals) != m * n:
        raise rsgispy.RSGISPyException(
            f"Expected {m * n} values in '{endmembers_file}', found {len(raw_vals)}."
        )

    end_mem_arr = np.zeros((m, n), dtype=np.float64)
    idx = 0
    for n_idx in range(n):
        for m_idx in range(m):
            end_mem_arr[m_idx, n_idx] = float(raw_vals[idx])
            idx += 1

    if gain > 1:
        end_mem_arr = end_mem_arr / float(gain)

    if weight is not None:
        weights = np.full((m, 1), float(weight), dtype=np.float64)
        end_mem_arr = np.hstack([end_mem_arr, weights])
        n += 1

    return m, n, end_mem_arr


def write_endmembers_mtxt(
    endmembers_np_arr: np.ndarray,
    endmembers_file: str,
) -> None:
    """
    Write an endmembers NumPy array of shape ``(m_endmembers, n_bands)`` to an
    RSGISLib ``.mtxt`` file.

    :param endmembers_np_arr: 2D NumPy array ``(m_endmembers, n_bands)``.
    :param endmembers_file: Output file path for the ``.mtxt`` matrix file.

    .. code:: python

        import numpy as np
        import rsgispy.imagecalc.spectralunmixing as su

        em = np.array([[0.2, 0.4, 0.8], [0.6, 0.2, 0.1]])
        su.write_endmembers_mtxt(em, "endmembers.mtxt")
    """
    arr = np.asarray(endmembers_np_arr, dtype=np.float64)
    if arr.ndim != 2:
        raise rsgispy.RSGISPyException(
            "endmembers_np_arr must be a 2D array with shape (m, n)."
        )
    out_m, out_n = arr.shape
    vals = [
        f"{float(arr[m_idx, n_idx])}"
        for n_idx in range(out_n)
        for m_idx in range(out_m)
    ]
    out_lines = [f"m={out_m}", f"n={out_n}", ",".join(vals)]

    out_dir = os.path.dirname(os.path.abspath(endmembers_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(endmembers_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out_lines) + "\n")


def plot_endmembers(
    endmembers_file: str,
    endmember_names: Sequence[str],
    out_plot_file: str,
    plot_title: Optional[str] = None,
    gain: float = 1,
    wavelengths: Optional[Sequence[float]] = None,
) -> None:
    """
    Plot spectral endmember signatures from an ``.mtxt`` file and save to disk.

    :param endmembers_file: Path to the input ``.mtxt`` endmembers file.
    :param endmember_names: Names of the endmembers in the same order as
                            ``endmembers_file``.
    :param out_plot_file: Output image path for the saved figure.
    :param plot_title: Optional plot title string.
    :param gain: Optional gain divisor applied to endmember spectra
                 (Default = ``1``).
    :param wavelengths: Optional sequence of band wavelengths (band indices used
                        when ``None``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.plot_endmembers(
            "endmembers.mtxt",
            ["Soil", "Veg"],
            "endmembers.png",
            plot_title="Endmember Signatures",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    n_endmembers, n_bands, endmembers = read_endmembers_mtxt(
        endmembers_file, gain=gain
    )

    if len(endmember_names) != n_endmembers:
        raise rsgispy.RSGISPyException(
            "The list of names provided does not match the number of "
            "endmembers in the input file."
        )

    x_axis_lbl = "Wavelength"
    if wavelengths is not None:
        if not isinstance(wavelengths, (list, tuple, np.ndarray)):
            raise rsgispy.RSGISPyException(
                "If provided the wavelengths variable must be a sequence."
            )
        if len(wavelengths) != n_bands:
            raise rsgispy.RSGISPyException(
                "The number of wavelengths provided is not equal to the "
                "number of image bands in the endmembers."
            )
        x_vals: Sequence[float] = list(wavelengths)
    else:
        x_vals = list(range(n_bands))
        x_axis_lbl = "Band"

    fig = plt.figure(figsize=(7, 5), dpi=300)
    try:
        ax1 = fig.add_subplot(111)
        for i in range(n_endmembers):
            ax1.plot(x_vals, endmembers[i], label=endmember_names[i])

        if plot_title is not None:
            ax1.set_title(plot_title)
        ax1.set_xlabel(x_axis_lbl)
        ax1.set_ylabel("Reflectance (%)")
        ax1.legend()
        fig.savefig(out_plot_file)
    finally:
        plt.close(fig)


def are_endmembers_equal(
    ref_endmember_file: str,
    cmp_endmember_file: str,
    flt_dif: float = 0.0001,
) -> bool:
    """
    Compare two ``.mtxt`` endmember files for element-wise equality within
    tolerance ``flt_dif``.

    :param ref_endmember_file: Reference ``.mtxt`` file path.
    :param cmp_endmember_file: Comparison ``.mtxt`` file path.
    :param flt_dif: Maximum allowed absolute floating-point difference
                    (Default = ``0.0001``).
    :return: ``True`` if dimensions match and all entries differ by at most
             ``flt_dif``, ``False`` otherwise.

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        same = su.are_endmembers_equal("em_ref.mtxt", "em_cmp.mtxt")
    """
    n_ref_m, n_ref_n, ref_em = read_endmembers_mtxt(ref_endmember_file)
    n_cmp_m, n_cmp_n, cmp_em = read_endmembers_mtxt(cmp_endmember_file)

    if n_ref_m != n_cmp_m or n_ref_n != n_cmp_n:
        return False

    return bool(np.all(np.abs(ref_em - cmp_em) <= flt_dif))


def _feature_window(
    geom_bounds: Tuple[float, float, float, float],
    transform: rasterio.Affine,
    width: int,
    height: int,
) -> Window:
    """Compute a clipped integer pixel Window covering ``geom_bounds``."""
    min_x, min_y, max_x, max_y = geom_bounds
    raw = rasterio.windows.from_bounds(min_x, min_y, max_x, max_y, transform=transform)
    c0 = max(0, int(math.floor(raw.col_off)))
    r0 = max(0, int(math.floor(raw.row_off)))
    c1 = min(width, int(math.ceil(raw.col_off + raw.width)))
    r1 = min(height, int(math.ceil(raw.row_off + raw.height)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def extract_avg_endmembers(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    out_file: str,
    pxl_in_poly_method: int = 1,
) -> None:
    """
    Extract the mean spectral signature per polygon in ``vec_file`` and save the
    resulting endmembers matrix in RSGISLib ``.mtxt`` format.

    Each polygon feature becomes one endmember row. If ``out_file`` does not
    end with ``'.mtxt'``, the extension is appended automatically.

    :param input_img: Path to the input multi-band raster image file.
    :param vec_file: Path to the input vector polygon file.
    :param vec_lyr: Name of the vector layer (or ``None`` for default layer).
    :param out_file: Output file path (with or without ``.mtxt`` extension).
    :param pxl_in_poly_method: Pixel-in-polygon selection constant from
                               :mod:`rsgispy.zonalstats` (``METHOD_*``).
                               Default = ``1`` (pixel centre within polygon).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su
        import rsgispy.zonalstats as zs

        su.extract_avg_endmembers(
            "image.tif",
            "rois.gpkg",
            "rois",
            "endmembers",
            pxl_in_poly_method=zs.METHOD_POLYCONTAINSPIXELCENTER,
        )
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ensure_file_exists(input_img, "Input image")
    ensure_file_exists(vec_file, "Input vector")

    out_mtxt = out_file if out_file.endswith(".mtxt") else f"{out_file}.mtxt"
    gdf = read_vec_gdf(vec_file, vec_lyr)
    if gdf.empty:
        raise rsgispy.RSGISPyException(
            f"Vector layer '{vec_lyr or vec_file}' contains no features."
        )

    all_touched = pxl_in_poly_method in (2, 3, 7, 8, 9)

    with rasterio.open(input_img, "r") as src:
        img_crs = pyproj.CRS.from_user_input(src.crs) if src.crs is not None else None
        geom_series: geopandas.GeoSeries = gdf.geometry
        if gdf.crs is not None and img_crs is not None:
            vec_crs = resolve_vector_crs(gdf)
            if vec_crs != img_crs:
                geom_series = geom_series.to_crs(img_crs)
                vec_bbox = reproj_bbox(get_gdf_bbox(gdf), vec_crs, img_crs)
            else:
                vec_bbox = get_gdf_bbox(gdf)
        else:
            vec_bbox = get_gdf_bbox(gdf)

        bounds = src.bounds
        img_bbox = (
            float(bounds.left),
            float(bounds.right),
            float(bounds.bottom),
            float(bounds.top),
        )
        if not do_bboxes_intersect(img_bbox, vec_bbox):
            raise rsgispy.RSGISPyException("The vector layer and image do not overlap.")

        endmembers = np.zeros((len(geom_series), src.count), dtype=np.float64)
        nodata_val = src.nodata

        for idx, geom in enumerate(geom_series):
            if geom is None or geom.is_empty:
                raise rsgispy.RSGISPyException(
                    f"Polygon feature at index {idx} is empty or null."
                )

            win = _feature_window(geom.bounds, src.transform, src.width, src.height)
            if win.width <= 0 or win.height <= 0:
                raise rsgispy.RSGISPyException(
                    f"Polygon feature at index {idx} does not intersect any raster pixels."
                )

            arr = src.read(window=win).astype(np.float64, copy=False)
            sub_transform = rasterio.windows.transform(win, src.transform)
            rv = rasterize(
                [(geom, 1)],
                out_shape=(int(win.height), int(win.width)),
                transform=sub_transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )
            mask = rv == 1
            if not np.any(mask) and pxl_in_poly_method == 6:
                rv = rasterize(
                    [(geom, 1)],
                    out_shape=(int(win.height), int(win.width)),
                    transform=sub_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=True,
                )
                mask = rv == 1

            pxls = arr[:, mask].T  # (n_pxls, n_bands)
            vld = np.all(np.isfinite(pxls), axis=1)
            if nodata_val is not None and not (
                isinstance(nodata_val, float) and math.isnan(nodata_val)
            ):
                vld &= np.all(pxls != nodata_val, axis=1)
            pxls = pxls[vld]

            if pxls.shape[0] == 0:
                if pxl_in_poly_method in (1, 5, 6):
                    centroid = geom.centroid
                    row, col = src.index(centroid.x, centroid.y)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        cent_win = Window(int(col), int(row), 1, 1)
                        c_arr = src.read(window=cent_win).astype(np.float64).ravel()
                        if np.all(np.isfinite(c_arr)):
                            endmembers[idx, :] = c_arr
                            continue
                raise rsgispy.RSGISPyException(
                    f"No valid pixels fell within polygon feature at index {idx}."
                )

            endmembers[idx, :] = np.mean(pxls, axis=0)

    write_endmembers_mtxt(endmembers, out_mtxt)


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
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    endmember_file: Optional[str] = None,
    step_res: float = 0.1,
    gain: float = 1.0,
    offset: float = 0.0,
    use_pbar: bool = True,
) -> None:
    """
    Perform exhaustive constrained linear spectral unmixing of ``input_img``.

    Both pixel spectra and endmember spectra are L2-normalised prior to
    searching all non-negative abundance combinations summing to 1.0 at grid
    step ``step_res``. The output raster contains ``m + 1`` bands: scaled
    abundances for each of the ``m`` endmembers followed by the scaled RMSE
    residual in band ``m + 1`` (computed as ``offset + gain * value``;
    zero-norm / invalid pixels are written as ``0``).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output unmixed raster image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param endmember_file: Path to the ``.mtxt`` endmembers matrix file.
    :param step_res: Search step resolution in ``(0, 1]`` (Default = ``0.1``).
    :param gain: Multiplicative gain applied to output values
                 (Default = ``1.0``).
    :param offset: Additive offset applied to output values (Default = ``0.0``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.spectralunmixing as su

        su.exhcon_linear_spec_unmix(
            input_img="s2_refl.tif",
            output_img="unmixed.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            endmember_file="endmembers.mtxt",
            step_res=0.1,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    if endmember_file is None:
        raise rsgispy.RSGISPyException("endmember_file must be provided.")

    m, n_bands, em_arr = read_endmembers_mtxt(endmember_file)
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
    n_out_bands = m + 1
    chunk_size = 8192

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.count != n_bands:
            raise rsgispy.RSGISPyException(
                f"Input image '{input_img}' has {src.count} bands, but "
                f"endmember file '{endmember_file}' expects {n_bands} bands."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=n_out_bands,
        )

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(window=win).astype(np.float64, copy=False)
            flat = np.moveaxis(blk, 0, -1).reshape(-1, n_bands)

            norms = np.linalg.norm(flat, axis=1)
            valid = np.all(np.isfinite(flat), axis=1) & (norms > 0.0)
            out_flat = np.zeros((flat.shape[0], n_out_bands), dtype=np.float64)

            if np.any(valid):
                v_idx = np.flatnonzero(valid)
                pxl_normed = flat[v_idx] / norms[v_idx, np.newaxis]
                pxl_sq = np.sum(pxl_normed * pxl_normed, axis=1)

                n_valid = v_idx.size
                for start in range(0, n_valid, chunk_size):
                    end = min(start + chunk_size, n_valid)
                    sub_p = pxl_normed[start:end]
                    sub_sq = pxl_sq[start:end, np.newaxis] + cand_sq_norms[np.newaxis, :]
                    sq_dists = sub_sq - 2.0 * (sub_p @ cand_spectra_t)
                    best_c = np.argmin(sq_dists, axis=1)
                    min_sq = np.maximum(0.0, sq_dists[np.arange(end - start), best_c])
                    rmse = np.sqrt(min_sq / float(n_bands))

                    out_idx = v_idx[start:end]
                    out_flat[out_idx, :m] = offset + (gain * cand_weights[best_c])
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
    endmembers_file: str,
    solver_fn,
    gdalformat: str = "GTIFF",
    gain: float = 1,
    weight: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Shared windowed linear spectral unmixing runner."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    m, _, endmembers_arr = read_endmembers_mtxt(endmembers_file, gain, weight)

    in_paths = [input_img] if valid_msk_img is None else [input_img, valid_msk_img]
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, in_paths)
        src_img = datasets[paths[0]]
        src_msk = datasets[paths[1]] if valid_msk_img is not None else None
        grid = compute_spatial_intersection(datasets)

        expected_bands = endmembers_arr.shape[1] - (1 if weight is not None else 0)
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

        img_xoff, img_yoff = grid.offsets[paths[0]]
        msk_xoff, msk_yoff = (
            grid.offsets[paths[1]] if valid_msk_img is not None else (0, 0)
        )

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            img_win = Window(
                int(win.col_off) + img_xoff,
                int(win.row_off) + img_yoff,
                win_w,
                win_h,
            )

            out_flat = np.zeros((win_h * win_w, m), dtype=np.float32)
            if src_msk is not None:
                msk_win = Window(
                    int(win.col_off) + msk_xoff,
                    int(win.row_off) + msk_yoff,
                    win_w,
                    win_h,
                )
                msk_vld = src_msk.read(1, window=msk_win).ravel() == valid_msk_val
            else:
                msk_vld = np.ones(win_h * win_w, dtype=bool)

            if np.any(msk_vld):
                img_blk = src_img.read(window=img_win).astype(np.float64, copy=False)
                img_flat = np.moveaxis(img_blk, 0, -1).reshape(-1, src_img.count)

                vld = (
                    msk_vld
                    & np.all(np.isfinite(img_flat), axis=1)
                    & np.all(img_flat > 0.0, axis=1)
                )
                if np.any(vld):
                    vld_pxls = img_flat[vld]
                    if gain > 1:
                        vld_pxls = vld_pxls / float(gain)
                    if weight is not None:
                        w_col = np.full(
                            (vld_pxls.shape[0], 1), float(weight), dtype=np.float64
                        )
                        vld_pxls = np.hstack([vld_pxls, w_col])

                    abundances = solver_fn(vld_pxls, endmembers_arr)
                    out_flat[vld] = abundances.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, m)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _numpy_ucls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
    """Vectorised unconstrained least squares solver using ``numpy.linalg.lstsq``."""
    abundances, _, _, _ = np.linalg.lstsq(
        endmembers_arr.T, vld_pxls.T, rcond=None
    )
    return abundances.T


def spec_unmix_ucls(
    input_img: str,
    valid_msk_img: Optional[str],
    valid_msk_val: int,
    output_img: str,
    endmembers_file: str,
    gdalformat: str = "GTIFF",
    gain: float = 1,
    weight: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Perform unconstrained least-squares (UCLS) linear spectral
    unmixing using vectorised ``numpy.linalg.lstsq``.

    Requires only core ``numpy`` (no optional unmixing packages needed).

    :param input_img: Path to the input image file.
    :param valid_msk_img: Optional path to the valid pixel mask image file
                          (if ``None``, all finite positive pixels are unmixed).
    :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
    :param output_img: Path to the output unmixed abundance image file.
    :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param gain: Gain divisor applied to image and endmember spectra when
                 greater than ``1`` (Default = ``1``).
    :param weight: Optional Scarth et al. (2010) sum-to-one weight.
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.spec_unmix_ucls(
            "img.tif", "mask.tif", 1, "unmix_ucls.tif", "endmembers.mtxt"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_windowed_unmixer(
        input_img=input_img,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        output_img=output_img,
        endmembers_file=endmembers_file,
        solver_fn=_numpy_ucls_solver,
        gdalformat=gdalformat,
        gain=gain,
        weight=weight,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


if HAVE_SCIPY:

    def _scipy_nnls_solver(
        vld_pxls: np.ndarray, endmembers_arr: np.ndarray
    ) -> np.ndarray:
        """Direct Lawson–Hanson NNLS solver using ``scipy.optimize.nnls``."""
        end_t = endmembers_arr.T
        n_pxls = vld_pxls.shape[0]
        m = endmembers_arr.shape[0]
        out = np.empty((n_pxls, m), dtype=np.float64)
        for idx in range(n_pxls):
            out[idx] = scipy.optimize.nnls(end_t, vld_pxls[idx])[0]
        return out

    def _scipy_fcls_solver(
        vld_pxls: np.ndarray, endmembers_arr: np.ndarray
    ) -> np.ndarray:
        """
        Scale-invariant Heinz & Chang (2001) FCLS solver using
        ``scipy.optimize.nnls``.
        """
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

    def spec_unmix_nnls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        **[Recommended]** Perform non-negative least-squares (NNLS) linear
        spectral unmixing using ``scipy.optimize.nnls``.

        .. tip::

           This NumPy/SciPy function is the recommended implementation for NNLS
           unmixing in ``rsgispy``. Unlike ``pysptools.abundance_maps.amaps.NNLS``,
           it solves :math:`\\min_{x \\ge 0} \\| E^T x - y \\|_2` directly
           without squaring the system via normal equations, and requires only
           ``scipy`` (no ``pysptools`` or ``pyMCR``).

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file
                              (if ``None``, all finite positive pixels are
                              unmixed).
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                              pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during
                         processing (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_nnls(
                "img.tif", "mask.tif", 1, "unmix_nnls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_scipy_nnls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_fcls(
        input_img: str,
        valid_msk_img: Optional[str],
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        **[Recommended]** Perform fully constrained least-squares (FCLS:
        non-negative and sum-to-one) linear spectral unmixing using a
        scale-invariant Heinz & Chang (2001) formulation via
        ``scipy.optimize.nnls``.

        .. tip::

           This NumPy/SciPy function is the recommended implementation for FCLS
           unmixing in ``rsgispy``. It avoids the compiled ``cvxopt`` dependency
           required by :func:`spec_unmix_spts_fcls` while delivering high
           accuracy and substantially faster runtime.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Optional path to the valid pixel mask image file
                              (if ``None``, all finite positive pixels are
                              unmixed).
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid
                              pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during
                         processing (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_fcls(
                "img.tif", "mask.tif", 1, "unmix_fcls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_scipy_fcls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=None,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


if HAVE_PYSPTOOLS:

    def spec_unmix_spts_ucls(
        input_img: str,
        valid_msk_img: str,
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
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

        :param input_img: Path to the input image file.
        :param valid_msk_img: Path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_spts_ucls(
                "img.tif", "mask.tif", 1, "unmix_ucls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_spts_amaps.UCLS,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_spts_nnls(
        input_img: str,
        valid_msk_img: str,
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
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

        :param input_img: Path to the input image file.
        :param valid_msk_img: Path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_spts_nnls(
                "img.tif", "mask.tif", 1, "unmix_nnls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_spts_amaps.NNLS,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    if HAVE_CVXOPT:

        def spec_unmix_spts_fcls(
            input_img: str,
            valid_msk_img: str,
            valid_msk_val: int,
            output_img: str,
            endmembers_file: str,
            gdalformat: str = "GTIFF",
            gain: float = 1,
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

            :param input_img: Path to the input image file.
            :param valid_msk_img: Path to the valid pixel mask image file.
            :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
            :param output_img: Path to the output unmixed abundance image file.
            :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
            :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
            :param gain: Gain divisor applied to image and endmember spectra when
                         greater than ``1`` (Default = ``1``).
            :param calc_stats: If ``True``, populate image statistics and pyramids
                               (Default = ``True``).
            :param use_pbar: If ``True``, display a tqdm progress bar during processing
                             (Default = ``True``).

            .. code:: python

                import rsgispy.imagecalc.spectralunmixing as su

                su.spec_unmix_spts_fcls(
                    "img.tif", "mask.tif", 1, "unmix_fcls.tif", "endmembers.mtxt"
                )
            """
            # pylint: disable=too-many-arguments,too-many-positional-arguments
            _run_windowed_unmixer(
                input_img=input_img,
                valid_msk_img=valid_msk_img,
                valid_msk_val=valid_msk_val,
                output_img=output_img,
                endmembers_file=endmembers_file,
                solver_fn=_spts_amaps.FCLS,
                gdalformat=gdalformat,
                gain=gain,
                weight=None,
                calc_stats=calc_stats,
                use_pbar=use_pbar,
            )


if HAVE_PYMCR:

    def _pymcr_nnls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
        """Fit PyMCR NNLS model and return abundance matrix."""
        mcrar = McrAR(
            max_iter=100,
            st_regr=_PyMcrNNLS(),
            c_regr=_PyMcrNNLS(),
            c_constraints=[ConstraintNonneg()],
        )
        mcrar.fit(vld_pxls, ST=endmembers_arr)
        return mcrar.C_opt_

    def _pymcr_fcls_solver(vld_pxls: np.ndarray, endmembers_arr: np.ndarray) -> np.ndarray:
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
        valid_msk_img: str,
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        weight: Optional[float] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform non-negative least-squares (NNLS) spectral unmixing using
        ``pymcr``.

        .. note::

           Prefer the recommended SciPy :func:`spec_unmix_nnls` function for
           general workflows.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param weight: Optional Scarth et al. (2010) sum-to-one weight.
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_pymcr_nnls(
                "img.tif", "mask.tif", 1, "unmix_pymcr_nnls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_pymcr_nnls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=weight,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )

    def spec_unmix_pymcr_fcls(
        input_img: str,
        valid_msk_img: str,
        valid_msk_val: int,
        output_img: str,
        endmembers_file: str,
        gdalformat: str = "GTIFF",
        gain: float = 1,
        calc_stats: bool = True,
        use_pbar: bool = True,
    ) -> None:
        """
        Perform fully constrained least-squares (FCLS: non-negative and
        sum-to-one) spectral unmixing using ``pymcr``.

        .. note::

           Prefer the recommended SciPy :func:`spec_unmix_fcls` function for
           general workflows.

        :param input_img: Path to the input image file.
        :param valid_msk_img: Path to the valid pixel mask image file.
        :param valid_msk_val: Pixel value in ``valid_msk_img`` marking valid pixels.
        :param output_img: Path to the output unmixed abundance image file.
        :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
        :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
        :param gain: Gain divisor applied to image and endmember spectra when
                     greater than ``1`` (Default = ``1``).
        :param calc_stats: If ``True``, populate image statistics and pyramids
                           (Default = ``True``).
        :param use_pbar: If ``True``, display a tqdm progress bar during processing
                         (Default = ``True``).

        .. code:: python

            import rsgispy.imagecalc.spectralunmixing as su

            su.spec_unmix_pymcr_fcls(
                "img.tif", "mask.tif", 1, "unmix_pymcr_fcls.tif", "endmembers.mtxt"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        _run_windowed_unmixer(
            input_img=input_img,
            valid_msk_img=valid_msk_img,
            valid_msk_val=valid_msk_val,
            output_img=output_img,
            endmembers_file=endmembers_file,
            solver_fn=_pymcr_fcls_solver,
            gdalformat=gdalformat,
            gain=gain,
            weight=None,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


def rescale_unmixing_results(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Clamp negative abundances to zero and normalise each pixel's abundances
    so they sum to ``1.0`` across bands.

    :param input_img: Path to the input unmixed abundance image file.
    :param output_img: Path to the output rescaled abundance image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.rescale_unmixing_results("unmix_ucls.tif", "unmix_rescaled.tif")
    """
    ensure_file_exists(input_img, "Input image")

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

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            blk = src.read(window=win).astype(np.float32, copy=True)
            blk[blk < 0.0] = 0.0
            band_sum = np.sum(blk, axis=0, keepdims=True)
            out_blk = np.zeros_like(blk, dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(blk, band_sum, out=out_blk, where=band_sum > 0.0)
            out_blk = np.where(np.isfinite(out_blk), out_blk, 0.0).astype(
                np.float32, copy=False
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def predict_refl_linear_unmixing(
    in_unmix_coefs_img: str,
    endmembers_file: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Predict per-band reflectance values from unmixed abundance coefficients and
    an ``.mtxt`` endmembers matrix.

    :param in_unmix_coefs_img: Path to the input abundance coefficients image.
    :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
    :param output_img: Path to the output predicted reflectance image.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.predict_refl_linear_unmixing(
            "unmix.tif", "endmembers.mtxt", "pred_refl.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(in_unmix_coefs_img, "Input unmixing coefficients image")
    m, n_bands, endmembers_arr = read_endmembers_mtxt(endmembers_file)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(in_unmix_coefs_img, "r"))
        if src.count < m:
            raise rsgispy.RSGISPyException(
                f"Unmixing image '{in_unmix_coefs_img}' has {src.count} bands, "
                f"but endmembers file '{endmembers_file}' defines {m}."
            )

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            count=n_bands,
        )

        for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            blk = src.read(list(range(1, m + 1)), window=win).astype(
                np.float64, copy=False
            )
            coefs_flat = np.moveaxis(blk, 0, -1).reshape(-1, m)
            pred_flat = (coefs_flat @ endmembers_arr).astype(np.float32, copy=False)
            out_blk = np.moveaxis(
                pred_flat.reshape((win_h, win_w, n_bands)), -1, 0
            )
            dst.write(out_blk, window=win)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def calc_unmixing_rmse_residual_err(
    input_img: str,
    input_unmix_img: str,
    endmembers_file: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate linear unmixing prediction RMSE, RMSPE, and mean absolute residual
    error against the original reference image.

    Output bands are: Band 1 ``RMSE``, Band 2 ``RMSPE`` (percentage), and
    Band 3 ``Residual``.

    :param input_img: Path to the original reference reflectance image file.
    :param input_unmix_img: Path to the unmixed abundance coefficients image.
    :param endmembers_file: Path to the ``.mtxt`` endmembers matrix file.
    :param output_img: Path to the 3-band output error image file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, assign band names and calculate image
                       statistics and pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.calc_unmixing_rmse_residual_err(
            "s2_refl.tif", "unmix.tif", "endmembers.mtxt", "unmix_err.tif"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    m, n_bands, endmembers_arr = read_endmembers_mtxt(endmembers_file)

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img, input_unmix_img])
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

        orig_xoff, orig_yoff = grid.offsets[paths[0]]
        unmix_xoff, unmix_yoff = grid.offsets[paths[1]]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
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

            orig_blk = src_orig.read(window=orig_win).astype(np.float64, copy=False)
            unmix_blk = src_unmix.read(
                list(range(1, m + 1)), window=unmix_win
            ).astype(np.float64, copy=False)

            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, n_bands)
            unmix_flat = np.moveaxis(unmix_blk, 0, -1).reshape(-1, m)

            vld = np.logical_not(np.all(orig_flat == 0.0, axis=1)) & np.all(
                np.isfinite(orig_flat), axis=1
            )
            out_flat = np.zeros((orig_flat.shape[0], 3), dtype=np.float32)

            if np.any(vld):
                v_orig = orig_flat[vld]
                v_coef = unmix_flat[vld]
                pred_refl = v_coef @ endmembers_arr
                diff = v_orig - pred_refl

                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)

                rmse = np.sqrt(np.sum(diff * diff, axis=1) / float(n_bands))
                rmspe = np.sqrt(np.sum(norm_diff * norm_diff, axis=1) / float(n_bands)) * 100.0
                residual = np.sum(np.abs(diff), axis=1) / float(n_bands)

                err_stack = np.stack([rmse, rmspe, residual], axis=-1)
                err_stack = np.nan_to_num(err_stack, nan=0.0, posinf=0.0, neginf=0.0)
                out_flat[vld] = err_stack.astype(np.float32, copy=False)

            out_blk = np.moveaxis(out_flat.reshape((win_h, win_w, 3)), -1, 0)
            dst.write(out_blk, window=win)

    if calc_stats:
        set_band_names(output_img, ["RMSE", "RMSPE", "Residual"], feedback=False)
        pop_img_stats(
            output_img,
            no_data_val=None,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def summarise_multi_endmember_linear_unmixing(
    input_img: str,
    in_unmixed_datasets: Sequence[ImageEndmemberInfo],
    out_unmix_img: str,
    out_ref_img: str,
    tmp_dir: str = "tmp",
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Merge multiple linear spectral unmixing results by selecting the model with
    the lowest Root Mean Square Percentage Error (RMSPE) at each pixel.

    :param input_img: Path to the original input reflectance image file.
    :param in_unmixed_datasets: Sequence of :class:`ImageEndmemberInfo`
                                objects describing each candidate unmixing run.
    :param out_unmix_img: Path to the combined output abundance image file.
    :param out_ref_img: Path to the 1-based reference index image indicating
                        which candidate dataset achieved the lowest RMSPE per
                        pixel (``0`` for invalid pixels).
    :param tmp_dir: Retained for API compatibility (Default = ``'tmp'``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, set band names and compute statistics and
                       pyramids (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        info_a = su.ImageEndmemberInfo("unmix_a.tif", "em_a.mtxt", ["Veg", "Soil"])
        info_b = su.ImageEndmemberInfo("unmix_b.tif", "em_b.mtxt", ["Veg", "Water"])
        su.summarise_multi_endmember_linear_unmixing(
            "s2_refl.tif",
            [info_a, info_b],
            "unmix_best.tif",
            "unmix_ref.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    _ = tmp_dir
    if not isinstance(in_unmixed_datasets, (list, tuple)) or not in_unmixed_datasets:
        raise rsgispy.RSGISPyException(
            "in_unmixed_datasets must be a non-empty list of ImageEndmemberInfo."
        )

    endmember_names: List[str] = []
    em_infos: List[Tuple[int, int, np.ndarray]] = []
    img_paths = [input_img]

    for ds_info in in_unmixed_datasets:
        if not isinstance(ds_info, ImageEndmemberInfo):
            raise rsgispy.RSGISPyException(
                "in_unmixed_datasets must contain ImageEndmemberInfo instances."
            )
        if not isinstance(ds_info.endmember_names, (list, tuple)):
            raise rsgispy.RSGISPyException(
                "unmixed_dataset.endmember_names must be a list."
            )
        em_m, em_n, em_arr = read_endmembers_mtxt(ds_info.endmembers_file)
        if em_m != len(ds_info.endmember_names):
            raise rsgispy.RSGISPyException(
                f"Endmember count ({em_m}) in '{ds_info.endmembers_file}' does "
                f"not match endmember_names length ({len(ds_info.endmember_names)})."
            )
        em_infos.append((em_m, em_n, em_arr))
        img_paths.append(ds_info.in_unmix_img)
        for name in ds_info.endmember_names:
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
            if u_ds.count != len(ds_info.endmember_names):
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmember_names ({len(ds_info.endmember_names)})."
                )
            if em_m != u_ds.count:
                raise rsgispy.RSGISPyException(
                    f"Band count ({u_ds.count}) of '{ds_info.in_unmix_img}' "
                    f"does not match endmembers file ({em_m})."
                )
            if em_n != src_orig.count:
                raise rsgispy.RSGISPyException(
                    f"Input image '{input_img}' band count ({src_orig.count}) "
                    f"does not match endmembers file ({em_n})."
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
        dst_ref = open_output_dataset(
            stack=stack,
            output_img=out_ref_img,
            gdalformat=gdalformat,
            out_rio_dtype=ref_dtype,
            grid=grid,
            count=1,
        )

        orig_xoff, orig_yoff = grid.offsets[resolved_paths[0]]
        model_band_maps = [
            [name_to_band[name] for name in ds_info.endmember_names]
            for ds_info in in_unmixed_datasets
        ]

        for win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h = int(win.height)
            win_w = int(win.width)
            n_pxls = win_h * win_w

            orig_win = Window(
                int(win.col_off) + orig_xoff,
                int(win.row_off) + orig_yoff,
                win_w,
                win_h,
            )
            orig_blk = src_orig.read(window=orig_win).astype(np.float64, copy=False)
            orig_flat = np.moveaxis(orig_blk, 0, -1).reshape(-1, src_orig.count)
            vld = np.logical_not(np.all(orig_flat == 0.0, axis=1)) & np.all(
                np.isfinite(orig_flat), axis=1
            )

            best_rmspe = np.full(n_pxls, np.inf, dtype=np.float64)
            best_ref = np.zeros(n_pxls, dtype=np.int32)
            best_abund = np.zeros((n_pxls, n_out_bands), dtype=np.float32)

            if np.any(vld):
                v_orig = orig_flat[vld]
                for ds_idx, ds_info in enumerate(in_unmixed_datasets):
                    u_path = resolved_paths[ds_idx + 1]
                    u_ds = datasets[u_path]
                    u_xoff, u_yoff = grid.offsets[u_path]
                    u_win = Window(
                        int(win.col_off) + u_xoff,
                        int(win.row_off) + u_yoff,
                        win_w,
                        win_h,
                    )
                    u_blk = u_ds.read(window=u_win).astype(np.float64, copy=False)
                    u_flat = np.moveaxis(u_blk, 0, -1).reshape(-1, u_ds.count)
                    v_coef = u_flat[vld]

                    _, n_bands_m, em_arr = em_infos[ds_idx]
                    pred_refl = v_coef @ em_arr
                    diff = v_orig - pred_refl
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_diff = np.where(v_orig != 0.0, diff / v_orig, 0.0)
                    rmspe = (
                        np.sqrt(np.sum(norm_diff * norm_diff, axis=1) / float(n_bands_m))
                        * 100.0
                    )
                    rmspe = np.where(np.isfinite(rmspe), rmspe, np.inf)

                    v_idx = np.flatnonzero(vld)
                    better = rmspe < best_rmspe[v_idx]
                    if np.any(better):
                        win_pxls = v_idx[better]
                        best_rmspe[win_pxls] = rmspe[better]
                        best_ref[win_pxls] = ds_idx + 1
                        best_abund[win_pxls, :] = 0.0
                        for local_b, global_b in enumerate(model_band_maps[ds_idx]):
                            best_abund[win_pxls, global_b] = v_coef[better, local_b]

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
            no_data_val=999,
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


def calc_ppi(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    niters: int = 1000,
    lthres: float = 0,
    uthres: float = 0,
    img_gain: float = 1,
    seed: Optional[int] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the Pixel Purity Index (PPI) using random linear projections of
    mean-centred spectral bands (Boardman et al., 1995).

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output ``uint16`` PPI count raster file.
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param niters: Number of random skewer projections (Default = ``1000``).
    :param lthres: Lower extreme threshold distance in scaled image units
                   (Default = ``0``).
    :param uthres: Upper extreme threshold distance in scaled image units
                   (Default = ``0``).
    :param img_gain: Gain divisor applied to input reflectance values when
                     greater than ``1`` (Default = ``1``).
    :param seed: Optional random seed for reproducible projections.
    :param calc_stats: If ``True``, compute statistics and pyramids on
                       ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during iterations
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagecalc.spectralunmixing as su

        su.calc_ppi(
            "pca_img.tif",
            "ppi_out.tif",
            gdalformat="GTIFF",
            niters=500,
            seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    if niters < 1:
        raise rsgispy.RSGISPyException("niters must be at least 1.")

    pbar_cls = rsgispy.get_tqdm()
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        n_bands = src.count
        height = src.height
        width = src.width
        n_pxls = height * width
        nodatavals = src.nodatavals

        img_data = np.zeros((n_bands, n_pxls), dtype=np.float32)
        img_msk = np.ones(n_pxls, dtype=bool)

        for b_idx in pbar_cls(
            range(n_bands), desc="Importing bands", disable=not use_pbar
        ):
            band_arr = src.read(b_idx + 1).ravel().astype(np.float32, copy=False)
            nd = nodatavals[b_idx]
            b_vld = np.isfinite(band_arr)
            if nd is not None and not (isinstance(nd, float) and math.isnan(nd)):
                b_vld &= band_arr != nd
            img_msk &= b_vld

            if img_gain > 1:
                band_arr = band_arr / float(img_gain)
            band_tmp = np.where(b_vld, band_arr, np.nan)
            band_mean = float(np.nanmean(band_tmp)) if np.any(b_vld) else 0.0
            img_data[b_idx] = band_arr - band_mean

        vld_data = img_data[:, img_msk].T  # (n_valid, n_bands)
        counts = np.zeros(vld_data.shape[0], dtype=np.uint16)

        if vld_data.shape[0] > 0:
            rng = np.random.default_rng(seed)
            for _ in pbar_cls(
                range(niters), desc="Calculating PPI", disable=not use_pbar
            ):
                skewer = rng.random(n_bands) - 0.5
                proj = vld_data @ skewer
                s_min = float(np.min(proj))
                s_max = float(np.max(proj))
                if lthres == 0:
                    counts[int(np.argmin(proj))] += 1
                else:
                    counts[proj <= (s_min + lthres)] += 1

                if uthres == 0:
                    counts[int(np.argmax(proj))] += 1
                else:
                    counts[proj >= (s_max - uthres)] += 1

        out_full = np.zeros(n_pxls, dtype=np.uint16)
        out_full[img_msk] = counts
        out_img_arr = out_full.reshape((height, width))

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint16",
            count=1,
        )
        dst.write(out_img_arr, 1)

    if calc_stats:
        pop_img_stats(
            output_img,
            no_data_val=0,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
