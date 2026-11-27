"""
Extrapolation / novelty maps for :mod:`rsgispy.modelling.species_distribution`:
the multivariate environmental similarity surface (MESS, Elith et al. 2010)
with the most dissimilar variable, and ExDet (Mesgaran et al. 2014) univariate
(NT1) and combinational (NT2) novelty. Both compare the environment of every
pixel (e.g. of a future climate or another region) with the environment of the
samples a model was trained on.
"""

import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageutils.colourtable import set_thematic_classes
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.modelling._helpers import (
    EnvVarInfo,
    lookup_env_var,
    predict_raster_from_env_vars,
    read_model_vector,
    resolve_analysis_vars,
)
from rsgispy.vectorprocess import VectorInput, validate_columns_exist

__all__ = [
    "calc_mess",
    "calc_exdet",
]

_NOVELTY_NO_DATA = -9999.0


class _RefVar:
    """Reference (training) distribution of one variable."""

    # pylint: disable=too-few-public-methods
    def __init__(self, name: str, vals: np.ndarray, categorical: bool) -> None:
        self.name = name
        self.categorical = categorical
        self.sorted = np.sort(vals)
        self.categories = np.unique(vals) if categorical else None
        self.v_min = float(self.sorted[0])
        self.v_max = float(self.sorted[-1])
        v_rng = self.v_max - self.v_min
        self.v_range = v_rng if v_rng > 0.0 else 1.0


def _read_reference(
    ref_smpls_vec_file: VectorInput,
    ref_smpls_vec_lyr: Optional[str],
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    ref_no_data_val: Optional[float],
) -> List[_RefVar]:
    var_names = resolve_analysis_vars(env_vars, analysis_vars)
    ref_gdf = read_model_vector(ref_smpls_vec_file, vec_lyr=ref_smpls_vec_lyr)
    validate_columns_exist(ref_gdf, var_names)
    vals = ref_gdf[var_names].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    good = np.all(np.isfinite(vals), axis=1)
    if ref_no_data_val is not None:
        good &= ~np.any(vals == float(ref_no_data_val), axis=1)
    if np.count_nonzero(good) < 2:
        raise rsgispy.RSGISPyException(
            "At least two reference samples with valid values for every variable "
            "are needed."
        )
    ref_vars = []
    for col_idx, var_name in enumerate(var_names):
        info = lookup_env_var(env_vars, var_name)
        ref_vars.append(
            _RefVar(
                var_name,
                vals[good, col_idx],
                info.data_type == rsgispy.VAR_TYPE_CATEGORICAL,
            )
        )
    return ref_vars


def _mess_similarity(ref: _RefVar, pvals: np.ndarray) -> np.ndarray:
    """Per-variable MESS similarity (Elith et al. 2010) of ``pvals``.
    Categorical variables: ``100`` for categories in the reference data and
    ``-100`` for categories not in it."""
    if ref.categorical:
        return np.where(np.isin(pvals, ref.categories), 100.0, -100.0)
    n_ref = ref.sorted.size
    pct_below = 100.0 * np.searchsorted(ref.sorted, pvals, side="left") / n_ref
    low = 100.0 * (pvals - ref.v_min) / ref.v_range
    high = 100.0 * (ref.v_max - pvals) / ref.v_range
    return np.select(
        [pct_below == 0.0, pct_below <= 50.0, pct_below < 100.0],
        [low, 2.0 * pct_below, 2.0 * (100.0 - pct_below)],
        default=high,
    )


def _var_colours(n_vars: int) -> List[Tuple[int, int, int]]:
    cmap = matplotlib.colormaps["tab20"]
    return [
        tuple(int(round(255 * c)) for c in cmap(idx % 20)[:3])  # type: ignore[misc]
        for idx in range(n_vars)
    ]


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_mess(
    ref_smpls_vec_file: VectorInput,
    ref_smpls_vec_lyr: Optional[str],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    out_mod_var_img: Optional[str] = None,
    ref_no_data_val: Optional[float] = -9999.0,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the multivariate environmental similarity surface (MESS; Elith et
    al. 2010): how similar the environment of every pixel is to the environment
    of the reference (training) samples. Use it before projecting a model to a
    new region or a future climate: where MESS is negative at least one
    variable is outside the range of the training data, and the model is
    extrapolating.

    For each variable, the similarity of a pixel value ``p`` is calculated from
    ``f``, the percentage of reference values below ``p``: ``2f`` when
    ``f <= 50``, ``2(100 - f)`` when ``50 < f < 100``, and, outside the
    reference range, ``100 (p - min) / (max - min)`` or
    ``100 (max - p) / (max - min)`` (negative). MESS is the minimum over the
    variables: ``100`` at the reference median of every variable, ``0`` at
    the edge of the range, negative outside it. Categorical variables
    (``EnvVarInfo.data_type == rsgispy.VAR_TYPE_CATEGORICAL``) score ``100``
    for categories seen in the reference samples and ``-100`` for unseen ones.

    ``env_vars`` may point at different rasters from those the samples were
    extracted from (e.g. a future scenario; see
    :func:`copy_env_vars_for_scenario`), as long as the variable names match
    the sample columns. The reference values must be in the same (raw) units
    as the rasters.

    :param ref_smpls_vec_file: Reference samples with one column per variable
                               (e.g. the training layer from
                               :func:`extract_env_var_data`): path or
                               :class:`geopandas.GeoDataFrame`.
    :param ref_smpls_vec_lyr: Reference layer name (or ``None``).
    :param in_msk_img: Mask raster of the area to map.
    :param img_msk_val: Value of the pixels to map in ``in_msk_img``.
    :param env_vars: Environmental variable descriptors of the rasters to assess.
    :param analysis_vars: Variables to compare.
    :param output_img: Output ``float32`` MESS raster (no data ``-9999``).
    :param out_mod_var_img: Optional output ``uint8`` thematic raster of the most
                            dissimilar variable (MoD) at each pixel: value ``i``
                            is ``analysis_vars[i - 1]``, named in the class names
                            and colour table; ``0`` is no data.
    :param ref_no_data_val: Reference value treated as missing (Default =
                            ``-9999.0``); rows with it are ignored.
    :param fixed_var_vals: Optional variables held at a constant instead of
                           being read (as in the ``pred_*`` functions).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param block_size: Processing block size in pixels (Default = ``None``).
    :param use_pbar: Show progress bars (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        future_vars = sdm.copy_env_vars_for_scenario(
            env_vars, {"winter_temp": "winter_temp_2080.tif"}
        )
        sdm.calc_mess(
            train_gdf, None, "valid_mask.tif", 1, future_vars, analysis_vars,
            "mess_2080.tif", out_mod_var_img="mess_mod_2080.tif",
        )
    """
    ref_vars = _read_reference(
        ref_smpls_vec_file, ref_smpls_vec_lyr, env_vars, analysis_vars, ref_no_data_val
    )

    def _similarities(env_df: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [
                _mess_similarity(ref, env_df[ref.name].to_numpy(dtype=np.float64))
                for ref in ref_vars
            ]
        )

    common: Dict[str, Any] = {
        "in_msk_img": in_msk_img,
        "img_msk_val": img_msk_val,
        "env_vars": env_vars,
        "analysis_vars": analysis_vars,
        "gdalformat": gdalformat,
        "block_size": block_size,
        "use_pbar": use_pbar,
        "fixed_var_vals": fixed_var_vals,
    }
    predict_raster_from_env_vars(
        output_img=output_img,
        predict_fn=lambda env_df: np.min(_similarities(env_df), axis=0),
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=_NOVELTY_NO_DATA,
        calc_stats=calc_stats,
        **common,
    )
    if out_mod_var_img is not None:
        if len(ref_vars) > 255:
            raise rsgispy.RSGISPyException(
                "out_mod_var_img supports at most 255 variables."
            )
        predict_raster_from_env_vars(
            output_img=out_mod_var_img,
            predict_fn=lambda env_df: np.argmin(_similarities(env_df), axis=0) + 1,
            datatype=rsgispy.TYPE_8UINT,
            no_data_val=0,
            thematic=True,
            calc_stats=False,
            **common,
        )
        colours = _var_colours(len(ref_vars))
        set_thematic_classes(
            os.fspath(out_mod_var_img),
            {idx + 1: (ref.name, colours[idx]) for idx, ref in enumerate(ref_vars)},
            ignore_zero=True,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


def _exdet_reference(
    ref_vars: Sequence[_RefVar],
    ref_smpls_vec_file: VectorInput,
    ref_smpls_vec_lyr: Optional[str],
    ref_no_data_val: Optional[float],
) -> Tuple[List[_RefVar], np.ndarray, np.ndarray, float]:
    """Continuous reference variables, their mean, (pseudo-)inverse covariance
    and the maximum squared Mahalanobis distance of the reference samples."""
    cont = [ref for ref in ref_vars if not ref.categorical]
    if not cont:
        raise rsgispy.RSGISPyException(
            "ExDet needs at least one continuous (non-categorical) variable."
        )
    ref_gdf = read_model_vector(ref_smpls_vec_file, vec_lyr=ref_smpls_vec_lyr)
    names = [ref.name for ref in ref_vars]
    vals = ref_gdf[names].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    good = np.all(np.isfinite(vals), axis=1)
    if ref_no_data_val is not None:
        good &= ~np.any(vals == float(ref_no_data_val), axis=1)
    cont_idx = [names.index(ref.name) for ref in cont]
    ref_mat = vals[good][:, cont_idx]
    mean = ref_mat.mean(axis=0)
    cov = np.atleast_2d(np.cov(ref_mat, rowvar=False))
    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        warnings.warn(
            "The covariance matrix of the reference samples is singular (e.g. "
            "perfectly collinear or constant variables); NT2 uses its "
            "pseudo-inverse.",
            UserWarning,
            stacklevel=3,
        )
    inv_cov = np.linalg.pinv(cov)
    diff = ref_mat - mean
    md2_ref = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    md2_max = float(np.max(md2_ref))
    if md2_max <= 0.0:
        raise rsgispy.RSGISPyException(
            "The reference samples have no spread; NT2 is undefined."
        )
    return cont, mean, inv_cov, md2_max


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_exdet(
    ref_smpls_vec_file: VectorInput,
    ref_smpls_vec_lyr: Optional[str],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    ref_no_data_val: Optional[float] = -9999.0,
    fixed_var_vals: Optional[Mapping[str, float]] = None,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Calculate the ExDet extrapolation detection metrics (Mesgaran et al. 2014)
    and write them as a two-band ``float32`` raster (no data ``-9999``):

    * **Band 1, NT1 — univariate novelty.** The sum over the variables of how
      far a pixel lies outside the reference range, relative to the range:
      ``sum(min(p - min, max - p, 0) / (max - min))``. ``0`` inside the
      univariate range of every variable, negative outside it.
    * **Band 2, NT2 — combinational novelty.** The squared Mahalanobis
      distance of the pixel from the reference mean (using the reference
      covariance) divided by the largest squared distance of any reference
      sample. Values ``> 1`` where NT1 is ``0`` are novel *combinations* of
      values that are each within range (e.g. warm and wet where the training
      data are warm and dry or cool and wet). NT2 is most useful where NT1 is
      ``0``.

    Categorical variables are left out of NT2; in NT1 a category not seen in
    the reference samples counts ``-1``. A singular reference covariance (e.g.
    collinear variables) is replaced by its pseudo-inverse, with a warning.
    See :func:`calc_mess` for the inputs.

    :param ref_smpls_vec_file: Reference samples with one column per variable.
    :param ref_smpls_vec_lyr: Reference layer name (or ``None``).
    :param in_msk_img: Mask raster of the area to map.
    :param img_msk_val: Value of the pixels to map in ``in_msk_img``.
    :param env_vars: Environmental variable descriptors of the rasters to assess.
    :param analysis_vars: Variables to compare.
    :param output_img: Output two-band raster (``NT1``, ``NT2``).
    :param ref_no_data_val: Reference value treated as missing
                            (Default = ``-9999.0``).
    :param fixed_var_vals: Optional variables held at a constant.
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param block_size: Processing block size in pixels (Default = ``None``).
    :param use_pbar: Show progress bars (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.calc_exdet(
            train_gdf, None, "valid_mask.tif", 1, future_vars, analysis_vars,
            "exdet_2080.tif",
        )
    """
    ref_vars = _read_reference(
        ref_smpls_vec_file, ref_smpls_vec_lyr, env_vars, analysis_vars, ref_no_data_val
    )
    cont, mean, inv_cov, md2_max = _exdet_reference(
        ref_vars, ref_smpls_vec_file, ref_smpls_vec_lyr, ref_no_data_val
    )

    def _exdet(env_df: pd.DataFrame) -> np.ndarray:
        nt1 = np.zeros(len(env_df), dtype=np.float64)
        for ref in ref_vars:
            pvals = env_df[ref.name].to_numpy(dtype=np.float64)
            if ref.categorical:
                nt1 -= (~np.isin(pvals, ref.categories)).astype(np.float64)
            else:
                nt1 += (
                    np.minimum(
                        np.minimum(pvals - ref.v_min, ref.v_max - pvals), 0.0
                    )
                    / ref.v_range
                )
        diff = env_df[[ref.name for ref in cont]].to_numpy(dtype=np.float64) - mean
        nt2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff) / md2_max
        return np.vstack([nt1, nt2])

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_exdet,
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=_NOVELTY_NO_DATA,
        calc_stats=False,
        gdalformat=gdalformat,
        block_size=block_size,
        use_pbar=use_pbar,
        fixed_var_vals=fixed_var_vals,
        out_band_names=["NT1", "NT2"],
    )
    if calc_stats:
        pop_img_stats(
            os.fspath(output_img),
            no_data_val=_NOVELTY_NO_DATA,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )
