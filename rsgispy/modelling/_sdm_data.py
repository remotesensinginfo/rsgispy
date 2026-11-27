"""
Data preparation, sampling, normalisation, and collinearity diagnostics for
:mod:`rsgispy.modelling.species_distribution`.
"""

# pylint: disable=too-many-lines

import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageutils.metadata import (
    get_epsg_proj_from_img,
    get_img_bbox,
    get_wkt_proj_from_img,
)
from rsgispy.modelling._helpers import (
    _MASK_BAND_KEY,
    EnvVarInfo,
    lookup_env_var,
    normalise_array_with_info,
    read_model_vector,
    resolve_analysis_vars,
    run_env_block_pipeline,
    write_model_vector,
)
from rsgispy.modelling._sdm_bias import sample_bias_weighted_pts
from rsgispy.tools._ml_explain import plot_class_feature_distributions
from rsgispy.tools._ml_features import run_correlation_matrix
from rsgispy.tools.featureselection import calc_pandas_vif
import rsgispy.tools.files
from rsgispy.vector.create import create_random_pts_in_bbox
from rsgispy.vector.selection import vec_lyr_random_subset
from rsgispy.vector.stats import calc_near_dist_to_feats
from rsgispy.vectorprocess import VectorInput, validate_columns_exist
from rsgispy.zonalstats.point import ext_point_band_values

__all__ = [
    "create_finite_mask",
    "gen_pseudo_absences_smpls",
    "extract_env_var_data",
    "thin_pts_env_space",
    "combine_presence_absence_data",
    "create_train_test_sets",
    "pop_normalise_coeffs",
    "apply_normalise_coeffs",
    "comparison_box_plots",
    "correlation_matrix",
    "calc_vif_multicollinearity",
]


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_finite_mask(
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,
    use_pbar: bool = True,
    calc_stats: bool = True,
) -> None:
    """
    Generate a binary valid-data mask (``1`` valid, ``0`` invalid) where
    ``in_msk_img == img_msk_val`` and all selected environmental covariate bands
    are finite and not equal to their raster no-data values.

    Processed in streaming blocks via :mod:`rsgispy.imageprocess` without
    writing intermediate raster files.

    :param in_msk_img: Path to the region-of-interest mask image.
    :param img_msk_val: Pixel value within ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of variable name to :class:`EnvVarInfo`.
    :param analysis_vars: Sequence of variable names in ``env_vars`` to check.
    :param output_img: Path to the output 8-bit thematic mask image.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param use_pbar: Display a tqdm progress bar during processing (Default = ``True``).
    :param calc_stats: Populate thematic statistics and pyramids (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.create_finite_mask(
            "roi_mask.tif", 1, env_vars, ["temp", "precip"], "valid_mask.tif"
        )
    """
    _ = tmp_dir  # Retained for legacy signature compatibility; streaming needs no tmp.

    def _eval_mask_block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        resolved_vars: List[str],
        var_keys: Dict[str, str],
        block_vars: Dict[str, np.ndarray],
        datasets: Mapping[str, Any],
        _win_h: int,
        _win_w: int,
    ) -> np.ndarray:
        valid = block_vars[_MASK_BAND_KEY] == img_msk_val
        for var_name in resolved_vars:
            info = lookup_env_var(env_vars, var_name)
            if info.file is None:
                raise rsgispy.RSGISPyException(
                    f"Environmental variable '{var_name}' has file=None."
                )
            arr = block_vars[var_keys[var_name]]
            valid &= np.isfinite(arr)
            nd_val = datasets[os.fspath(info.file)].nodatavals[info.band - 1]
            if nd_val is not None and not (
                isinstance(nd_val, float) and np.isnan(nd_val)
            ):
                valid &= arr != float(nd_val)
            if info.min_vld_val is not None:
                valid &= arr >= float(info.min_vld_val)
            if info.max_vld_val is not None:
                valid &= arr <= float(info.max_vld_val)
        return valid.astype(np.uint8)

    run_env_block_pipeline(
        in_msk_img=in_msk_img,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        block_eval_fn=_eval_mask_block,
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        calc_stats=calc_stats,
        thematic=True,
        no_data_val=0.0,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _check_pseudo_abs_strategies(
    presence_smpls_vec_file: Optional[VectorInput],
    presence_smpls_dist_thres: float,
    presence_smpls_max_dist: Optional[float],
    env_profile: Optional[str],
    env_profile_quantiles: Tuple[float, float],
    env_vars: Optional[Mapping[str, EnvVarInfo]],
    analysis_vars: Optional[Sequence[str]],
) -> None:
    """Validate the optional pseudo-absence strategies of
    :func:`gen_pseudo_absences_smpls` before any sampling."""
    if presence_smpls_max_dist is not None:
        if presence_smpls_vec_file is None:
            raise rsgispy.RSGISPyException(
                "presence_smpls_max_dist needs presence_smpls_vec_file."
            )
        if float(presence_smpls_max_dist) <= float(presence_smpls_dist_thres):
            raise rsgispy.RSGISPyException(
                "presence_smpls_max_dist must be greater than "
                "presence_smpls_dist_thres."
            )
    if env_profile is None:
        return
    if str(env_profile).strip().lower() != "sre":
        raise rsgispy.RSGISPyException(
            f"Unsupported env_profile '{env_profile}'; use 'sre'."
        )
    if presence_smpls_vec_file is None:
        raise rsgispy.RSGISPyException("env_profile needs presence_smpls_vec_file.")
    if not env_vars or not analysis_vars:
        raise rsgispy.RSGISPyException("env_profile needs env_vars and analysis_vars.")
    q_lo, q_hi = (float(q) for q in env_profile_quantiles)
    if not 0.0 <= q_lo < q_hi <= 1.0:
        raise rsgispy.RSGISPyException(
            "env_profile_quantiles must be (low, high) with 0 <= low < high <= 1."
        )


def _env_values(
    gdf: gpd.GeoDataFrame,
    env_vars: Mapping[str, EnvVarInfo],
    var_names: Sequence[str],
    use_pbar: bool,
) -> np.ndarray:
    """Covariate values at ``gdf``'s points (``NaN`` for no data)."""
    sub_vars = {name: lookup_env_var(env_vars, name) for name in var_names}
    nd_val = -9999.0
    ext_gdf = extract_env_var_data(
        sub_vars,
        gdf[[gdf.geometry.name]].reset_index(drop=True),
        out_no_data_val=nd_val,
        reproj_vec=True,
        use_pbar=use_pbar,
    )
    vals = np.array(
        ext_gdf[[sub_vars[name].name for name in var_names]].to_numpy(
            dtype=np.float64
        ),
        copy=True,
    )
    vals[vals == nd_val] = np.nan
    return vals


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _filter_env_profile(
    pts_gdf: gpd.GeoDataFrame,
    pres_gdf: gpd.GeoDataFrame,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    quantiles: Tuple[float, float],
    use_pbar: bool,
) -> gpd.GeoDataFrame:
    """Surface range envelope (SRE) filter: drop candidates inside the
    presences' per-variable quantile envelope on every variable."""
    var_names = resolve_analysis_vars(env_vars, analysis_vars)
    pres_vals = _env_values(pres_gdf, env_vars, var_names, use_pbar)
    pres_vals = pres_vals[np.all(np.isfinite(pres_vals), axis=1)]
    if pres_vals.shape[0] < 2:
        raise rsgispy.RSGISPyException(
            "Fewer than two presences have valid values for every variable; "
            "the environmental envelope cannot be defined."
        )
    lows = np.quantile(pres_vals, float(quantiles[0]), axis=0)
    highs = np.quantile(pres_vals, float(quantiles[1]), axis=0)
    pts_gdf = pts_gdf.reset_index(drop=True)
    cand_vals = _env_values(pts_gdf, env_vars, var_names, use_pbar)
    valid = np.all(np.isfinite(cand_vals), axis=1)
    inside = np.all((cand_vals >= lows) & (cand_vals <= highs), axis=1)
    out_gdf = pts_gdf[valid & ~inside].copy()
    if out_gdf.empty:
        raise rsgispy.RSGISPyException(
            "All candidate pseudo-absences lie inside the presences' environmental "
            "envelope (or on no data); widen env_profile_quantiles or increase "
            "xtra_n_smpls."
        )
    return out_gdf


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
def gen_pseudo_absences_smpls(
    in_msk_img: str,
    img_msk_val: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    n_smpls: int = 10000,
    xtra_n_smpls: int = 30000,
    presence_smpls_vec_file: Optional[VectorInput] = None,
    presence_smpls_vec_lyr: Optional[str] = None,
    presence_smpls_dist_thres: float = 1000.0,
    out_format: str = "GeoJSON",
    rnd_seed: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
    use_pbar: bool = True,
    *,
    bias_img: Optional[str] = None,
    bias_img_band: int = 1,
    presence_smpls_max_dist: Optional[float] = None,
    env_profile: Optional[str] = None,
    env_profile_quantiles: Tuple[float, float] = (0.025, 0.975),
    env_vars: Optional[Mapping[str, EnvVarInfo]] = None,
    analysis_vars: Optional[Sequence[str]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate pseudo-absence point samples within ``in_msk_img == img_msk_val``,
    optionally excluding points within ``presence_smpls_dist_thres`` of
    known presence points.

    By default the candidate points are uniformly random. With ``bias_img``
    (e.g. from :func:`create_bias_grid`) they are drawn with probability
    proportional to the bias raster instead, so the background has the same
    sampling bias as the presences ("bias-weighted" or "target-group"
    background). ``bias_img`` must be on the same grid as ``in_msk_img``;
    zero, negative and no-data bias pixels are never sampled. A pixel can be
    drawn more than once, but each point is placed randomly within its pixel.

    Two further strategies filter the candidates (both need the presences):

    * ``presence_smpls_max_dist`` — keep only candidates no further than this
      from a presence; with ``presence_smpls_dist_thres`` this gives a "disk"
      or band of pseudo-absences between a minimum and a maximum distance,
      e.g. to represent the area the species could have reached.
    * ``env_profile="sre"`` — surface range envelope: drop candidates whose
      environment lies inside the presences' envelope on every variable in
      ``analysis_vars`` (between the ``env_profile_quantiles`` of the presence
      values), so pseudo-absences are placed only where the environment is
      unlike the presences'. This makes the classes easier to separate and
      inflates AUC; use it for models that need clearly unsuitable absences.
      Candidates with no-data variable values are dropped.

    Bias sampling (``bias_img``) decides where candidates are drawn; the
    strategies then filter them, before the final random subset of
    ``n_smpls``.

    :param in_msk_img: Path to the valid ROI mask raster.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param n_smpls: Maximum number of pseudo-absence points to retain.
    :param xtra_n_smpls: Additional candidate points drawn prior to masking.
    :param presence_smpls_vec_file: Optional presence points path or GeoDataFrame.
    :param presence_smpls_vec_lyr: Optional presence layer name.
    :param presence_smpls_dist_thres: Minimum allowed distance from any
                                      presence point (in CRS units).
    :param out_format: Output vector driver (Default = ``'GeoJSON'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param out_compress: Optional GeoParquet compression setting.
    :param use_pbar: Display a tqdm progress bar during sampling (Default = ``True``).
    :param bias_img: Optional sampling-bias raster on the grid of ``in_msk_img``
                     (Default = ``None``: uniform sampling).
    :param bias_img_band: Band of ``bias_img`` to use (Default = ``1``).
    :param presence_smpls_max_dist: Optional maximum distance from the nearest
                                    presence (CRS units).
    :param env_profile: Optional environmental profile filter: ``"sre"``.
    :param env_profile_quantiles: Lower and upper quantiles of the presence
                                  values defining the envelope
                                  (Default = ``(0.025, 0.975)``; ``(0, 1)`` is
                                  the full range).
    :param env_vars: Environmental variable descriptors (needed for
                     ``env_profile``).
    :param analysis_vars: Variables of the envelope (needed for
                          ``env_profile``).
    :return: Pseudo-absence :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        pts_gdf = sdm.gen_pseudo_absences_smpls(
            "valid_mask.tif", 1, n_smpls=500, xtra_n_smpls=1500, rnd_seed=42
        )
    """
    if n_smpls <= 0:
        raise rsgispy.RSGISPyException("n_smpls must be a positive integer.")
    if xtra_n_smpls < 0:
        raise rsgispy.RSGISPyException("xtra_n_smpls must be >= 0.")
    _check_pseudo_abs_strategies(
        presence_smpls_vec_file,
        presence_smpls_dist_thres,
        presence_smpls_max_dist,
        env_profile,
        env_profile_quantiles,
        env_vars,
        analysis_vars,
    )

    if bias_img is not None:
        pts_gdf = sample_bias_weighted_pts(
            in_msk_img,
            img_msk_val,
            bias_img,
            int(n_smpls) + int(xtra_n_smpls),
            bias_img_band=bias_img_band,
            rnd_seed=rnd_seed,
            use_pbar=use_pbar,
        )
    else:
        img_bbox = get_img_bbox(in_msk_img)
        epsg_code = get_epsg_proj_from_img(in_msk_img)
        crs_spec: Union[int, str]
        if epsg_code is not None:
            crs_spec = epsg_code
        else:
            wkt = get_wkt_proj_from_img(in_msk_img)
            if not wkt:
                raise rsgispy.RSGISPyException(
                    f"Could not determine projection from mask image '{in_msk_img}'."
                )
            crs_spec = wkt

        pts_gdf = create_random_pts_in_bbox(
            bbox=img_bbox,
            n_pts=int(n_smpls) + int(xtra_n_smpls),
            epsg_code=crs_spec,
            out_vec_file=None,
            rnd_seed=rnd_seed,
        )
        if pts_gdf is None:
            raise rsgispy.RSGISPyException(
                "Failed to generate random candidate points within the mask extent."
            )

        tmp_col = "__rsgispy_tmp_msk__"
        pts_gdf = ext_point_band_values(
            vec_gdf=pts_gdf,
            input_img=in_msk_img,
            img_band=1,
            min_thres=-np.inf,
            max_thres=np.inf,
            out_no_data_val=-9999.0,
            out_field=tmp_col,
            reproj_vec=False,
            use_pbar=use_pbar,
        )
        pts_gdf = (
            pts_gdf[pts_gdf[tmp_col] == img_msk_val].drop(columns=[tmp_col]).copy()
        )
        if pts_gdf.empty:
            raise rsgispy.RSGISPyException(
                "No pseudo-absence samples fell within the specified mask region."
            )

    if presence_smpls_vec_file is not None:
        pres_gdf = read_model_vector(
            presence_smpls_vec_file, vec_lyr=presence_smpls_vec_lyr
        )
        if not pres_gdf.empty:
            if pres_gdf.crs is not None and pts_gdf.crs is not None:
                if pres_gdf.crs != pts_gdf.crs:
                    pres_gdf = pres_gdf.to_crs(pts_gdf.crs)
            orig_cols = list(pts_gdf.columns)
            pts_reset = pts_gdf.reset_index(drop=True)
            dist_gdf = calc_near_dist_to_feats(
                vec_file=pts_reset,
                vec_lyr=None,
                vec_dist_file=pres_gdf[["geometry"]],
                vec_dist_lyr=None,
                out_vec_file=None,
                out_col="__presence_dist__",
            )
            if dist_gdf is None:
                raise rsgispy.RSGISPyException(
                    "Failed to compute nearest distances to presence samples."
                )
            dist_gdf = dist_gdf[~dist_gdf.index.duplicated(keep="first")]
            keep = dist_gdf["__presence_dist__"] > float(presence_smpls_dist_thres)
            if presence_smpls_max_dist is not None:
                keep &= dist_gdf["__presence_dist__"] <= float(presence_smpls_max_dist)
            dist_gdf = dist_gdf[keep]
            pts_gdf = dist_gdf[orig_cols].copy()
            if pts_gdf.empty:
                raise rsgispy.RSGISPyException(
                    "All candidate pseudo-absence samples were within "
                    "presence_smpls_dist_thres of presence points"
                    + (
                        " or beyond presence_smpls_max_dist."
                        if presence_smpls_max_dist is not None
                        else "."
                    )
                )
        if env_profile is not None and env_vars is not None and analysis_vars:
            pts_gdf = _filter_env_profile(
                pts_gdf,
                pres_gdf,
                env_vars,
                analysis_vars,
                env_profile_quantiles,
                use_pbar,
            )

    if len(pts_gdf) > n_smpls:
        pts_gdf = vec_lyr_random_subset(
            vec_file=pts_gdf,
            vec_lyr=None,
            out_vec_file=None,
            n_smpl=int(n_smpls),
            rnd_seed=rnd_seed,
        )
        if pts_gdf is None:
            raise rsgispy.RSGISPyException(
                "Failed to subsample candidate pseudo-absence points."
            )

    pts_gdf = pts_gdf.reset_index(drop=True)
    return write_model_vector(
        pts_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def extract_env_var_data(
    env_vars: Mapping[str, EnvVarInfo],
    smpls_vec_file: VectorInput,
    smpls_vec_lyr: Optional[str] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_no_data_val: float = -9999.0,
    replace: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
    use_pbar: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Extract environmental covariate values at sample point locations using
    :func:`rsgispy.zonalstats.point.ext_point_band_values`.

    :param env_vars: Mapping of variable key to :class:`EnvVarInfo`.
    :param smpls_vec_file: Sample point vector file path or :class:`geopandas.GeoDataFrame`.
    :param smpls_vec_lyr: Sample vector layer name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_no_data_val: Fill value for out-of-bounds/invalid/nodata pixels.
    :param replace: Legacy parameter; output files are overwritten when written.
    :param reproj_vec: Reproject sample points on the fly when CRSs differ.
    :param vec_def_epsg: Optional fallback EPSG code if input vector lacks CRS.
    :param out_compress: Optional GeoParquet compression setting.
    :param use_pbar: Display a tqdm progress bar during extraction (Default = ``True``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        pts_gdf = sdm.extract_env_var_data(env_vars, "presence.gpkg", "presence")
    """
    _ = replace  # Legacy parameter; outputs are always overwritten when written.
    if not env_vars:
        raise rsgispy.RSGISPyException("env_vars must not be empty.")

    vec_gdf = read_model_vector(smpls_vec_file, vec_lyr=smpls_vec_lyr)

    # Group covariates by raster file so multi-band files are opened once.
    grouped: Dict[str, List[EnvVarInfo]] = {}
    for key, info in env_vars.items():
        if info.file is None:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{key}' has file=None."
            )
        if not info.name:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{key}' has no column name defined."
            )
        grouped.setdefault(os.fspath(info.file), []).append(info)

    for img_path, infos in grouped.items():
        bands = [info.band for info in infos]
        tmp_fields = [f"__rsgispy_env_{idx}__" for idx in range(len(infos))]
        vec_gdf = ext_point_band_values(
            vec_gdf=vec_gdf,
            input_img=img_path,
            img_band=bands,
            min_thres=-np.inf,
            max_thres=np.inf,
            out_no_data_val=out_no_data_val,
            out_field=tmp_fields,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
        )
        for tmp_col, info in zip(tmp_fields, infos):
            col_vals = vec_gdf[tmp_col].to_numpy(dtype=np.float64)
            valid = col_vals != float(out_no_data_val)
            if info.min_vld_val is not None:
                valid &= col_vals >= float(info.min_vld_val)
            if info.max_vld_val is not None:
                valid &= col_vals <= float(info.max_vld_val)
            col_vals = np.where(valid, col_vals, float(out_no_data_val))
            vec_gdf = vec_gdf.drop(columns=[tmp_col])
            if not info.name:
                raise rsgispy.RSGISPyException(
                    "Environmental variable is missing a column name."
                )
            vec_gdf[info.name] = col_vals

    return write_model_vector(
        vec_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def thin_pts_env_space(
    smpls_vec_file: VectorInput,
    smpls_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    n_bins: Union[int, Mapping[str, int]] = 10,
    keep: str = "random",
    rnd_seed: Optional[int] = None,
    no_data_val: Optional[float] = -9999.0,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Thin samples in **environmental** space: divide the range of each variable
    in ``analysis_vars`` into ``n_bins`` equal-width bins and keep at most one
    sample per occupied combination of bins. This reduces the over-representation
    of commonly sampled environmental conditions, complementing geographic
    thinning (:func:`rsgispy.vector.geoms.thin_pts_by_raster_grid`,
    :func:`rsgispy.vector.geoms.thin_pts_min_dist`).

    The samples must already have one column per variable (e.g. from
    :func:`extract_env_var_data`). Rows with a non-finite value or a value equal
    to ``no_data_val`` in any variable are dropped with a warning. All
    attributes are kept, and the retained rows stay in their input order. With
    many variables the bins become sparse and little is removed: use few
    variables (e.g. the main 2–4) or few bins.

    :param smpls_vec_file: Sample vector file path or :class:`geopandas.GeoDataFrame`.
    :param smpls_vec_lyr: Sample vector layer name (or ``None``).
    :param analysis_vars: Variable columns defining the environmental space.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param n_bins: Number of bins per variable, or a mapping of variable name to
                   number of bins (Default = ``10``).
    :param keep: Sample kept per occupied bin: ``"random"`` (Default) or
                 ``"first"`` (first in input order).
    :param rnd_seed: Optional random seed used when ``keep="random"``.
    :param no_data_val: Value treated as missing (Default = ``-9999.0``, the
                        :func:`extract_env_var_data` default); ``None`` to disable.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Thinned :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        thin_gdf = sdm.thin_pts_env_space(
            pres_gdf, None, ["winter_temp", "elevation"], n_bins=8, rnd_seed=42
        )
    """
    keep_mode = str(keep).strip().lower()
    if keep_mode not in ("random", "first"):
        raise rsgispy.RSGISPyException(
            f"Unsupported keep '{keep}'; use 'random' or 'first'."
        )
    var_names = list(analysis_vars)
    if not var_names:
        raise rsgispy.RSGISPyException("analysis_vars must not be empty.")

    if isinstance(n_bins, Mapping):
        missing_bins = [v for v in var_names if v not in n_bins]
        if missing_bins:
            raise rsgispy.RSGISPyException(
                f"n_bins has no entry for variable(s): {missing_bins}."
            )
        bins_per_var = {v: int(n_bins[v]) for v in var_names}
    else:
        bins_per_var = {v: int(n_bins) for v in var_names}
    bad_bins = {v: b for v, b in bins_per_var.items() if b < 1}
    if bad_bins:
        raise rsgispy.RSGISPyException(f"n_bins must be >= 1, got {bad_bins}.")

    smpls_gdf = read_model_vector(smpls_vec_file, vec_lyr=smpls_vec_lyr)
    validate_columns_exist(smpls_gdf, var_names)

    vals = (
        smpls_gdf[var_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    valid = np.all(np.isfinite(vals), axis=1)
    if no_data_val is not None:
        valid &= ~np.any(vals == float(no_data_val), axis=1)
    n_invalid = int(np.sum(~valid))
    if n_invalid > 0:
        warnings.warn(
            f"{n_invalid} sample(s) have missing or no-data variable values and "
            "were dropped.",
            UserWarning,
            stacklevel=2,
        )

    n_smpls = vals.shape[0]
    bin_idx = np.zeros((n_smpls, len(var_names)), dtype=np.int64)
    for var_i, var_name in enumerate(var_names):
        col_vals = vals[valid, var_i]
        if col_vals.size == 0:
            break
        v_min = float(np.min(col_vals))
        v_max = float(np.max(col_vals))
        n_var_bins = bins_per_var[var_name]
        if v_max > v_min:
            idx = np.floor((col_vals - v_min) / (v_max - v_min) * n_var_bins)
            bin_idx[valid, var_i] = np.clip(idx, 0, n_var_bins - 1).astype(np.int64)

    if keep_mode == "random":
        order = np.random.default_rng(rnd_seed).permutation(n_smpls)
    else:
        order = np.arange(n_smpls)
    order = order[valid[order]]
    cand_df = pd.DataFrame(
        bin_idx[order], columns=[f"b{i}" for i in range(len(var_names))]
    )
    cand_df["pos"] = order
    kept_pos = np.sort(
        cand_df.drop_duplicates(
            subset=[c for c in cand_df.columns if c != "pos"], keep="first"
        )["pos"].to_numpy()
    )
    thin_gdf = smpls_gdf.iloc[kept_pos].copy()
    return write_model_vector(
        thin_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def combine_presence_absence_data(
    presence_smpls_vec_file: VectorInput,
    presence_smpls_vec_lyr: Optional[str],
    absence_smpls_vec_file: VectorInput,
    absence_smpls_vec_lyr: Optional[str],
    env_vars: Mapping[str, EnvVarInfo],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    equalise_smpls: bool = False,
    cls_col: str = "clsid",
    rnd_seed: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[List[str], Tuple[gpd.GeoDataFrame, List[str]]]:
    """
    Combine presence (``1``) and absence (``0``) sample layers into a single
    modelling dataset with consistent covariate columns and unique indices.

    :param presence_smpls_vec_file: Presence sample vector path or GeoDataFrame.
    :param presence_smpls_vec_lyr: Presence layer name.
    :param absence_smpls_vec_file: Absence sample vector path or GeoDataFrame.
    :param absence_smpls_vec_lyr: Absence layer name.
    :param env_vars: Mapping of environmental variable descriptors.
    :param out_vec_file: Output vector file path, or ``None`` to return
                         ``(combined_gdf, analysis_vars)`` in memory.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param equalise_smpls: Subsample the larger class to match the smaller class
                           count (Default = ``False``).
    :param cls_col: Output binary class column name (Default = ``'clsid'``).
    :param rnd_seed: Optional random seed used when ``equalise_smpls=True``.
    :param out_compress: Optional GeoParquet compression setting.
    :return: Ordered list of ``analysis_vars`` when ``out_vec_file`` is given,
             or ``(combined_gdf, analysis_vars)`` when ``out_vec_file`` is ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        vars_list = sdm.combine_presence_absence_data(
            "pres.gpkg", "pres", "abs.gpkg", "abs", env_vars, "combined.gpkg", "combined"
        )
    """
    presence_gdf = read_model_vector(
        presence_smpls_vec_file, vec_lyr=presence_smpls_vec_lyr
    )
    absence_gdf = read_model_vector(
        absence_smpls_vec_file, vec_lyr=absence_smpls_vec_lyr
    )

    if presence_gdf.crs is not None and absence_gdf.crs is not None:
        if presence_gdf.crs != absence_gdf.crs:
            absence_gdf = absence_gdf.to_crs(presence_gdf.crs)
    elif presence_gdf.crs is not None or absence_gdf.crs is not None:
        raise rsgispy.RSGISPyException(
            "Cannot combine presence and absence datasets when only one has "
            "a CRS defined."
        )

    analysis_vars = [
        info.name if info.name is not None else str(key)
        for key, info in env_vars.items()
    ]
    validate_columns_exist(presence_gdf, analysis_vars)
    validate_columns_exist(absence_gdf, analysis_vars)

    pres_geom_col = presence_gdf.geometry.name
    abs_geom_col = absence_gdf.geometry.name
    presence_gdf = presence_gdf[analysis_vars + [pres_geom_col]].copy()
    absence_gdf = absence_gdf[analysis_vars + [abs_geom_col]].copy()
    if abs_geom_col != pres_geom_col:
        absence_gdf = absence_gdf.rename_geometry(pres_geom_col)

    presence_gdf[cls_col] = np.ones(len(presence_gdf), dtype=int)
    absence_gdf[cls_col] = np.zeros(len(absence_gdf), dtype=int)

    if equalise_smpls:
        n_pres = len(presence_gdf)
        n_abs = len(absence_gdf)
        if n_pres > n_abs:
            presence_gdf = presence_gdf.sample(
                n=n_abs, random_state=rnd_seed, axis=0
            )
        elif n_abs > n_pres:
            absence_gdf = absence_gdf.sample(
                n=n_pres, random_state=rnd_seed, axis=0
            )

    data_gdf = gpd.GeoDataFrame(
        pd.concat([presence_gdf, absence_gdf], ignore_index=True),
        geometry=pres_geom_col,
        crs=presence_gdf.crs,
    )

    if out_vec_file is None:
        return data_gdf, analysis_vars

    write_model_vector(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )
    return analysis_vars


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def create_train_test_sets(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    train_vec_file: Optional[Union[str, os.PathLike]] = None,
    train_vec_lyr: Optional[str] = None,
    test_vec_file: Optional[Union[str, os.PathLike]] = None,
    test_vec_lyr: Optional[str] = None,
    split_prop: float = 0.8,
    cls_col: str = "clsid",
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    """
    Stratified split of combined presence (``1``) and absence (``0``) samples
    into training and testing sets preserving class proportions.

    :param vec_file: Combined vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param train_vec_file: Output training file path (or ``None`` for in-memory).
    :param train_vec_lyr: Output training layer name.
    :param test_vec_file: Output testing file path (or ``None`` for in-memory).
    :param test_vec_lyr: Output testing layer name.
    :param split_prop: Fraction of each class assigned to training (in ``(0, 1)``).
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param out_format: Output vector driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param out_compress: Optional GeoParquet compression setting.
    :return: ``(train_gdf, test_gdf)`` when both output file paths are ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.create_train_test_sets(
            "combined.gpkg", "combined",
            "train.gpkg", "train",
            "test.gpkg", "test",
            split_prop=0.8, rnd_seed=42,
        )
    """
    if not 0.0 < float(split_prop) < 1.0:
        raise rsgispy.RSGISPyException(
            f"split_prop must be strictly between 0 and 1, got {split_prop!r}."
        )

    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr).reset_index(drop=True)
    validate_columns_exist(data_gdf, [cls_col])

    presence_gdf = data_gdf[data_gdf[cls_col] == 1]
    absence_gdf = data_gdf[data_gdf[cls_col] == 0]

    n_train_pres = (
        max(1, int(len(presence_gdf) * split_prop))
        if len(presence_gdf) > 1
        else len(presence_gdf)
    )
    n_train_abs = (
        max(1, int(len(absence_gdf) * split_prop))
        if len(absence_gdf) > 1
        else len(absence_gdf)
    )

    train_pres_gdf = presence_gdf.sample(n=n_train_pres, random_state=rnd_seed)
    test_pres_gdf = presence_gdf.drop(train_pres_gdf.index)

    train_abs_gdf = absence_gdf.sample(n=n_train_abs, random_state=rnd_seed)
    test_abs_gdf = absence_gdf.drop(train_abs_gdf.index)

    geom_col = data_gdf.geometry.name
    train_data_gdf = gpd.GeoDataFrame(
        pd.concat([train_pres_gdf, train_abs_gdf], ignore_index=True),
        geometry=geom_col,
        crs=data_gdf.crs,
    )
    test_data_gdf = gpd.GeoDataFrame(
        pd.concat([test_pres_gdf, test_abs_gdf], ignore_index=True),
        geometry=geom_col,
        crs=data_gdf.crs,
    )

    if train_vec_file is None and test_vec_file is None:
        return train_data_gdf, test_data_gdf

    if train_vec_file is not None:
        write_model_vector(
            train_data_gdf,
            out_vec_file=train_vec_file,
            out_vec_lyr=train_vec_lyr,
            out_format=out_format,
            overwrite=True,
            out_compress=out_compress,
        )
    if test_vec_file is not None:
        write_model_vector(
            test_data_gdf,
            out_vec_file=test_vec_file,
            out_vec_lyr=test_vec_lyr,
            out_format=out_format,
            overwrite=True,
            out_compress=out_compress,
        )
    return None


def pop_normalise_coeffs(
    env_vars: Mapping[str, EnvVarInfo],
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
) -> Mapping[str, EnvVarInfo]:
    """
    Populate mean and standard deviation normalisation parameters in ``env_vars``
    for all continuous covariates from ``vec_file``.

    Zero-variance continuous covariates safely receive ``norm_std = 1.0``.

    :param env_vars: Mapping of variable name to :class:`EnvVarInfo`.
    :param vec_file: Input sample vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :return: Updated ``env_vars`` mapping.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pop_normalise_coeffs(env_vars, "combined.gpkg", "combined")
    """
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)

    cont_cols: List[str] = []
    for var_key, info in env_vars.items():
        if info.data_type == rsgispy.VAR_TYPE_CONTINUOUS:
            col_name = info.name if info.name is not None else str(var_key)
            cont_cols.append(col_name)

    if cont_cols:
        validate_columns_exist(data_gdf, cont_cols)
        num_df = data_gdf[cont_cols].apply(pd.to_numeric, errors="coerce")
        means = num_df.mean()
        stds = num_df.std()
    else:
        means = pd.Series(dtype=float)
        stds = pd.Series(dtype=float)

    for var_key, info in env_vars.items():
        if info.data_type == rsgispy.VAR_TYPE_CONTINUOUS:
            col_name = info.name if info.name is not None else str(var_key)
            mean_v = float(means[col_name])
            std_v = float(stds[col_name])
            if not np.isfinite(mean_v):
                mean_v = 0.0
            if not np.isfinite(std_v) or std_v == 0.0:
                std_v = 1.0
            info.set_norm_mean_std(mean_v, std_v)
        else:
            info.set_norm_mean_std(0.0, 1.0)

    return env_vars


# pylint: disable=too-many-arguments,too-many-positional-arguments
def apply_normalise_coeffs(
    env_vars: Mapping[str, EnvVarInfo],
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Z-score normalise continuous variable columns in ``vec_file`` using mean
    and standard deviation stored on ``env_vars``.

    :param env_vars: Mapping of :class:`EnvVarInfo` populated with normalisation coefficients.
    :param vec_file: Input vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_vec_file: Output vector file path (or ``None`` to return GeoDataFrame).
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Normalised :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        norm_gdf = sdm.apply_normalise_coeffs(env_vars, "train.gpkg", "train")
    """
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)

    for var_key, info in env_vars.items():
        if (
            info.data_type == rsgispy.VAR_TYPE_CONTINUOUS
            and info.norm_mean is not None
            and info.norm_std is not None
        ):
            col_name = info.name if info.name is not None else str(var_key)
            validate_columns_exist(data_gdf, [col_name])
            data_gdf[col_name] = normalise_array_with_info(
                data_gdf[col_name].astype(float), info
            )

    return write_model_vector(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def comparison_box_plots(
    env_vars: Mapping[str, EnvVarInfo],
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    out_dir: str = "boxplots",
    cls_col: str = "clsid",
    box_plots: bool = False,
    use_pbar: bool = True,
) -> List[str]:
    """
    Create distribution comparison plots (violin or box plots) grouped by
    ``cls_col`` for every covariate in ``env_vars``.

    :param env_vars: Mapping of environmental variable descriptors.
    :param vec_file: Input combined vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_dir: Output directory where ``<var>_comp_plot.png`` images are saved.
    :param cls_col: Class column name (Default = ``'clsid'``).
    :param box_plots: If ``True``, render box plots; if ``False``, violin plots.
    :param use_pbar: Display a progress bar over the covariates (Default = ``True``).
    :return: List of the plot files written (one per covariate; each figure is
             closed after it is saved).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.comparison_box_plots(env_vars, "combined.gpkg", "combined", out_dir="plots")
    """
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)
    analysis_vars = resolve_analysis_vars(env_vars)
    validate_columns_exist(data_gdf, [*analysis_vars, cls_col])
    return plot_class_feature_distributions(
        data_df=data_gdf,
        feature_cols=analysis_vars,
        cls_col=cls_col,
        out_dir=out_dir,
        box_plots=box_plots,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def correlation_matrix(
    env_vars: Mapping[str, EnvVarInfo],
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    out_corr_file: Optional[str] = "correlation_matrix.csv",
    out_plt_file: Optional[str] = "correlation_matrix.png",
    fig_width: int = 15,
    fig_height: int = 14,
) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlations among environmental covariates,
    saving the matrix CSV and an optional lower-triangular heatmap plot.

    :param env_vars: Mapping of environmental variable descriptors.
    :param vec_file: Input vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_corr_file: Output correlation matrix CSV path (or ``None``).
    :param out_plt_file: Output heatmap PNG path (or ``None`` to skip plotting).
    :param fig_width: Figure width in inches (Default = ``15``).
    :param fig_height: Figure height in inches (Default = ``14``).
    :return: Correlation matrix :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        corr_df = sdm.correlation_matrix(env_vars, "combined.gpkg", "combined")
    """
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)
    analysis_vars = resolve_analysis_vars(env_vars)
    validate_columns_exist(data_gdf, analysis_vars)
    return run_correlation_matrix(
        df=data_gdf,
        cols=analysis_vars,
        method="pearson",
        out_corr_file=out_corr_file,
        out_plt_file=out_plt_file,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def calc_vif_multicollinearity(
    env_vars: Union[Sequence[str], Mapping[str, EnvVarInfo]],
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
    out_vif_file: Optional[str] = "vif_multicollinearity.csv",
) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) across predictor variables using
    :func:`rsgispy.tools.featureselection.calc_pandas_vif`.

    :param env_vars: Sequence of predictor column names or mapping of :class:`EnvVarInfo`.
    :param vec_file: Input vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_vif_file: Output VIF CSV path (or ``None`` to skip saving).
    :return: :class:`pandas.Series` indexed by variable name with VIF values.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        vif = sdm.calc_vif_multicollinearity(["b1", "b2", "b3"], "combined.gpkg")
    """
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)
    var_names = resolve_analysis_vars(env_vars)
    validate_columns_exist(data_gdf, var_names)

    vif_scores = calc_pandas_vif(data_gdf[var_names])
    if out_vif_file is not None:
        vif_path = rsgispy.tools.files.ensure_parent_dir(out_vif_file)
        vif_scores.to_csv(vif_path, index=True)
    return vif_scores
