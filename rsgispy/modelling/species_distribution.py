# pylint: disable=too-many-lines
"""
The modelling.species_distribution module provides pure-Python species
distribution modelling workflows built on ``scikit-learn``, ``geopandas``,
``numpy``, ``pandas``, and ``rasterio`` via :mod:`rsgispy.imageprocess`.
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageutils.metadata import (
    get_epsg_proj_from_img,
    get_img_bbox,
    get_wkt_proj_from_img,
)
from rsgispy.modelling._helper import (
    _MASK_BAND_KEY,
    EnvVarInfo,
    calc_binary_roc_auc,
    calc_single_class_predictions,
    check_sklearn_available,
    extract_features_and_target,
    finalise_plot,
    fit_clean_estimator,
    get_positive_class_probs,
    normalise_array_with_info,
    plot_roc_curve,
    predict_raster_from_env_vars,
    read_model_vector,
    resolve_analysis_vars,
    run_env_block_pipeline,
    write_model_vector,
)
from rsgispy.tools.featureselection import calc_pandas_vif
from rsgispy.vectorgeoms.create import create_random_pts_in_bbox
from rsgispy.vectorprocess import VectorInput, validate_columns_exist
from rsgispy.vectorstats.metrics import calc_near_dist_to_feats
from rsgispy.vectorutils.selection import vec_lyr_random_subset
from rsgispy.zonalstats.point import ext_point_band_values

try:
    import seaborn as sns  # pylint: disable=import-error

    HAVE_SEABORN = True
except ImportError:
    sns = None  # type: ignore[assignment]
    HAVE_SEABORN = False

try:
    import shap  # pylint: disable=import-error

    HAVE_SHAP = True
except ImportError:
    shap = None  # type: ignore[assignment]
    HAVE_SHAP = False

try:
    import SALib.analyze.sobol  # pylint: disable=import-error
    import SALib.sample.sobol  # pylint: disable=import-error

    HAVE_SALIB = True
except ImportError:
    SALib = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    HAVE_SALIB = False

try:
    import sklearn.metrics  # pylint: disable=import-error

    HAVE_SKLEARN_METRICS = True
except ImportError:
    sklearn = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    HAVE_SKLEARN_METRICS = False

__all__ = [
    "EnvVarInfo",
    "create_finite_mask",
    "gen_pseudo_absences_smpls",
    "extract_env_var_data",
    "combine_presence_absence_data",
    "create_train_test_sets",
    "pop_normalise_coeffs",
    "apply_normalise_coeffs",
    "comparison_box_plots",
    "correlation_matrix",
    "calc_vif_multicollinearity",
    "search_mdl_params",
    "fit_sklearn_mdl",
    "fit_sklearn_slg_cls_mdl",
    "fit_kfold_sklearn_mdls",
    "fit_kfold_sklearn_sgl_cls_mdls",
    "shap_sklearn_mdl_explainer",
    "salib_sklearn_mdl_sensitivity",
    "sklearn_mdl_variable_response_curves",
    "sklearn_jackknife_var_importance",
    "pred_sklearn_mdl_prob",
    "pred_sklearn_mdl_cls",
    "pred_sklearn_slg_cls_mdl_prob",
    "pred_sklearn_slg_cls_mdl_cls",
    "pred_ensemble_sklearn_mdls_prob",
    "pred_ensemble_sklearn_mdls_cls",
    "pred_ensemble_sklearn_slg_cls_mdls_prob",
    "pred_ensemble_sklearn_sgl_cls_mdls_cls",
]


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_finite_mask(
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,  # pylint: disable=unused-argument
    use_pbar: bool = True,
    calc_img_stats: bool = True,
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
    :param calc_img_stats: Populate thematic statistics and pyramids (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.create_finite_mask(
            "roi_mask.tif", 1, env_vars, ["temp", "precip"], "valid_mask.tif"
        )
    """

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
            info = env_vars[var_name]
            assert info.file is not None
            arr = block_vars[var_keys[var_name]]
            valid &= np.isfinite(arr)
            nd_val = datasets[os.fspath(info.file)].nodatavals[info.band - 1]
            if nd_val is not None and not (
                isinstance(nd_val, float) and np.isnan(nd_val)
            ):
                valid &= arr != float(nd_val)
        return valid.astype(np.uint8)

    run_env_block_pipeline(
        in_msk_img=in_msk_img,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        block_eval_fn=_eval_mask_block,
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        calc_img_stats=calc_img_stats,
        thematic=True,
        no_data_val=0.0,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate pseudo-absence point samples within ``in_msk_img == img_msk_val``,
    optionally excluding points within ``presence_smpls_dist_thres`` of
    known presence points.

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
    assert pts_gdf is not None

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
    pts_gdf = pts_gdf[pts_gdf[tmp_col] == img_msk_val].drop(columns=[tmp_col]).copy()
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
            assert dist_gdf is not None
            dist_gdf = dist_gdf[~dist_gdf.index.duplicated(keep="first")]
            dist_gdf = dist_gdf[
                dist_gdf["__presence_dist__"] > float(presence_smpls_dist_thres)
            ]
            pts_gdf = dist_gdf[orig_cols].copy()
            if pts_gdf.empty:
                raise rsgispy.RSGISPyException(
                    "All candidate pseudo-absence samples were within "
                    "presence_smpls_dist_thres of presence points."
                )

    if len(pts_gdf) > n_smpls:
        pts_gdf = vec_lyr_random_subset(
            vec_file=pts_gdf,
            vec_lyr=None,
            out_vec_file=None,
            n_smpl=int(n_smpls),
            rnd_seed=rnd_seed,
        )
        assert pts_gdf is not None

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
    replace: bool = False,  # pylint: disable=unused-argument
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
            assert info.name is not None
            vec_gdf[info.name] = col_vals

    return write_model_vector(
        vec_gdf,
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
) -> None:
    """
    Create distribution comparison plots (violin or box plots) grouped by
    ``cls_col`` for every covariate in ``env_vars``.

    :param env_vars: Mapping of environmental variable descriptors.
    :param vec_file: Input combined vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_dir: Output directory where ``<var>_comp_plot.png`` images are saved.
    :param cls_col: Class column name (Default = ``'clsid'``).
    :param box_plots: If ``True``, render box plots; if ``False``, violin plots.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.comparison_box_plots(env_vars, "combined.gpkg", "combined", out_dir="plots")
    """
    os.makedirs(out_dir, exist_ok=True)
    data_gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(data_gdf, [cls_col])

    pbar = rsgispy.get_tqdm()
    for var_key in pbar(env_vars):
        info = env_vars[var_key]
        col_name = info.name if info.name is not None else str(var_key)
        validate_columns_exist(data_gdf, [col_name])

        out_path = os.path.join(out_dir, f"{col_name}_comp_plot.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            if HAVE_SEABORN:
                if box_plots:
                    sns.boxplot(data=data_gdf, x=cls_col, y=col_name, ax=ax)
                else:
                    sns.violinplot(data=data_gdf, x=cls_col, y=col_name, ax=ax)
            else:
                groups = sorted(data_gdf[cls_col].dropna().unique().tolist())
                series_list = [
                    data_gdf.loc[data_gdf[cls_col] == g, col_name].dropna().to_numpy()
                    for g in groups
                ]
                if box_plots:
                    ax.boxplot(series_list, tick_labels=[str(g) for g in groups])
                else:
                    ax.violinplot(series_list)
                    ax.set_xticks(range(1, len(groups) + 1))
                    ax.set_xticklabels([str(g) for g in groups])
                ax.set_xlabel(cls_col)
                ax.set_ylabel(col_name)
        finally:
            finalise_plot(fig, out_file=out_path)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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

    corr_df = data_gdf[analysis_vars].corr()
    if out_corr_file is not None:
        corr_dir = os.path.dirname(os.path.abspath(os.fspath(out_corr_file)))
        if corr_dir:
            os.makedirs(corr_dir, exist_ok=True)
        corr_df.to_csv(out_corr_file, index=True)

    if out_plt_file is not None:
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        try:
            if HAVE_SEABORN:
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(
                    corr_df,
                    mask=mask,
                    cmap=cmap,
                    vmax=0.3,
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.5},
                    ax=ax,
                )
            else:
                masked_vals = np.ma.array(corr_df.to_numpy(dtype=float), mask=mask)
                im = ax.imshow(masked_vals, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                fig.colorbar(im, ax=ax, shrink=0.5)
                ticks = range(len(analysis_vars))
                ax.set_xticks(ticks)
                ax.set_xticklabels(analysis_vars, rotation=45, ha="right")
                ax.set_yticks(ticks)
                ax.set_yticklabels(analysis_vars)
        finally:
            finalise_plot(fig, out_file=out_plt_file)

    return corr_df


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
        vif_dir = os.path.dirname(os.path.abspath(os.fspath(out_vif_file)))
        if vif_dir:
            os.makedirs(vif_dir, exist_ok=True)
        vif_scores.to_csv(out_vif_file, index=True)
    return vif_scores


def search_mdl_params(
    search_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute a cloned ``scikit-learn`` hyperparameter search object (e.g.,
    ``GridSearchCV`` or ``RandomizedSearchCV``) without mutating stale state.

    :param search_obj: Unfitted ``scikit-learn`` SearchCV estimator.
    :param train_vec_file: Training sample vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :return: Tuple ``(best_estimator, best_params)``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import rsgispy.modelling.species_distribution as sdm

        search = GridSearchCV(RandomForestClassifier(), {"n_estimators": [20, 50]})
        best_est, best_params = sdm.search_mdl_params(
            search, train_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("search_mdl_params")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )

    clean_search = fit_clean_estimator(
        search_obj, train_x, train_y, sync_input=False
    )
    if not getattr(clean_search, "refit", False):
        raise rsgispy.RSGISPyException(
            "Search did not find a refitted model and failed."
        )

    print(
        f"Best score was {clean_search.best_score_} and has parameters "
        f"{clean_search.best_params_}."
    )
    return clean_search.best_estimator_, clean_search.best_params_


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_sklearn_mdl(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
    roc_curve_plot: Optional[str] = None,
    return_model: bool = False,
) -> Union[Tuple[float, float, float], Tuple[Any, float, float, float]]:
    """
    Fit a ``scikit-learn`` binary classifier on a clean cloned instance so no
    state leaks across repeated model-comparison loops, returning training
    accuracy, testing accuracy, and ROC AUC (and optionally the fitted model).

    :param est_cls_obj: ``scikit-learn`` classifier estimator template.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or :class:`geopandas.GeoDataFrame`.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param roc_curve_plot: Optional output path for the ROC curve PNG.
    :param return_model: When ``True``, prepend the cleanly cloned fitted
                         estimator to the returned tuple (without mutating
                         ``est_cls_obj``). When ``False`` (default),
                         ``est_cls_obj`` is updated cleanly with the fitted
                         clone's state and ``(train_acc, test_acc, roc_auc)``
                         is returned.
    :return: ``(train_acc, test_acc, roc_auc)`` or
             ``(fitted_model, train_acc, test_acc, roc_auc)``.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.modelling.species_distribution as sdm

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        train_acc, test_acc, auc_val = sdm.fit_sklearn_mdl(
            clf, train_gdf, None, test_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("fit_sklearn_mdl")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    assert train_y is not None and test_y is not None

    fitted_mdl = fit_clean_estimator(
        est_cls_obj, train_x, train_y, sync_input=not return_model
    )

    train_acc_val = float(fitted_mdl.score(train_x, train_y))
    print(f"Classifier Train Score = {round(train_acc_val * 100, 2)}%")

    test_acc_val = float(fitted_mdl.score(test_x, test_y))
    print(f"Classifier Test Score = {round(test_acc_val * 100, 2)}%")

    test_probs = get_positive_class_probs(fitted_mdl, test_x, pos_label=1)
    fpr, tpr, roc_auc = calc_binary_roc_auc(test_y, test_probs)

    plot_roc_curve(fpr, tpr, roc_auc, out_file=roc_curve_plot)

    if return_model:
        return fitted_mdl, train_acc_val, test_acc_val, roc_auc
    return train_acc_val, test_acc_val, roc_auc


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_sklearn_slg_cls_mdl(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    cls_col: str = "clsid",
    roc_curve_plot: Optional[str] = None,
    return_model: bool = False,
) -> Union[Tuple[float, float], Tuple[Any, float, float]]:
    """
    Fit a single-class / outlier ``scikit-learn`` estimator on a clean clone,
    returning testing accuracy and ROC AUC (and optionally the fitted model).

    :param est_cls_obj: Single-class ``scikit-learn`` estimator template.
    :param train_vec_file: Presence-only training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame with ``cls_col``.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param cls_col: Binary class column name in ``test_vec_file`` (Default = ``'clsid'``).
    :param roc_curve_plot: Optional output path for ROC curve PNG.
    :param return_model: When ``True``, return ``(fitted_model, test_acc, roc_auc)``
                         without mutating ``est_cls_obj``.
    :return: ``(test_acc, roc_auc)`` or ``(fitted_model, test_acc, roc_auc)``.

    .. code:: python

        from sklearn.ensemble import IsolationForest
        import rsgispy.modelling.species_distribution as sdm

        mdl = IsolationForest(random_state=42)
        test_acc, auc_val = sdm.fit_sklearn_slg_cls_mdl(
            mdl, pres_train_gdf, None, test_gdf, None, ["v1", "v2"]
        )
    """
    check_sklearn_available("fit_sklearn_slg_cls_mdl")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    assert test_y is not None

    fitted_mdl = fit_clean_estimator(
        est_cls_obj, train_x, train_y=None, sync_input=not return_model
    )

    test_pred = calc_single_class_predictions(fitted_mdl, test_x)
    test_acc_val = float(sklearn.metrics.accuracy_score(test_y, test_pred))
    print(f"Classifier Test Score = {round(test_acc_val * 100, 2)}%")

    test_scores = fitted_mdl.decision_function(test_x)
    fpr, tpr, roc_auc = calc_binary_roc_auc(test_y, test_scores)

    plot_roc_curve(fpr, tpr, roc_auc, out_file=roc_curve_plot)

    if return_model:
        return fitted_mdl, test_acc_val, roc_auc
    return test_acc_val, roc_auc


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_kfold_sklearn_mdls(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_kfolds: int = 10,
    fold_prop: float = 0.6,
    cls_col: str = "clsid",
    rnd_seed: Optional[int] = None,
    sel_replacement: bool = True,
    use_pbar: bool = True,
) -> List[Any]:
    """
    Fit an ensemble of cleanly cloned ``scikit-learn`` binary classifiers across
    ``n_kfolds`` random training subsets without mutating ``est_cls_obj`` or
    global NumPy random state.

    :param est_cls_obj: Base ``scikit-learn`` classifier estimator.
    :param train_vec_file: Training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_kfolds: Number of ensemble folds to fit (Default = ``10``).
    :param fold_prop: Proportion of training rows per fold in ``(0, 1]``.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param rnd_seed: Optional local random seed for reproducibility.
    :param sel_replacement: Sample training subsets with replacement (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of cleanly cloned and fitted ``scikit-learn`` estimators.

    .. code:: python

        from sklearn.ensemble import RandomForestClassifier
        import rsgispy.modelling.species_distribution as sdm

        mdls = sdm.fit_kfold_sklearn_mdls(
            RandomForestClassifier(n_estimators=20),
            train_gdf, None, test_gdf, None, ["v1", "v2"],
            n_kfolds=5, rnd_seed=42,
        )
    """
    check_sklearn_available("fit_kfold_sklearn_mdls")
    if n_kfolds < 1:
        raise rsgispy.RSGISPyException("n_kfolds must be >= 1.")
    if not 0.0 < float(fold_prop) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"fold_prop must be in (0, 1], got {fold_prop!r}."
        )

    rng = np.random.default_rng(rnd_seed)
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, train_y = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=cls_col
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    assert train_y is not None and test_y is not None

    n_train = train_y.shape[0]
    n_fold_size = max(1, int(n_train * fold_prop))
    unique_classes = np.unique(train_y)

    train_scores: List[float] = []
    test_scores: List[float] = []
    roc_scores: List[float] = []
    est_cls_objs: List[Any] = []

    pbar = rsgispy.get_tqdm()
    for _ in pbar(range(n_kfolds), disable=not use_pbar):
        train_idx = rng.choice(n_train, size=n_fold_size, replace=sel_replacement)
        # Ensure degenerate draws do not drop a class when multiple classes exist.
        if unique_classes.size > 1 and np.unique(train_y[train_idx]).size < unique_classes.size:
            per_class_idx = [
                rng.choice(np.flatnonzero(train_y == cls_val), size=1)[0]
                for cls_val in unique_classes
            ]
            train_idx[: len(per_class_idx)] = per_class_idx

        train_y_set = train_y[train_idx]
        train_x_set = train_x.iloc[train_idx]

        fold_seed = int(rng.integers(0, 2**31 - 1)) if rnd_seed is not None else None
        est_cls_fold = fit_clean_estimator(
            est_cls_obj,
            train_x_set,
            train_y_set,
            sync_input=False,
            random_state=fold_seed,
        )
        est_cls_objs.append(est_cls_fold)

        train_scores.append(float(est_cls_fold.score(train_x_set, train_y_set)))
        test_scores.append(float(est_cls_fold.score(test_x, test_y)))

        test_probs = get_positive_class_probs(est_cls_fold, test_x, pos_label=1)
        _, _, roc_auc = calc_binary_roc_auc(test_y, test_probs)
        roc_scores.append(roc_auc)

    print(
        f"ROC: {np.min(roc_scores)} - {np.median(roc_scores)} - {np.max(roc_scores)}"
    )
    print(
        f"Train: {np.min(train_scores)} - {np.median(train_scores)} - "
        f"{np.max(train_scores)}"
    )
    print(
        f"Test: {np.min(test_scores)} - {np.median(test_scores)} - "
        f"{np.max(test_scores)}"
    )

    return est_cls_objs


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_kfold_sklearn_sgl_cls_mdls(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_kfolds: int = 10,
    fold_prop: float = 0.6,
    cls_col: str = "clsid",
    rnd_seed: Optional[int] = None,
    sel_replacement: bool = True,
    use_pbar: bool = True,
) -> List[Any]:
    """
    Fit an ensemble of cleanly cloned single-class / outlier ``scikit-learn``
    estimators across ``n_kfolds`` random presence subsets.

    :param est_cls_obj: Base single-class ``scikit-learn`` estimator.
    :param train_vec_file: Presence-only training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame with ``cls_col``.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_kfolds: Number of ensemble folds (Default = ``10``).
    :param fold_prop: Proportion of training rows per fold in ``(0, 1]``.
    :param cls_col: Binary class column name in ``test_vec_file`` (Default = ``'clsid'``).
    :param rnd_seed: Optional local random seed for reproducibility.
    :param sel_replacement: Sample training subsets with replacement (Default = ``True``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: List of cleanly cloned and fitted ``scikit-learn`` estimators.

    .. code:: python

        from sklearn.ensemble import IsolationForest
        import rsgispy.modelling.species_distribution as sdm

        mdls = sdm.fit_kfold_sklearn_sgl_cls_mdls(
            IsolationForest(), pres_train_gdf, None, test_gdf, None,
            ["v1", "v2"], n_kfolds=5, rnd_seed=42,
        )
    """
    check_sklearn_available("fit_kfold_sklearn_sgl_cls_mdls")
    if n_kfolds < 1:
        raise rsgispy.RSGISPyException("n_kfolds must be >= 1.")
    if not 0.0 < float(fold_prop) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"fold_prop must be in (0, 1], got {fold_prop!r}."
        )

    rng = np.random.default_rng(rnd_seed)
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )
    test_x, test_y = extract_features_and_target(
        test_gdf, analysis_vars, cls_col=cls_col
    )
    assert test_y is not None

    n_train = train_x.shape[0]
    n_fold_size = max(1, int(n_train * fold_prop))

    test_scores: List[float] = []
    roc_scores: List[float] = []
    est_cls_objs: List[Any] = []

    pbar = rsgispy.get_tqdm()
    for _ in pbar(range(n_kfolds), disable=not use_pbar):
        train_idx = rng.choice(n_train, size=n_fold_size, replace=sel_replacement)
        train_x_set = train_x.iloc[train_idx]

        fold_seed = int(rng.integers(0, 2**31 - 1)) if rnd_seed is not None else None
        est_cls_fold = fit_clean_estimator(
            est_cls_obj,
            train_x_set,
            train_y=None,
            sync_input=False,
            random_state=fold_seed,
        )
        est_cls_objs.append(est_cls_fold)

        test_pred = calc_single_class_predictions(est_cls_fold, test_x)
        test_scores.append(float(sklearn.metrics.accuracy_score(test_y, test_pred)))

        test_probs = est_cls_fold.decision_function(test_x)
        _, _, roc_auc = calc_binary_roc_auc(test_y, test_probs)
        roc_scores.append(roc_auc)

    print(
        f"ROC: {np.min(roc_scores)} - {np.median(roc_scores)} - {np.max(roc_scores)}"
    )
    print(
        f"Test: {np.min(test_scores)} - {np.median(test_scores)} - "
        f"{np.max(test_scores)}"
    )

    return est_cls_objs


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def shap_sklearn_mdl_explainer(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    shap_summary_plot: Optional[str] = None,
    shap_heatmap_plot: Optional[str] = None,
    shap_scatter_plots_dir: Optional[str] = None,
    shap_depend_plots_dir: Optional[str] = None,
    subsample_n_smpls: Optional[int] = None,
    use_tree_explainer: bool = False,
    use_linear_explainer: bool = False,
) -> None:
    """
    Compute SHAP feature explanations and generate summary, heatmap, scatter,
    and partial dependence plots for a fitted ``scikit-learn`` model.

    Requires optional dependency ``shap``.

    :param est_cls_obj: Fitted ``scikit-learn`` estimator.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param shap_summary_plot: Optional output path for SHAP summary plot.
    :param shap_heatmap_plot: Optional output path for SHAP heatmap plot.
    :param shap_scatter_plots_dir: Optional directory for per-variable SHAP scatter plots.
    :param shap_depend_plots_dir: Optional directory for per-variable partial dependence plots.
    :param subsample_n_smpls: Optional subsample size for SHAP evaluation.
    :param use_tree_explainer: Use ``shap.TreeExplainer`` when ``True``.
    :param use_linear_explainer: Use ``shap.LinearExplainer`` when ``True``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.shap_sklearn_mdl_explainer(
            clf, train_gdf, None, ["v1", "v2"],
            shap_summary_plot="shap_summary.png", use_tree_explainer=True,
        )
    """
    rsgispy.check_optional_dep(
        HAVE_SHAP, "shap_sklearn_mdl_explainer", "shap", "modelling"
    )
    if use_tree_explainer and use_linear_explainer:
        raise rsgispy.RSGISPyException(
            "You can only use one explainer: use_tree_explainer and "
            "use_linear_explainer cannot both be True."
        )

    show_plots = rsgispy.is_notebook()
    if show_plots:
        shap.initjs()

    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    train_x, _ = extract_features_and_target(
        train_gdf, analysis_vars, cls_col=None
    )

    if subsample_n_smpls is not None and subsample_n_smpls > 10:
        train_x_smpl = shap.sample(train_x, min(subsample_n_smpls, len(train_x)))
    else:
        train_x_smpl = train_x

    if use_tree_explainer:
        explainer = shap.TreeExplainer(
            est_cls_obj, feature_names=list(analysis_vars)
        )
    elif use_linear_explainer:
        explainer = shap.LinearExplainer(
            est_cls_obj, train_x_smpl, feature_perturbation="correlation_dependent"
        )
    else:
        explainer = shap.KernelExplainer(est_cls_obj.predict_proba, train_x_smpl)

    shap_values = explainer(train_x_smpl)
    if use_tree_explainer and getattr(shap_values, "values", None) is not None:
        if shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, -1]

    # pylint: disable=unreachable
    if shap_summary_plot is not None:
        shap.summary_plot(shap_values, train_x_smpl, show=False)
        finalise_plot(plt.gcf(), out_file=shap_summary_plot)

    if shap_heatmap_plot is not None:
        shap.plots.heatmap(shap_values, show=False)
        finalise_plot(plt.gcf(), out_file=shap_heatmap_plot)

    if shap_scatter_plots_dir is not None:
        os.makedirs(shap_scatter_plots_dir, exist_ok=True)
        for name in analysis_vars:
            out_plot_file = os.path.join(
                shap_scatter_plots_dir, f"shap_scatter_{name}.png"
            )
            shap.plots.scatter(shap_values[:, name], show=False)
            finalise_plot(plt.gcf(), out_file=out_plot_file)

    if shap_depend_plots_dir is not None:
        os.makedirs(shap_depend_plots_dir, exist_ok=True)
        sample_ind = min(100, max(0, len(train_x_smpl) - 1))
        for name in analysis_vars:
            out_plot_file = os.path.join(
                shap_depend_plots_dir, f"shap_depends_{name}.png"
            )
            fig, _ = shap.partial_dependence_plot(
                name,
                est_cls_obj.predict_proba,
                train_x_smpl,
                model_expected_value=True,
                feature_expected_value=True,
                show=False,
                ice=False,
                shap_values=shap_values[sample_ind:sample_ind + 1, :],
            )
            finalise_plot(fig, out_file=out_plot_file)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def salib_sklearn_mdl_sensitivity(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_samples: int = 10000,
    sobol_plot_file: Optional[str] = None,
    sobol_overall_file: Optional[str] = None,
    sobol_first_file: Optional[str] = None,
    sobol_second_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform SALib Sobol sensitivity analysis to quantify first-order,
    second-order, and total-order feature sensitivities of ``est_cls_obj``.

    Requires optional dependency ``SALib``.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param n_samples: Base Saltelli sample count (Default = ``10000``).
    :param sobol_plot_file: Optional output Sobol sensitivity plot path.
    :param sobol_overall_file: Optional total-order Sobol CSV output path.
    :param sobol_first_file: Optional first-order Sobol CSV output path.
    :param sobol_second_file: Optional second-order Sobol CSV output path.
    :return: Tuple ``(total_Si, first_Si, second_Si)`` DataFrames.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        total_si, first_si, second_si = sdm.salib_sklearn_mdl_sensitivity(
            clf, train_gdf, None, ["v1", "v2"], n_samples=256
        )
    """
    rsgispy.check_optional_dep(
        HAVE_SALIB, "salib_sklearn_mdl_sensitivity", "SALib", "modelling"
    )
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    var_list = list(analysis_vars)
    validate_columns_exist(train_gdf, var_list)

    var_bounds: List[List[float]] = []
    for var in var_list:
        min_val = float(train_gdf[var].min())
        max_val = float(train_gdf[var].max())
        if max_val <= min_val:
            max_val = min_val + 1e-6
        var_bounds.append([min_val, max_val])
        print(f"{var} Range:\t({min_val} - {max_val})")

    problem = {
        "num_vars": len(var_list),
        "names": var_list,
        "bounds": var_bounds,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        param_values = SALib.sample.sobol.sample(problem, N=n_samples)
    env_param_val_df = pd.DataFrame(data=param_values, columns=var_list)

    pred_y = get_positive_class_probs(est_cls_obj, env_param_val_df, pos_label=1)
    si = SALib.analyze.sobol.analyze(
        problem, pred_y, calc_second_order=True, print_to_console=False
    )
    total_si, first_si, second_si = si.to_df()

    if sobol_overall_file is not None:
        total_si.to_csv(sobol_overall_file, index=False)
    if sobol_first_file is not None:
        first_si.to_csv(sobol_first_file, index=False)
    if sobol_second_file is not None:
        second_si.to_csv(sobol_second_file, index=False)

    if sobol_plot_file is not None:
        plt_width = ((len(var_list) * 0.4) * 3) + 2
        si.plot()
        fig = plt.gcf()
        fig.set_size_inches(plt_width, 10)
        fig.tight_layout()
        finalise_plot(fig, out_file=sobol_plot_file)

    return total_si, first_si, second_si


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def sklearn_mdl_variable_response_curves(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    output_file: Optional[str],
    response_plots_dir: Optional[str],
    n_samples: int = 1000,
    cls_col: str = "clsid",
    normalised_data: bool = False,
    env_vars: Optional[Mapping[str, EnvVarInfo]] = None,
) -> pd.DataFrame:
    """
    Generate marginal response curves for each environmental variable while
    holding other covariates at their mean presence value.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param train_vec_file: Training vector path or :class:`geopandas.GeoDataFrame`.
    :param train_vec_lyr: Training vector layer name.
    :param analysis_vars: Ordered predictor column names.
    :param output_file: Output CSV path for response curve data (or ``None``).
    :param response_plots_dir: Output directory for PNG plots (or ``None``).
    :param n_samples: Number of sample points along each variable range.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param normalised_data: Un-normalise X-axis values back to raw units using
                            ``env_vars`` when ``True``.
    :param env_vars: Mapping of :class:`EnvVarInfo` required if
                     ``normalised_data=True``.
    :return: :class:`pandas.DataFrame` containing ``<var>_vals`` and ``<var>_resp`` columns.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        df = sdm.sklearn_mdl_variable_response_curves(
            clf, train_gdf, None, ["v1", "v2"], "resp.csv", "resp_plots", n_samples=50
        )
    """
    if normalised_data and env_vars is None:
        raise rsgispy.RSGISPyException(
            "env_vars must be provided when normalised_data=True."
        )
    if response_plots_dir is not None:
        os.makedirs(response_plots_dir, exist_ok=True)

    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    var_list = list(analysis_vars)
    validate_columns_exist(train_gdf, var_list + [cls_col])
    train_pres_gdf = train_gdf[train_gdf[cls_col] == 1]
    if train_pres_gdf.empty:
        train_pres_gdf = train_gdf

    var_bounds: Dict[str, Tuple[float, float]] = {}
    var_avg: Dict[str, float] = {}
    for var in var_list:
        min_val = float(train_gdf[var].min())
        max_val = float(train_gdf[var].max())
        var_bounds[var] = (min_val, max_val)
        var_avg[var] = float(train_pres_gdf[var].mean())
        print(f"{var} Range:\t({min_val} - {max_val}): Mean = {var_avg[var]}")

    response_vals: Dict[str, np.ndarray] = {}
    for var in var_list:
        print(f"Calculating Response Curve for '{var}'")
        var_vals = np.linspace(
            var_bounds[var][0], var_bounds[var][1], num=n_samples, dtype=float
        )
        data_dict: Dict[str, np.ndarray] = {}
        for tmp_var in var_list:
            if tmp_var == var:
                data_dict[tmp_var] = var_vals
            else:
                data_dict[tmp_var] = np.full(n_samples, var_avg[tmp_var], dtype=float)

        env_param_val_df = pd.DataFrame(data=data_dict)
        env_resp_probs = get_positive_class_probs(
            est_cls_obj, env_param_val_df, pos_label=1
        )

        if normalised_data and env_vars is not None and var in env_vars:
            info = env_vars[var]
            if info.norm_mean is not None and info.norm_std is not None:
                var_vals = (var_vals * float(info.norm_std)) + float(info.norm_mean)

        response_vals[f"{var}_vals"] = var_vals
        response_vals[f"{var}_resp"] = env_resp_probs

        if response_plots_dir is not None:
            out_plot_file = os.path.join(
                response_plots_dir, f"{var}_response_curve.png"
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(var_vals, env_resp_probs)
            ax.set_ylim([0, 1])
            ax.set_title(f"{var} Response Curve")
            ax.set_xlabel(f"{var}")
            ax.set_ylabel("Class Probability")
            finalise_plot(fig, out_file=out_plot_file)

    response_vals_df = pd.DataFrame(data=response_vals)
    if output_file is not None:
        out_dir = os.path.dirname(os.path.abspath(os.fspath(output_file)))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        response_vals_df.to_csv(output_file, index=False)
    return response_vals_df


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def sklearn_jackknife_var_importance(
    est_cls_obj: Any,
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    test_vec_file: VectorInput,
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    out_var_importance_file: Optional[str],
    out_importance_auc_plot: Optional[str] = None,
    out_importance_test_plot: Optional[str] = None,
    cls_col: str = "clsid",
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Evaluate variable importance via jackknife leave-one-out and single-variable
    training runs using clean estimator clones in every iteration.

    :param est_cls_obj: Base ``scikit-learn`` classifier estimator.
    :param train_vec_file: Training vector path or GeoDataFrame.
    :param train_vec_lyr: Training vector layer name.
    :param test_vec_file: Testing vector path or GeoDataFrame.
    :param test_vec_lyr: Testing vector layer name.
    :param analysis_vars: Ordered predictor column names (at least 2).
    :param out_var_importance_file: Output CSV path for jackknife metrics (or ``None``).
    :param out_importance_auc_plot: Optional horizontal bar plot path for ROC AUC.
    :param out_importance_test_plot: Optional horizontal bar plot path for test accuracy.
    :param cls_col: Binary class column name (Default = ``'clsid'``).
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).
    :return: :class:`pandas.DataFrame` indexed by ``analysis_vars``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        imp_df = sdm.sklearn_jackknife_var_importance(
            clf, train_gdf, None, test_gdf, None, ["v1", "v2", "v3"], "jackknife.csv"
        )
    """
    check_sklearn_available("sklearn_jackknife_var_importance")
    var_list = list(analysis_vars)
    if len(var_list) < 2:
        raise rsgispy.RSGISPyException(
            "Jackknife variable importance requires at least 2 variables."
        )

    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)

    train_x_all, train_y = extract_features_and_target(
        train_gdf, var_list, cls_col=cls_col
    )
    test_x_all, test_y = extract_features_and_target(
        test_gdf, var_list, cls_col=cls_col
    )
    assert train_y is not None and test_y is not None

    def _eval_subset(sub_vars: List[str]) -> Tuple[float, float, float]:
        tr_sub = train_x_all[sub_vars]
        te_sub = test_x_all[sub_vars]
        est_sub = fit_clean_estimator(
            est_cls_obj, tr_sub, train_y, sync_input=False
        )
        tr_acc = float(est_sub.score(tr_sub, train_y))
        te_acc = float(est_sub.score(te_sub, test_y))
        te_probs = get_positive_class_probs(est_sub, te_sub, pos_label=1)
        _, _, roc_auc = calc_binary_roc_auc(test_y, te_probs)
        return tr_acc, te_acc, roc_auc

    _, all_test_acc_val, all_roc_auc = _eval_subset(var_list)

    out_data: Dict[str, List[float]] = {
        "train_acc_leave_var_out": [],
        "test_acc_leave_var_out": [],
        "roc_test_acc_leave_var_out": [],
        "train_acc_only_var": [],
        "test_acc_only_var": [],
        "roc_test_acc_only_var": [],
    }

    pbar = rsgispy.get_tqdm()
    for var in pbar(var_list, disable=not use_pbar):
        tr_acc, te_acc, roc_auc = _eval_subset([v for v in var_list if v != var])
        out_data["train_acc_leave_var_out"].append(tr_acc)
        out_data["test_acc_leave_var_out"].append(te_acc)
        out_data["roc_test_acc_leave_var_out"].append(roc_auc)

    for var in pbar(var_list, disable=not use_pbar):
        tr_acc, te_acc, roc_auc = _eval_subset([var])
        out_data["train_acc_only_var"].append(tr_acc)
        out_data["test_acc_only_var"].append(te_acc)
        out_data["roc_test_acc_only_var"].append(roc_auc)

    var_import_vals_df = pd.DataFrame(data=out_data, index=var_list)
    if out_var_importance_file is not None:
        out_dir = os.path.dirname(
            os.path.abspath(os.fspath(out_var_importance_file))
        )
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        var_import_vals_df.to_csv(out_var_importance_file, index=True)

    n_vars = len(var_list)
    var_idx = list(range(n_vars))
    plt_height = (0.5 * n_vars) + 1.5
    var_idx_all = var_idx + [n_vars]
    all_var_names = var_list + ["All"]

    def _plot_bars(
        leave_vals: List[float],
        only_vals: List[float],
        all_val: float,
        title: str,
        xlabel: str,
        out_plot: str,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        fig, ax = plt.subplots(figsize=(8, plt_height))
        try:
            ax.barh(var_idx, leave_vals, height=0.8, color="#1E88E5", label="Without Var")
            ax.barh(var_idx, only_vals, height=0.8, color="#FFC107", label="Only Var")
            ax.barh([n_vars], [all_val], height=0.8, color="#D81B60", label="All Vars")
            ax.set_yticks(var_idx_all, labels=all_var_names)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.11),
                fancybox=True,
                shadow=False,
                ncol=3,
            )
            fig.tight_layout()
        finally:
            finalise_plot(fig, out_file=out_plot)

    if out_importance_auc_plot is not None:
        _plot_bars(
            out_data["roc_test_acc_leave_var_out"],
            out_data["roc_test_acc_only_var"],
            all_roc_auc,
            "Jackknife AUC Gain",
            "Testing AUC",
            out_importance_auc_plot,
        )

    if out_importance_test_plot is not None:
        _plot_bars(
            out_data["test_acc_leave_var_out"],
            out_data["test_acc_only_var"],
            all_test_acc_val,
            "Jackknife Test Gain",
            "Testing Accuracy",
            out_importance_test_plot,
        )

    return var_import_vals_df


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_mdl_prob(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict spatial probability of species presence (``float32``, nodata ``-1.0``)
    from a fitted ``scikit-learn`` classifier using block-wise streaming via
    :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output single-band probability image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_mdl_prob(
            clf, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "prob.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: get_positive_class_probs(est_cls_obj, df, pos_label=1),
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=-1.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_mdl_cls(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict binary presence/absence thematic classification (``uint8``) from a
    fitted ``scikit-learn`` classifier using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output thematic classification image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Populate thematic statistics and colour table (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_mdl_cls(
            clf, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "cls.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=est_cls_obj.predict,
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        fill_val=0.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=True,
        no_data_val=0.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )


def _score_single_class_samples(est_cls_obj: Any, env_df: pd.DataFrame) -> np.ndarray:
    """Return continuous single-class score values (``score_samples`` or fallback)."""
    if hasattr(est_cls_obj, "score_samples"):
        return np.asarray(est_cls_obj.score_samples(env_df), dtype=np.float64)
    return np.asarray(est_cls_obj.decision_function(env_df), dtype=np.float64)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_slg_cls_mdl_prob(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    out_no_data_val: float = -9999.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict continuous habitat suitability/density scores from a fitted
    single-class ``scikit-learn`` estimator using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted single-class estimator (with ``score_samples`` or
                        ``decision_function``).
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` score image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param out_no_data_val: No-data fill value outside mask (Default = ``-9999.0``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_slg_cls_mdl_prob(
            oc_mdl, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_score.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: _score_single_class_samples(est_cls_obj, df),
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=float(out_no_data_val),
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=float(out_no_data_val),
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_sklearn_slg_cls_mdl_cls(
    est_cls_obj: Any,
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict binary presence (``1``) vs absence (``0``) thematic map from a
    fitted single-class ``scikit-learn`` estimator using :mod:`rsgispy.imageprocess`.

    :param est_cls_obj: Fitted single-class estimator with ``predict``.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output thematic classification image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Populate thematic statistics and colour table (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_sklearn_slg_cls_mdl_cls(
            oc_mdl, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_cls.tif"
        )
    """
    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=lambda df: calc_single_class_predictions(est_cls_obj, df),
        gdalformat=gdalformat,
        out_rio_dtype="uint8",
        fill_val=0.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=True,
        no_data_val=0.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_mdls_prob(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,  # pylint: disable=unused-argument
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict ensemble species presence probability across ``est_cls_objs`` by
    computing the median probability per pixel in a single streaming pass via
    :mod:`rsgispy.imageprocess` (writing no intermediate raster files).

    :param est_cls_objs: Sequence of fitted ``scikit-learn`` classifiers.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble probability image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_mdls_prob(
            mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "ens_prob.tif"
        )
    """
    if not est_cls_objs:
        raise rsgispy.RSGISPyException("est_cls_objs must not be empty.")

    def _ensemble_prob(env_df: pd.DataFrame) -> np.ndarray:
        stack = np.vstack(
            [get_positive_class_probs(mdl, env_df, pos_label=1) for mdl in est_cls_objs]
        )
        return np.median(stack, axis=0)

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_ensemble_prob,
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=-1.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_mdls_cls(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,  # pylint: disable=unused-argument
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict the proportion of ensemble classifiers voting presence (in ``[0, 1]``,
    nodata ``-1.0`` outside mask) in a single streaming pass via
    :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted ``scikit-learn`` classifiers.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble vote proportion image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_mdls_cls(
            mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "ens_cls.tif"
        )
    """
    if not est_cls_objs:
        raise rsgispy.RSGISPyException("est_cls_objs must not be empty.")

    def _ensemble_cls(env_df: pd.DataFrame) -> np.ndarray:
        stack = np.vstack(
            [
                (np.asarray(mdl.predict(env_df)) > 0).astype(np.float64)
                for mdl in est_cls_objs
            ]
        )
        return np.mean(stack, axis=0)

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_ensemble_cls,
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=-1.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_slg_cls_mdls_prob(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,  # pylint: disable=unused-argument
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    out_no_data_val: float = -9999.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict median suitability/density score across an ensemble of fitted
    single-class estimators in a single streaming pass via :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted single-class ``scikit-learn`` estimators.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble score image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param out_no_data_val: No-data fill value outside mask (Default = ``-9999.0``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_slg_cls_mdls_prob(
            oc_mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_ens_score.tif"
        )
    """
    if not est_cls_objs:
        raise rsgispy.RSGISPyException("est_cls_objs must not be empty.")

    def _ensemble_slg_prob(env_df: pd.DataFrame) -> np.ndarray:
        stack = np.vstack(
            [_score_single_class_samples(mdl, env_df) for mdl in est_cls_objs]
        )
        return np.median(stack, axis=0)

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_ensemble_slg_prob,
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=float(out_no_data_val),
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=float(out_no_data_val),
        block_size=block_size,
        use_pbar=use_pbar,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def pred_ensemble_sklearn_sgl_cls_mdls_cls(
    est_cls_objs: Sequence[Any],
    in_msk_img: str,
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    gdalformat: str = "GTIFF",
    tmp_dir: Optional[str] = None,  # pylint: disable=unused-argument
    normalise_data: bool = False,
    calc_img_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Predict the proportion of single-class ensemble models voting inlier
    (presence) per pixel (in ``[0, 1]``, nodata ``-1.0`` outside mask) in a
    single streaming pass via :mod:`rsgispy.imageprocess`.

    :param est_cls_objs: Sequence of fitted single-class ``scikit-learn`` estimators.
    :param in_msk_img: Path to ROI mask image.
    :param img_msk_val: Pixel value inside ``in_msk_img`` defining the ROI.
    :param env_vars: Mapping of environmental variable descriptors.
    :param analysis_vars: Ordered covariate names matching training columns.
    :param output_img: Output ``float32`` ensemble vote proportion image path.
    :param gdalformat: Output GDAL/rasterio format (Default = ``'GTIFF'``).
    :param tmp_dir: Unused legacy parameter retained for signature compatibility.
    :param normalise_data: Apply z-score normalisation from ``env_vars`` when ``True``.
    :param calc_img_stats: Compute output image statistics and pyramids (Default = ``True``).
    :param block_size: Optional block size in pixels for :mod:`rsgispy.imageprocess`.
    :param use_pbar: Display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.pred_ensemble_sklearn_sgl_cls_mdls_cls(
            oc_mdls, "valid_mask.tif", 1, env_vars, ["v1", "v2"], "oc_ens_cls.tif"
        )
    """
    if not est_cls_objs:
        raise rsgispy.RSGISPyException("est_cls_objs must not be empty.")

    def _ensemble_slg_cls(env_df: pd.DataFrame) -> np.ndarray:
        stack = np.vstack(
            [
                calc_single_class_predictions(mdl, env_df).astype(np.float64)
                for mdl in est_cls_objs
            ]
        )
        return np.mean(stack, axis=0)

    predict_raster_from_env_vars(
        in_msk_img=in_msk_img,
        img_msk_val=img_msk_val,
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=output_img,
        predict_fn=_ensemble_slg_cls,
        gdalformat=gdalformat,
        out_rio_dtype="float32",
        fill_val=-1.0,
        normalise_data=normalise_data,
        calc_img_stats=calc_img_stats,
        thematic=False,
        no_data_val=-1.0,
        block_size=block_size,
        use_pbar=use_pbar,
    )
