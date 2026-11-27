"""
The modelling.species_distribution module provides pure-Python species
distribution modelling workflows built on ``scikit-learn``, ``geopandas``,
``numpy``, ``pandas``, and ``rasterio`` via :mod:`rsgispy.imageprocess`.
"""

from rsgispy.modelling._helpers import (
    EnvVarInfo,
    calc_binary_roc_auc,
    calc_binary_threshold,
    calc_single_class_predictions,
    get_positive_class_probs,
    plot_roc_curve,
    predict_raster_from_env_vars,
)
from rsgispy.modelling._sdm_bias import calc_bias_weights, create_bias_grid
from rsgispy.modelling._sdm_data import (
    apply_normalise_coeffs,
    calc_vif_multicollinearity,
    combine_presence_absence_data,
    comparison_box_plots,
    correlation_matrix,
    create_finite_mask,
    create_train_test_sets,
    extract_env_var_data,
    gen_pseudo_absences_smpls,
    pop_normalise_coeffs,
    thin_pts_env_space,
)
from rsgispy.modelling._sdm_models import _ensure_all_classes_in_fold  # noqa: F401
from rsgispy.modelling._sdm_models import (
    fit_kfold_sklearn_mdls,
    fit_kfold_sklearn_sgl_cls_mdls,
    fit_sklearn_mdl,
    fit_sklearn_sgl_cls_mdl,
    salib_sklearn_mdl_sensitivity,
    search_mdl_params,
    shap_sklearn_mdl_explainer,
    sklearn_jackknife_var_importance,
    sklearn_mdl_variable_response_curves,
)
from rsgispy.modelling._sdm_predict import (
    pred_ensemble_sklearn_mdls_cls,
    pred_ensemble_sklearn_mdls_prob,
    pred_ensemble_sklearn_sgl_cls_mdls_cls,
    pred_ensemble_sklearn_sgl_cls_mdls_prob,
    pred_sklearn_mdl_cls,
    pred_sklearn_mdl_prob,
    pred_sklearn_sgl_cls_mdl_cls,
    pred_sklearn_sgl_cls_mdl_prob,
)

__all__ = [
    "EnvVarInfo",
    "create_finite_mask",
    "create_bias_grid",
    "gen_pseudo_absences_smpls",
    "extract_env_var_data",
    "thin_pts_env_space",
    "calc_bias_weights",
    "combine_presence_absence_data",
    "create_train_test_sets",
    "pop_normalise_coeffs",
    "apply_normalise_coeffs",
    "comparison_box_plots",
    "correlation_matrix",
    "calc_vif_multicollinearity",
    "search_mdl_params",
    "fit_sklearn_mdl",
    "fit_sklearn_sgl_cls_mdl",
    "fit_kfold_sklearn_mdls",
    "fit_kfold_sklearn_sgl_cls_mdls",
    "get_positive_class_probs",
    "calc_single_class_predictions",
    "calc_binary_roc_auc",
    "calc_binary_threshold",
    "plot_roc_curve",
    "shap_sklearn_mdl_explainer",
    "salib_sklearn_mdl_sensitivity",
    "sklearn_mdl_variable_response_curves",
    "sklearn_jackknife_var_importance",
    "pred_sklearn_mdl_prob",
    "pred_sklearn_mdl_cls",
    "pred_sklearn_sgl_cls_mdl_prob",
    "pred_sklearn_sgl_cls_mdl_cls",
    "pred_ensemble_sklearn_mdls_prob",
    "pred_ensemble_sklearn_mdls_cls",
    "pred_ensemble_sklearn_sgl_cls_mdls_prob",
    "pred_ensemble_sklearn_sgl_cls_mdls_cls",
    "predict_raster_from_env_vars",
]
