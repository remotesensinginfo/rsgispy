"""
API contract tests: the package and module exports (``__all__``), the
function signatures (required positional inputs, keyword-only arguments and
their defaults) and the literalinclude markers the worked-example guides
use. Each check is one parametrised case, so a failure names the function,
argument or guide it is about.
"""

import importlib
import inspect
import os

import pytest

DOCS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "docs")


def _obj(path):
    """Resolve ``"package.module:attr.sub"`` (or just a module) to the object."""
    mod_name, _, attr = path.partition(":")
    obj = importlib.import_module(mod_name)
    for part in filter(None, attr.split(".")):
        obj = getattr(obj, part)
    return obj


def _module_funcs(mod_name, keep):
    """``"module:name"`` for each name in the module ``__all__`` that ``keep`` accepts."""
    mod = importlib.import_module(mod_name)
    return [f"{mod_name}:{name}" for name in mod.__all__ if keep(name)]


# ---------------------------------------------------------------------------
# Worked-example guides: every literalinclude marker exists in the script
# ---------------------------------------------------------------------------
# Marker filters: "all" uses every ``:start-after:`` marker, "step" only the
# ``[step-`` markers, "no_tutorial_data" skips the ``[tutorial-data]`` marker
# (it is in the *_tutorial.py script) and "included" only the markers of the
# blocks included from the example script (the guide also includes the
# tutorial-data script).
GUIDE_EXAMPLES = [
    ("analysis_ready_data", "all", 8),
    ("change_detection", "step", 7),
    ("hydrology_workflow", "included", 10),
    ("image_coregistration", "no_tutorial_data", 9),
    ("large_rasters", "no_tutorial_data", 9),
    ("making_maps", "included", 9),
    ("multi_criteria_analysis", "all", 8),
    ("pixel_classification_workflow", "included", 12),
    ("pixel_regression_workflow", "all", 10),
    ("sar_processing", "all", 9),
    ("segment_ml_workflow", "no_tutorial_data", 12),
    ("species_distribution_modelling", "step", 11),
    ("spectral_unmixing", "no_tutorial_data", 9),
    ("tile_graph", "included", 11),
    ("zonal_stats_and_vectors", "all", 8),
]


def _guide_markers(guide, example_name, marker_filter):
    markers = []
    included = ""
    for line in guide.splitlines():
        if marker_filter == "included" and "literalinclude::" in line:
            included = line.split("literalinclude::")[1].strip()
            continue
        if ":start-after:" not in line:
            continue
        if marker_filter == "step" and "[step-" not in line:
            continue
        if marker_filter == "no_tutorial_data" and "tutorial-data" in line:
            continue
        if marker_filter == "included" and not included.endswith(f"/{example_name}.py"):
            continue
        markers.append(line.split(":start-after:")[1].strip())
    return markers


@pytest.mark.parametrize("example_name, marker_filter, min_markers", GUIDE_EXAMPLES)
def test_guide_markers_exist(example_name, marker_filter, min_markers):
    """Every literalinclude marker used by the guide exists in the script."""
    example = os.path.join(DOCS_DIR, "examples", f"{example_name}.py")
    guide_path = os.path.join(DOCS_DIR, "guides", f"{example_name}.rst")
    with open(example, encoding="utf-8") as f_obj:
        script = f_obj.read()
    with open(guide_path, encoding="utf-8") as f_obj:
        guide = f_obj.read()
    markers = _guide_markers(guide, example_name, marker_filter)
    assert len(markers) >= min_markers
    for marker in markers:
        assert marker in script
        assert marker.replace("[step-", "[end-step-") in script


# ---------------------------------------------------------------------------
# gdalformat defaults to 'GTIFF'
# ---------------------------------------------------------------------------
IMAGESPECTRAL_GTIFF_FUNCS = [
    "rsgispy.imagespectral.unmixing:spec_unmix_ucls",
    "rsgispy.imagespectral.unmixing:spec_unmix_nnls",
    "rsgispy.imagespectral.unmixing:spec_unmix_fcls",
    "rsgispy.imagespectral.unmixing:spec_unmix_spts_ucls",
    "rsgispy.imagespectral.unmixing:spec_unmix_spts_nnls",
    "rsgispy.imagespectral.unmixing:spec_unmix_spts_fcls",
    "rsgispy.imagespectral.unmixing:spec_unmix_pymcr_nnls",
    "rsgispy.imagespectral.unmixing:spec_unmix_pymcr_fcls",
    "rsgispy.imagespectral.unmixing:exhcon_linear_spec_unmix",
    "rsgispy.imagespectral.unmixing:rescale_unmixing_results",
    "rsgispy.imagespectral.unmixing:predict_refl_linear_unmixing",
    "rsgispy.imagespectral.unmixing:calc_unmixing_rmse_residual_err",
    "rsgispy.imagespectral.unmixing:summarise_multi_endmember_linear_unmixing",
    "rsgispy.imagespectral.endmembers:calc_ppi",
    "rsgispy.imagespectral.classification:spectral_angle_mapper",
    "rsgispy.imagespectral.classification:spectral_correlation_mapper",
    "rsgispy.imagespectral.preprocessing:spectral_smoothing",
    "rsgispy.imagespectral.preprocessing:mask_outliers_data_values",
    "rsgispy.imagespectral.preprocessing:polyfill_nan_data_values",
    "rsgispy.imagespectral.transforms:whiten_image",
    "rsgispy.imagespectral.transforms:calc_pca",
    "rsgispy.imagespectral.transforms:calc_mnf",
    "rsgispy.imagespectral.transforms:calc_tasseled_cap",
    "rsgispy.imagespectral.transforms:histogram_matching",
]

GTIFF_DEFAULT_FUNCS = [
    "rsgispy.changedetect.imgchngmthds:image_difference",
    "rsgispy.changedetect.imgchngmthds:image_ratio",
    "rsgispy.changedetect.imgchngmthds:change_vector_analysis",
    "rsgispy.changedetect.imgchngmthds:slow_feature_analysis",
    "rsgispy.changedetect.imgchngmthds:multivariate_alteration_detection",
    "rsgispy.changedetect.pxloutlierchng:find_class_pyod_outliers",
    "rsgispy.changedetect.pxloutlierchng:find_class_otsu_outliers",
    "rsgispy.changedetect.pxloutlierchng:find_class_li_outliers",
    "rsgispy.changedetect.pxloutlierchng:find_class_kurt_skew_outliers",
    "rsgispy.imagecalc.stats:image_pixel_column_summary",
    "rsgispy.imagecalc.stats:calc_multi_img_band_stats",
    "rsgispy.imagecalc.stats:get_img_sum_stats_in_pxl",
    "rsgispy.imagecalc.stats:calc_mask_img_pxl_val_prob",
    "rsgispy.imagecalc.comparison:calc_img_difference",
    "rsgispy.imagecalc.encoding:recode_int_raster",
    "rsgispy.imagecalc.encoding:one_hot_category_encoding",
    "rsgispy.imagecalc.rescale:calc_img_rescale",
    "rsgispy.imagecalc.rescale:rescale_img_pxl_vals",
    "rsgispy.imagecalc.distance:calc_dist_to_img_vals",
    "rsgispy.imagecalc.distance:calc_dist_to_img_vals_tiled",
    "rsgispy.imagecalc.distance:buffer_img_pxl_vals",
    "rsgispy.imagecalc.fill:calc_fill_regions_knn",
    "rsgispy.imagecalc.fill:region_growing_fill",
    "rsgispy.imagecalc.fill:region_growing_fill_tiled",
    "rsgispy.imagecalc.leastcostpath:perform_least_cost_path_calc",
] + IMAGESPECTRAL_GTIFF_FUNCS


@pytest.mark.parametrize("func", GTIFF_DEFAULT_FUNCS)
def test_default_gdalformat_is_gtiff(func):
    """gdalformat defaults to 'GTIFF'."""
    sig = inspect.signature(_obj(func))
    assert sig.parameters["gdalformat"].default == "GTIFF"


# ---------------------------------------------------------------------------
# Other argument defaults
# ---------------------------------------------------------------------------
_CHNG = "rsgispy.changedetect.imgchngmthds"
PARAM_DEFAULTS = [
    # CVA and ISFA centre on the threshold by default; IR-MAD does not.
    (f"{_CHNG}:change_vector_analysis", "centre_on_threshold", True),
    (f"{_CHNG}:slow_feature_analysis", "centre_on_threshold", True),
    (f"{_CHNG}:multivariate_alteration_detection", "centre_on_threshold", False),
    (f"{_CHNG}:change_vector_analysis", "no_data_val", -9999.0),
    (f"{_CHNG}:slow_feature_analysis", "no_data_val", -9999.0),
    (f"{_CHNG}:multivariate_alteration_detection", "no_data_val", -9999.0),
    # Every function that can use K-means takes rnd_seed (Default = 42).
    (f"{_CHNG}:change_vector_analysis", "rnd_seed", 42),
    (f"{_CHNG}:slow_feature_analysis", "rnd_seed", 42),
    (f"{_CHNG}:multivariate_alteration_detection", "rnd_seed", 42),
]


@pytest.mark.parametrize("func, param, expected", PARAM_DEFAULTS)
def test_param_default(func, param, expected):
    """The argument has the documented default (booleans compared by identity)."""
    default = inspect.signature(_obj(func)).parameters[param].default
    if isinstance(expected, bool):
        assert default is expected
    else:
        assert default == expected


# ---------------------------------------------------------------------------
# Raster functions: input_img/output_img first, keyword-only tail
# ---------------------------------------------------------------------------
KW_TAIL = {
    "gdalformat": "GTIFF",
    "datatype": None,
    "no_data_val": None,
    "calc_stats": True,
    "use_pbar": True,
}
BORDER_KWARGS = {"border_mode": "edge", "border_fill": 0.0}

_HYD = "rsgispy.elevation.hydrology"
HYDROLOGY_SINGLE_INPUT_FUNCS = [
    f"{_HYD}:{name}"
    for name in (
        "identify_sinks",
        "fill_depressions",
        "breach_depressions",
        "resolve_flats",
        "condition_dem",
        "flow_direction",
        "flow_accumulation",
        "flow_length",
        "extract_streams",
        "delineate_basins",
        "flood_inundation_from_hand",
    )
]
HYDROLOGY_MULTI_INPUT_FUNCS = [
    f"{_HYD}:{name}"
    for name in (
        "burn_streams",
        "flow_distance",
        "stream_link",
        "stream_order",
        "delineate_watersheds",
        "delineate_subcatchments",
        "calc_hand",
        "calc_twi",
        "calc_spi",
        "calc_ls_factor",
    )
]
SEGMENTATION_FUNCS = [
    f"rsgispy.segmentation:{name}"
    for name in (
        "shepherd_segmentation",
        "region_merge_segmentation",
        "slic_segmentation",
        "graph_segmentation",
        "watershed_segmentation",
        "eliminate_small_segments",
        "relabel_segments",
    )
]
IMAGEFILTER_APPLY_FUNCS = [
    func
    for sub in ("stats", "smooth", "edge", "sar", "texture", "chained")
    for func in _module_funcs(
        f"rsgispy.imagefilter.{sub}", lambda name: name.startswith("apply_")
    )
]
IMAGEFILTER_BORDER_FUNCS = [
    func
    for sub in ("stats", "smooth", "edge", "sar", "texture")
    for func in _module_funcs(
        f"rsgispy.imagefilter.{sub}",
        lambda name: name.startswith("apply_") or name.endswith("_node"),
    )
]
MORPHOLOGY_RASTER_FUNCS = _module_funcs(
    "rsgispy.imagemorphology",
    lambda name: name != "create_circular_op" and not name.endswith("_node"),
)
MORPHOLOGY_BORDER_FUNCS = _module_funcs(
    "rsgispy.imagemorphology",
    lambda name: name != "create_circular_op" and "local_minima" not in name,
)

KEYWORD_ONLY_CASES = (
    [
        (func, kw, default)
        for func in (
            HYDROLOGY_SINGLE_INPUT_FUNCS
            + HYDROLOGY_MULTI_INPUT_FUNCS
            + IMAGEFILTER_APPLY_FUNCS
            + MORPHOLOGY_RASTER_FUNCS
            + SEGMENTATION_FUNCS
        )
        for kw, default in KW_TAIL.items()
    ]
    + [
        (func, kw, default)
        for func in IMAGEFILTER_BORDER_FUNCS + MORPHOLOGY_BORDER_FUNCS
        for kw, default in BORDER_KWARGS.items()
    ]
)


@pytest.mark.parametrize("func, kw_name, kw_default", KEYWORD_ONLY_CASES)
def test_keyword_only_default(func, kw_name, kw_default):
    """The argument exists, is keyword-only and has the convention default."""
    sig = inspect.signature(_obj(func))
    assert kw_name in sig.parameters, f"{func} missing {kw_name}"
    par = sig.parameters[kw_name]
    assert par.kind == inspect.Parameter.KEYWORD_ONLY, f"{func}.{kw_name} must be keyword-only"
    assert par.default == kw_default


@pytest.mark.parametrize(
    "funcs, expected",
    [
        # 17 stats + 4 smooth + 9 edge + 7 sar + 1 texture + 1 chained
        pytest.param(IMAGEFILTER_APPLY_FUNCS, 39, id="imagefilter_apply"),
        pytest.param(IMAGEFILTER_BORDER_FUNCS, 76, id="imagefilter_border_mode"),
        pytest.param(MORPHOLOGY_RASTER_FUNCS, 12, id="imagemorphology_raster"),
        pytest.param(MORPHOLOGY_BORDER_FUNCS, 17, id="imagemorphology_border_mode"),
    ],
)
def test_signature_checked_function_count(funcs, expected):
    """The signature checks cover the expected number of functions."""
    assert len(funcs) == expected


def _required_positional(func):
    return [
        p.name
        for p in inspect.signature(_obj(func)).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]


@pytest.mark.parametrize("func", IMAGEFILTER_APPLY_FUNCS + MORPHOLOGY_RASTER_FUNCS)
def test_first_params_are_input_and_output_img(func):
    """The first two parameters are input_img and output_img."""
    params = list(inspect.signature(_obj(func)).parameters.values())
    assert params[0].name == "input_img"
    assert params[1].name == "output_img"


@pytest.mark.parametrize("func", HYDROLOGY_SINGLE_INPUT_FUNCS)
def test_required_positional_start_with_input_and_output_img(func):
    """The required positional arguments start with input_img, output_img."""
    assert _required_positional(func)[:2] == ["input_img", "output_img"], func


@pytest.mark.parametrize("func", SEGMENTATION_FUNCS)
def test_required_positional_are_input_and_output_img(func):
    """The only required positional arguments are input_img, output_img."""
    assert _required_positional(func) == ["input_img", "output_img"], func


@pytest.mark.parametrize(
    "func, param",
    [
        ("rsgispy.segmentation.clump:clump", "add_to_rat"),
        ("rsgispy.segmentation.clump:union_of_clumps", "add_to_rat"),
    ],
)
def test_param_removed(func, param):
    """The argument is no longer in the signature."""
    assert param not in inspect.signature(_obj(func)).parameters


def test_segmentation_is_numba_available_returns_bool():
    """is_numba_available() reports a bool."""
    import rsgispy.segmentation as rseg  # pylint: disable=import-outside-toplevel

    assert isinstance(rseg.is_numba_available(), bool)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
CLUSTERSKLEARN_FUNCS = ["img_pixel_cluster", "img_pixel_sample_cluster", "apply_img_pixel_clusterer"]
CLASSFEATURES_FUNCS = [
    "comparison_box_plots",
    "correlation_matrix",
    "calc_vif_multicollinearity",
    "select_features_cqv",
    "select_features_corr",
    "select_features_vif",
    "select_features_univariate",
    "select_features_rfe",
    "select_features_from_model",
    "select_features_boruta_shap",
    "calc_feature_importance",
    "calc_jackknife_feature_importance",
    "calc_feature_response_curves",
    "calc_shap_feature_contributions",
    "calc_salib_feature_sensitivity",
]
SDM_PUBLIC_HELPERS = [
    "get_positive_class_probs",
    "calc_single_class_predictions",
    "calc_binary_roc_auc",
    "calc_binary_threshold",
    "plot_roc_curve",
    "predict_raster_from_env_vars",
]
_SDM = "rsgispy.modelling.species_distribution"
CLI_MODULES = [
    "batchconvert2cog",
    "buildimglut",
    "calcimgstats",
    "chkgdalfile",
    "copybandnames",
    "filehash",
    "imginfo",
    "proj",
    "userpassfile",
    "vectools",
    "wgs84imgres",
]
ML_SUBMODULES = ["stack", "scaling", "predict", "cv", "tune", "vector", "segments"]
IMAGEFILTER_SUBMODULES = ["stats", "smooth", "edge", "sar", "texture", "chained"]

IN_ALL_CASES = (
    [("rsgispy.classification.clustersklearn", name) for name in CLUSTERSKLEARN_FUNCS]
    + [
        (func.split(":")[0], func.split(":")[1])
        for func in IMAGESPECTRAL_GTIFF_FUNCS
    ]
    + [(_SDM, name) for name in SDM_PUBLIC_HELPERS]
    + [
        (f"rsgispy.cli.{mod}", name)
        for mod in CLI_MODULES
        for name in (f"build_{mod}_parser", "main", "cli_main")
    ]
)


@pytest.mark.parametrize("module, name", IN_ALL_CASES)
def test_name_in_module_all(module, name):
    """The module has __all__ and exports the name."""
    mod = _obj(module)
    assert hasattr(mod, "__all__")
    assert name in mod.__all__


@pytest.mark.parametrize(
    "module, expected, ordered",
    [
        ("rsgispy.classification.classfeatures", CLASSFEATURES_FUNCS, False),
        ("rsgispy.ml", ML_SUBMODULES, True),
        ("rsgispy.mapping", ["layout", "elements", "raster", "vector", "basemaps", "legend"], False),
        # The package exports only its submodules, not individual filter functions.
        ("rsgispy.imagefilter", IMAGEFILTER_SUBMODULES, True),
    ],
)
def test_module_all_is_exactly(module, expected, ordered):
    """The module exports exactly these names (in order where it matters)."""
    exported = _obj(module).__all__
    if ordered:
        assert exported == expected
    else:
        assert set(exported) == set(expected)


_IMREG = "rsgispy.imageregistration"
HAS_ATTR_CASES = (
    [("rsgispy.elevation", "hydrology")]
    + [("rsgispy.classification.clustersklearn", name) for name in CLUSTERSKLEARN_FUNCS]
    + [
        ("rsgispy.imagespectral.endmembers", name)
        for name in ("read_endmembers_df", "write_endmembers_df", "convert_mtxt_to_df")
    ]
    + [
        ("rsgispy", name)
        for name in (
            "METRIC_EUCLIDEAN",
            "METRIC_SQDIFF",
            "METRIC_MANHATTAN",
            "METRIC_CORRELATION",
            "METRIC_MUTUAL_INFO",
            "METRIC_NORM_MUTUAL_INFO",
            "METRIC_PHASE_CORR",
        )
    ]
    + [
        (_IMREG, name)
        for name in (
            "TYPE_ENVI_IMG2IMG",
            "TYPE_ENVI_IMG2MAP",
            "TYPE_RSGIS_IMG2MAP",
            "TYPE_RSGIS_MAPOFFS",
            "TIE_METHOD_PHASE_CORR",
            "TIE_METHOD_TEMPLATE_MATCH",
            "OUTLIER_METHOD_RANSAC",
            "OUTLIER_METHOD_RESIDUAL",
            "OUTLIER_METHOD_MAD",
            "TRANSFORM_SHIFT",
            "TRANSFORM_SIMILARITY",
            "TRANSFORM_AFFINE",
            "TRANSFORM_POLY1",
            "TRANSFORM_POLY2",
            "TRANSFORM_POLY3",
            "TRANSFORM_TPS",
        )
    ]
    + [("rsgispy.ml", name) for name in ML_SUBMODULES]
)


@pytest.mark.parametrize("obj, name", HAS_ATTR_CASES)
def test_has_attribute(obj, name):
    """The package or module provides the name."""
    assert hasattr(_obj(obj), name)


NOT_HAS_ATTR_CASES = (
    # Submodule functions are not re-exported from rsgispy.classification.
    [("rsgispy.classification", name) for name in CLUSTERSKLEARN_FUNCS + CLASSFEATURES_FUNCS]
    + [
        ("rsgispy.imagespectral.endmembers", name)
        for name in ("read_endmembers_mtxt", "write_endmembers_mtxt")
    ]
    # Misspelt metric names were removed.
    + [
        (obj, name)
        for name in ("METRIC_CORELATION", "METRIC_MANHATTEN")
        for obj in ("rsgispy", _IMREG)
    ]
    + [
        ("rsgispy.segmentation", "union_of_clumps"),
        ("rsgispy.mapping", "create_raster_img_map"),
        ("rsgispy.imagefilter", "apply_mean_filter"),
    ]
)


@pytest.mark.parametrize("obj, name", NOT_HAS_ATTR_CASES)
def test_does_not_have_attribute(obj, name):
    """The name is not provided there (it lives in a submodule or was removed)."""
    assert not hasattr(_obj(obj), name)


@pytest.mark.parametrize(
    "name",
    [
        "__all__",
        "estimate_image_shift",
        "warp_with_gcps_with_gdal",
        "find_image_offset",
        "apply_offset_to_image",
        "generate_tie_points",
        "basic_registration",
        "filter_tie_points_outliers",
        "fit_geometric_model",
        "evaluate_geometric_model",
        "gcp_to_gdal",
        "add_vec_pts_as_gcps_to_img",
        "warp_with_gcps",
        "warp_with_tie_points",
        "coregister_images",
    ],
)
def test_imageregistration_package_defines_no_functions(name):
    """rsgispy.imageregistration defines no __all__ and no functions (they are
    in its submodules)."""
    assert name not in vars(_obj(_IMREG))


@pytest.mark.parametrize(
    "path", [f"rsgispy.classification.clustersklearn:{name}" for name in CLUSTERSKLEARN_FUNCS]
)
def test_is_function(path):
    """The export is a plain function."""
    assert inspect.isfunction(_obj(path))


@pytest.mark.parametrize(
    "path",
    [f"rsgispy.classification.classfeatures:{name}" for name in CLASSFEATURES_FUNCS]
    + [f"{_SDM}:{name}" for name in SDM_PUBLIC_HELPERS]
    + ["rsgispy.segmentation.clump:clump", "rsgispy.segmentation.clump:union_of_clumps"],
)
def test_is_callable(path):
    """The export is callable."""
    assert callable(_obj(path))


@pytest.mark.parametrize(
    "path", ["rsgispy.classification.classfeatures", "rsgispy.segmentation:clump"]
)
def test_is_module(path):
    """The attribute is a submodule."""
    assert inspect.ismodule(_obj(path))


@pytest.mark.parametrize(
    "path_a, path_b",
    [(f"rsgispy.imagefilter:{sub}", f"rsgispy.imagefilter.{sub}") for sub in IMAGEFILTER_SUBMODULES[:5]]
    # Internal callers import the SDM helpers from rsgispy.modelling._helpers.
    + [(f"rsgispy.modelling._helpers:{name}", f"{_SDM}:{name}") for name in SDM_PUBLIC_HELPERS],
)
def test_same_object(path_a, path_b):
    """Both paths give the same object."""
    assert _obj(path_a) is _obj(path_b)


@pytest.mark.parametrize("name", SDM_PUBLIC_HELPERS)
def test_sdm_public_helper_docstring_has_example(name):
    """The public SDM helper documents a code example."""
    func = _obj(f"{_SDM}:{name}")
    assert func.__doc__ and ".. code:: python" in func.__doc__


def test_sdm_public_helpers_import_from_public_module():
    """The helpers import from rsgispy.modelling.species_distribution."""
    # pylint: disable=import-outside-toplevel,unused-import
    from rsgispy.modelling.species_distribution import (  # noqa: F401
        calc_binary_roc_auc,
        calc_binary_threshold,
        calc_single_class_predictions,
        get_positive_class_probs,
        plot_roc_curve,
        predict_raster_from_env_vars,
    )


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_parser_prog_starts_with_rsgis(module):
    """Each CLI tool's parser is named rsgis*."""
    parser = getattr(_obj(f"rsgispy.cli.{module}"), f"build_{module}_parser")()
    assert parser.prog.startswith("rsgis")
