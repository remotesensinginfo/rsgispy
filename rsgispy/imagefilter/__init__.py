"""
The imagefilter package provides spatial moving-window and convolution filters
for raster images using ``rasterio`` and ``numpy``, organised into submodules by
filter type:

* :mod:`rsgispy.imagefilter.stats` — statistical, dispersion, and local-difference filters.
* :mod:`rsgispy.imagefilter.smooth` — Kuwahara, Gaussian smoothing, and custom kernels.
* :mod:`rsgispy.imagefilter.edge` — Sobel, Prewitt, Gaussian derivative, and Laplacian edge filters.
* :mod:`rsgispy.imagefilter.sar` — Lee, Enhanced Lee, Frost, Enhanced Frost, Kuan, Gamma MAP,
  and Refined Lee SAR speckle filters.
* :mod:`rsgispy.imagefilter.texture` — Gray-Level Co-occurrence Matrix (GLCM) texture filters.

For backwards compatibility, all filter functions are also re-exported directly
under :mod:`rsgispy.imagefilter`.
"""

from typing import List

from rsgispy.imagefilter import edge, sar, smooth, stats, texture
from rsgispy.imagefilter.edge import (
    apply_gaussian_1st_deriv_filter,
    apply_gaussian_2nd_deriv_filter,
    apply_laplacian_filter,
    apply_prewitt_filter,
    apply_prewitt_x_filter,
    apply_prewitt_y_filter,
    apply_sobel_filter,
    apply_sobel_x_filter,
    apply_sobel_y_filter,
)
from rsgispy.imagefilter.sar import (
    apply_enhanced_frost_filter,
    apply_enhanced_lee_filter,
    apply_frost_filter,
    apply_gamma_map_filter,
    apply_kuan_filter,
    apply_lee_filter,
    apply_refined_lee_filter,
)
from rsgispy.imagefilter.smooth import (
    apply_gaussian_smooth_filter,
    apply_kernel_filter,
    apply_kuwahara_filter,
)
from rsgispy.imagefilter.stats import (
    apply_coeff_of_var_filter,
    apply_max_filter,
    apply_mean_diff_abs_filter,
    apply_mean_diff_filter,
    apply_mean_filter,
    apply_median_filter,
    apply_min_filter,
    apply_mode_filter,
    apply_norm_var_filter,
    apply_norm_var_ln_filter,
    apply_norm_var_sqrt_filter,
    apply_range_filter,
    apply_stddev_filter,
    apply_texture_var_filter,
    apply_total_diff_abs_filter,
    apply_total_diff_filter,
    apply_total_filter,
)
from rsgispy.imagefilter.texture import (
    GLCM_ALL_PROPERTIES,
    GLCM_ASM,
    GLCM_CONTRAST,
    GLCM_CORRELATION,
    GLCM_DISSIMILARITY,
    GLCM_ENERGY,
    GLCM_ENTROPY,
    GLCM_HOMOGENEITY,
    GLCM_MAX_PROB,
    GLCM_MEAN,
    GLCM_VARIANCE,
    apply_glcm_filter,
)

__all__: List[str] = [
    "stats",
    "smooth",
    "edge",
    "sar",
    "texture",
    "apply_median_filter",
    "apply_mean_filter",
    "apply_min_filter",
    "apply_max_filter",
    "apply_mode_filter",
    "apply_stddev_filter",
    "apply_range_filter",
    "apply_mean_diff_filter",
    "apply_mean_diff_abs_filter",
    "apply_total_diff_filter",
    "apply_total_diff_abs_filter",
    "apply_coeff_of_var_filter",
    "apply_total_filter",
    "apply_norm_var_filter",
    "apply_norm_var_sqrt_filter",
    "apply_norm_var_ln_filter",
    "apply_texture_var_filter",
    "apply_kuwahara_filter",
    "apply_gaussian_smooth_filter",
    "apply_kernel_filter",
    "apply_sobel_filter",
    "apply_sobel_x_filter",
    "apply_sobel_y_filter",
    "apply_prewitt_filter",
    "apply_prewitt_x_filter",
    "apply_prewitt_y_filter",
    "apply_gaussian_1st_deriv_filter",
    "apply_gaussian_2nd_deriv_filter",
    "apply_laplacian_filter",
    "apply_lee_filter",
    "apply_enhanced_lee_filter",
    "apply_frost_filter",
    "apply_enhanced_frost_filter",
    "apply_kuan_filter",
    "apply_gamma_map_filter",
    "apply_refined_lee_filter",
    "GLCM_CONTRAST",
    "GLCM_DISSIMILARITY",
    "GLCM_HOMOGENEITY",
    "GLCM_ASM",
    "GLCM_ENERGY",
    "GLCM_CORRELATION",
    "GLCM_ENTROPY",
    "GLCM_MEAN",
    "GLCM_VARIANCE",
    "GLCM_MAX_PROB",
    "GLCM_ALL_PROPERTIES",
    "apply_glcm_filter",
]
