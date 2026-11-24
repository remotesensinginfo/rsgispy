"""
The imagemorphology module contains functions for performing morphological
operations on raster images using ``rasterio`` and ``numpy``.
"""

from rsgispy.imagemorphology._ops import (
    create_circular_op,
    image_black_top_hat,
    image_closing,
    image_dilate,
    image_dilate_combined_out,
    image_erode,
    image_erode_combined_out,
    image_gradient,
    image_gradient_combined_out,
    image_local_minima,
    image_local_minima_combined_out,
    image_opening,
    image_white_top_hat,
)

__all__ = [
    "create_circular_op",
    "image_dilate",
    "image_erode",
    "image_gradient",
    "image_dilate_combined_out",
    "image_erode_combined_out",
    "image_gradient_combined_out",
    "image_local_minima",
    "image_local_minima_combined_out",
    "image_opening",
    "image_closing",
    "image_black_top_hat",
    "image_white_top_hat",
]
