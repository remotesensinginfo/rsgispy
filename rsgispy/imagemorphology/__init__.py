"""
The imagemorphology module contains functions for performing morphological
operations on raster images using ``rasterio`` and ``numpy``.

Each operation also has a tile-graph node builder (``image_dilate_node``,
``image_opening_node``, ...) for chaining it with other operations in a
:class:`~rsgispy.imageprocess.TileGraph` without intermediate files.
"""

from rsgispy.imagemorphology._ops import (
    create_circular_op,
    image_black_top_hat,
    image_black_top_hat_node,
    image_closing,
    image_closing_node,
    image_dilate,
    image_dilate_combined_out,
    image_dilate_node,
    image_erode,
    image_erode_combined_out,
    image_erode_node,
    image_gradient,
    image_gradient_combined_out,
    image_gradient_node,
    image_local_minima,
    image_local_minima_combined_out,
    image_local_minima_node,
    image_opening,
    image_opening_node,
    image_white_top_hat,
    image_white_top_hat_node,
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
    "image_dilate_node",
    "image_erode_node",
    "image_gradient_node",
    "image_local_minima_node",
    "image_opening_node",
    "image_closing_node",
    "image_white_top_hat_node",
    "image_black_top_hat_node",
]
