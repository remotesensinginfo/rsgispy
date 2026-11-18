"""
The imageutils subpackage provides modules for raster metadata, creation
options, blank/copy images, rasterisation, array/pixel I/O, band operations,
sampling, checks, spatial grids, warping, mosaicking, tiling, masking,
subsetting, VRT generation, colour tables, image statistics, and spatial LUTs.
"""

from rsgispy.imageutils.masking import create_valid_mask

__all__ = [
    "create_valid_mask",
]
