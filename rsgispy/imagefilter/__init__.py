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
* :mod:`rsgispy.imagefilter.chained` — In-memory halo-fused multi-step filter & band-math pipelines.
"""

from typing import List

from rsgispy.imagefilter import chained, edge, sar, smooth, stats, texture

__all__: List[str] = [
    "stats",
    "smooth",
    "edge",
    "sar",
    "texture",
    "chained",
]
