"""
The imagecalc.calcindices module provides unified spectral index stack
calculation using ``rasterio`` and ``numpy``.
"""

from rsgispy.imagecalc._indices_optical import calc_indices, calc_indices_node
from rsgispy.imagecalc._indices_registry import (
    get_available_indices,
    get_index_equation,
    get_index_req_bands,
)
from rsgispy.imagecalc._indices_sar import calc_sar_indices, calc_sar_indices_node
from rsgispy.imagecalc._sar_indices import (
    get_available_sar_indices,
    get_sar_index_equation,
    get_sar_index_req_bands,
    get_sar_index_units,
)

__all__ = [
    "calc_indices",
    "calc_indices_node",
    "get_available_indices",
    "get_index_req_bands",
    "get_index_equation",
    "calc_sar_indices",
    "calc_sar_indices_node",
    "get_available_sar_indices",
    "get_sar_index_req_bands",
    "get_sar_index_equation",
    "get_sar_index_units",
]
