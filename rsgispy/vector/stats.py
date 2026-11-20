"""
Spatial statistics, geometric measurements, feature counts, nearest-neighbour
metrics, and empirical semi-variogram calculation for
:mod:`rsgispy.vector.stats`.
"""

from rsgispy.vector._stats_metrics import (
    HAVE_SCIPY,
    calc_near_dist_to_feats,
    calc_npts_in_radius,
    calc_vec_area,
    calc_vec_length,
    calc_vec_pt_dist_angle,
    count_feats_per_att_val,
    count_pt_intersects,
    get_poly_hole_area,
)
from rsgispy.vector._stats_variogram import calc_empirical_semivariogram

__all__ = [
    "HAVE_SCIPY",
    "calc_vec_area",
    "calc_vec_length",
    "get_poly_hole_area",
    "calc_vec_pt_dist_angle",
    "calc_near_dist_to_feats",
    "count_pt_intersects",
    "calc_npts_in_radius",
    "count_feats_per_att_val",
    "calc_empirical_semivariogram",
]
