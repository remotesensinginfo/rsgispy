"""
Geostatistical variogram utilities for :mod:`rsgispy.vectorstats` using
``geopandas``, ``numpy``, and ``skgstat``.
"""

from typing import Optional, Union

import numpy

import rsgispy
import rsgispy.tools.stats
from rsgispy.vectorprocess import (
    get_gdf_geom_type,
    read_vec_gdf,
    validate_columns_exist,
)

__all__ = [
    "calc_empirical_semivariogram",
]


def calc_empirical_semivariogram(
    vec_file: str,
    vec_lyr: Optional[str],
    vals_col: str,
    pts_sel_col: Optional[str] = None,
    pts_sel_val: Optional[Union[int, float, str]] = None,
    out_data_file: Optional[str] = None,
    out_plot_file: Optional[str] = None,
    max_lag: Union[float, str] = "median",
    n_lags: int = 25,
    normalise_vals: bool = False,
):
    """
    Calculate empirical semi-variance for a point vector layer using
    ``skgstat`` via :func:`rsgispy.tools.stats.calc_semivariogram`.

    The vector layer must use point geometries in a projected coordinate system
    with valid X and Y coordinates.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None`` for single-layer files).
    :param vals_col: Column name containing the attribute values.
    :param pts_sel_col: Optional column name used to filter points prior to
                        semi-variance calculation.
    :param pts_sel_val: Optional value in ``pts_sel_col`` selecting the subset
                        of points to analyse.
    :param out_data_file: Optional CSV file path for distance, semi-variance,
                          and pair count outputs.
    :param out_plot_file: Optional image file path for a variogram plot.
    :param max_lag: Maximum lag distance, or ``'median'`` / ``'mean'``
                    (Default = ``'median'``).
    :param n_lags: Number of lag classes (Default = ``25``).
    :param normalise_vals: If ``True``, normalise distance and semi-variance
                           values to ``[0, 1]`` (Default = ``False``).
    :return: ``pandas.DataFrame`` containing ``distance``, ``semivariance``,
             and ``count`` columns.

    .. code:: python

        import rsgispy.vectorstats.variogram

        vario_df = rsgispy.vectorstats.variogram.calc_empirical_semivariogram(
            vec_file="samples.gpkg",
            vec_lyr="samples",
            vals_col="height",
            n_lags=20,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    if data_gdf.empty:
        raise rsgispy.RSGISPyException("Input vector layer contains no features.")

    if get_gdf_geom_type(data_gdf) != rsgispy.GEOM_PT:
        raise rsgispy.RSGISPyException("Vector geometry type must be points.")

    validate_columns_exist(data_gdf, [vals_col])

    geom_pts = data_gdf.geometry.get_coordinates()
    if not {"x", "y"}.issubset(geom_pts.columns):
        raise rsgispy.RSGISPyException(
            "The geometry is not as expected - need x and y fields."
        )

    x_coords = data_gdf.geometry.x.to_numpy()
    y_coords = data_gdf.geometry.y.to_numpy()
    vals = data_gdf[vals_col].to_numpy()

    if (pts_sel_col is not None) and (pts_sel_val is not None):
        validate_columns_exist(data_gdf, [pts_sel_col])
        pt_sets = data_gdf[pts_sel_col].to_numpy()
        mask = pt_sets == pts_sel_val
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        vals = vals[mask]

    pts_coords = numpy.stack([x_coords, y_coords]).T

    return rsgispy.tools.stats.calc_semivariogram(
        pts_coords=pts_coords,
        data_vals=vals,
        out_data_file=out_data_file,
        out_plot_file=out_plot_file,
        max_lag=max_lag,
        n_lags=n_lags,
        normalise_vals=normalise_vals,
    )
