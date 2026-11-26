"""
Grey-Level Co-occurrence Matrix (GLCM) spatial texture filters for raster images.
"""

from typing import List, Optional, Sequence, Tuple, Union

import rsgispy
from rsgispy._io import RasterSource, TileNode, resolve_input_nodata
from rsgispy.imagefilter._glcm_helpers import (
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
    GlcmWindowOp,
    calc_band_bounds,
    glcm_band_names,
    validate_glcm_params,
)
from rsgispy.imagefilter._helpers import run_filter_node
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageutils.metadata import get_band_names

__all__: List[str] = [
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
    "calc_glcm_band_bounds",
    "glcm_filter_node",
]


def apply_glcm_filter(
    input_img: str,
    output_img: str,
    properties: Union[str, Sequence[str]] = GLCM_ALL_PROPERTIES,
    filter_size: int = 5,
    distances: Union[int, Sequence[int]] = (1,),
    angles: Union[float, Sequence[float]] = (0.0, 45.0, 90.0, 135.0),
    n_levels: int = 32,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    symmetric: bool = True,
    average_angles: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute moving-window Grey-Level Co-occurrence Matrix (GLCM) texture
    features for each band of ``input_img``.

    Supported ``properties`` include ``'contrast'``, ``'dissimilarity'``,
    ``'homogeneity'``, ``'asm'``, ``'energy'``, ``'correlation'``,
    ``'entropy'``, ``'mean'``, ``'variance'``, and ``'max_prob'``.

    When ``average_angles=True`` (the default), each selected property is
    averaged over all ``distances`` and ``angles``, producing
    ``len(properties)`` output bands per input band (with band names
    ``<band>_glcm_<property>``). When ``average_angles=False``, each
    ``(property, distance, angle)`` combination is written as a separate band.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output texture stack image.
    :param properties: GLCM property name or sequence of property names
                       (Default = :data:`GLCM_ALL_PROPERTIES`).
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param distances: Pixel offset distance(s) ``>= 1`` and ``< filter_size``
                      (Default = ``(1,)``).
    :param angles: Co-occurrence orientation angle(s) in degrees
                   (Default = ``(0.0, 45.0, 90.0, 135.0)``).
    :param n_levels: Number of quantised grey levels ``>= 2`` (Default = ``32``).
    :param min_val: Optional explicit lower bound for grey-level quantisation
                    (both ``min_val`` and ``max_val`` or neither must be set).
    :param max_val: Optional explicit upper bound for grey-level quantisation.
    :param symmetric: If ``True``, construct symmetric GLCMs by counting both
                      ``(i, j)`` and ``(j, i)`` pairs (Default = ``True``).
    :param average_angles: If ``True``, average each property over all
                           ``(distance, angle)`` offsets (Default = ``True``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Output pixel data type constant (Default = ``None``,
                     resolves to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional explicit input nodata value (Default = ``None``).
    :param calc_stats: If ``True``, compute output stats and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).

    .. code:: python

        import rsgispy.imagefilter.texture as tex

        tex.apply_glcm_filter(
            input_img="s2.tif",
            output_img="s2_glcm.tif",
            properties=[tex.GLCM_CONTRAST, tex.GLCM_HOMOGENEITY, tex.GLCM_ENTROPY],
            filter_size=5,
            n_levels=32,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img, "Input image")
    _, prop_list, offsets, _ = validate_glcm_params(
        properties, filter_size, distances, angles, n_levels, min_val, max_val
    )
    band_bounds = None
    if min_val is None or max_val is None:
        band_bounds = calc_band_bounds(
            input_img, no_data_val=no_data_val, use_pbar=use_pbar
        )
    src = RasterSource(input_img)
    node = glcm_filter_node(
        src,
        properties=properties,
        filter_size=filter_size,
        distances=distances,
        angles=angles,
        n_levels=n_levels,
        min_val=min_val,
        max_val=max_val,
        symmetric=symmetric,
        average_angles=average_angles,
        band_bounds=band_bounds,
    )
    run_filter_node(
        src,
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        band_names=glcm_band_names(
            get_band_names(input_img), prop_list, offsets, bool(average_angles)
        ),
    )


def calc_glcm_band_bounds(
    input_img: str,
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> List[Tuple[float, float]]:
    """
    Calculate the per-band ``(min, max)`` of the valid pixels of
    ``input_img``: the grey-level range :func:`apply_glcm_filter` uses when
    ``min_val`` and ``max_val`` are not given. Pass it as ``band_bounds`` to
    :func:`glcm_filter_node` to reproduce :func:`apply_glcm_filter` in a tile
    graph (a whole-image parameter, so it is calculated in a first pass).

    :param input_img: Path to the input raster image.
    :param no_data_val: Optional explicit input nodata value (Default =
                        ``None``, read from the image header).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: List of ``(min, max)`` tuples, one per band (``(0.0, 1.0)`` for a
             band without valid pixels).

    .. code:: python

        import rsgispy.imagefilter.texture as tex

        bounds = tex.calc_glcm_band_bounds("s2.tif")
    """
    return calc_band_bounds(input_img, no_data_val=no_data_val, use_pbar=use_pbar)


def glcm_filter_node(
    source: TileNode,
    properties: Union[str, Sequence[str]] = GLCM_ALL_PROPERTIES,
    filter_size: int = 5,
    distances: Union[int, Sequence[int]] = (1,),
    angles: Union[float, Sequence[float]] = (0.0, 45.0, 90.0, 135.0),
    n_levels: int = 32,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    symmetric: bool = True,
    average_angles: bool = True,
    *,
    band_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_glcm_filter`: moving-window Grey-Level
    Co-occurrence Matrix (GLCM) texture measures of each band of ``source``,
    for chaining in a :class:`~rsgispy.imageprocess.TileGraph` without
    intermediate files.

    The grey levels are quantised over a fixed range, which must be known
    before the graph runs: give ``min_val`` and ``max_val`` (one range for all
    bands), or ``band_bounds`` (one ``(min, max)`` per band, e.g. from
    :func:`calc_glcm_band_bounds`, which is what :func:`apply_glcm_filter`
    uses when ``min_val`` / ``max_val`` are not given).

    The node outputs ``float64`` blocks with ``len(properties)`` bands per
    input band (times the number of offsets when ``average_angles`` is
    ``False``), named as by :func:`apply_glcm_filter`
    (``<band>_glcm_<property>``); its halo radius is ``filter_size // 2``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param properties: GLCM property name or sequence of property names
                       (Default = :data:`GLCM_ALL_PROPERTIES`).
    :param filter_size: Odd filter window width/height >= 3 (Default = ``5``).
    :param distances: Pixel offset distance(s) ``>= 1`` and ``< filter_size``
                      (Default = ``(1,)``).
    :param angles: Co-occurrence orientation angle(s) in degrees
                   (Default = ``(0.0, 45.0, 90.0, 135.0)``).
    :param n_levels: Number of quantised grey levels ``>= 2`` (Default = ``32``).
    :param min_val: Lower bound of the grey-level range (with ``max_val``).
    :param max_val: Upper bound of the grey-level range (with ``min_val``).
    :param symmetric: If ``True``, construct symmetric GLCMs by counting both
                      ``(i, j)`` and ``(j, i)`` pairs (Default = ``True``).
    :param average_angles: If ``True``, average each property over all
                           ``(distance, angle)`` offsets (Default = ``True``).
    :param band_bounds: Per-band ``(min, max)`` grey-level ranges, used when
                        ``min_val`` and ``max_val`` are ``None``.
    :param no_data_val: Optional input no-data value: these pixels are
                        ignored, and output pixels without a value are set to
                        it (Default = ``None``: the no-data value of
                        ``source``'s metadata; see :ref:`tilegraph-nodata`).
    :param name: Optional node name (Default = ``'glcm_filter'``).
    :return: :class:`~rsgispy.imageprocess.WindowOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        import rsgispy.imagefilter.texture as tex
        from rsgispy.imagefilter.stats import median_filter_node

        src = RasterSource("s2_nir.tif")
        smooth = median_filter_node(src, 3)
        glcm = tex.glcm_filter_node(
            smooth,
            properties=[tex.GLCM_CONTRAST, tex.GLCM_ENTROPY],
            filter_size=7,
            min_val=0.0,
            max_val=0.6,
        )
        TileGraph(DiskSink(glcm, "nir_median_glcm.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    size, prop_list, offsets, levels = validate_glcm_params(
        properties, filter_size, distances, angles, n_levels, min_val, max_val
    )
    if min_val is not None and max_val is not None:
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (
            float(min_val),
            float(max_val),
        )
    elif band_bounds is not None:
        bounds = [(float(lo), float(hi)) for lo, hi in band_bounds]
        if not bounds:
            raise rsgispy.RSGISPyException("band_bounds must not be empty.")
        for lo, hi in bounds:
            if not hi >= lo:
                raise rsgispy.RSGISPyException(
                    f"band_bounds entries must have max >= min, got {(lo, hi)!r}."
                )
    else:
        raise rsgispy.RSGISPyException(
            "glcm_filter_node needs the grey-level range before the graph "
            "runs: give min_val and max_val, or band_bounds (e.g. from "
            "calc_glcm_band_bounds())."
        )
    return GlcmWindowOp(
        source,
        size,
        prop_list,
        offsets,
        levels,
        bool(symmetric),
        bool(average_angles),
        bounds,
        no_data_val=resolve_input_nodata(source, no_data_val),
        name=name or "glcm_filter",
    )
