"""
Gray-Level Co-occurrence Matrix (GLCM) spatial texture filters for raster images.
"""

from typing import List, Optional, Sequence, Union

import rsgispy
from rsgispy.imagefilter._glcm_helper import (
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
    run_glcm_filter,
)

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
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute moving-window Gray-Level Co-occurrence Matrix (GLCM) texture
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
    :param datatype: Output pixel data type constant
                     (Default = :data:`rsgispy.TYPE_32FLOAT`).
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
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_glcm_filter(
        input_img=input_img,
        output_img=output_img,
        properties=properties,
        filter_size=filter_size,
        distances=distances,
        angles=angles,
        n_levels=n_levels,
        min_val=min_val,
        max_val=max_val,
        symmetric=symmetric,
        average_angles=average_angles,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
