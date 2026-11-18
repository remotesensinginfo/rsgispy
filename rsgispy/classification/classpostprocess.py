"""
Post-processing filters for thematic classification rasters.

Functions:

* :func:`apply_majority_filter` — Smooth classification boundaries across the
  entire scene by replacing each centre pixel with the modal (most frequent)
  class inside a moving window.
* :func:`apply_sieve_filter` — Connected-component clump elimination (MMU
  filtering) that identifies 4- or 8-connected regions smaller than a specified
  pixel count threshold and merges them into their surrounding majority
  neighbour.
"""

import os
from typing import Any, Mapping, Optional, Union

from rsgispy.classification._postprocess_helpers import (
    run_majority_filter,
    run_sieve_filter,
)

__all__ = [
    "apply_majority_filter",
    "apply_sieve_filter",
]


def apply_majority_filter(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    filter_size: int = 3,
    img_band: int = 1,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    keep_centre_on_tie: bool = True,
    calc_stats: bool = True,
    copy_colour_table: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Apply a moving-window majority (modal) filter to a thematic classification
    raster.

    For every valid pixel in ``img_band``, replaces the centre pixel with the
    most frequently occurring valid class within the ``filter_size x
    filter_size`` moving window. Window edges and no-data pixels are excluded
    from frequency counts so out-of-bounds areas do not bias boundary pixels.

    Reuses :mod:`rsgispy.imagefilter` modal voting and
    :mod:`rsgispy.imageprocess` windowed halo block execution.

    :param input_img: Path to the input classification raster.
    :param output_img: Path to the filtered output classification raster.
    :param filter_size: Odd square window dimension ``>= 3`` (e.g., ``3`` or
                        ``5``; Default = ``3``).
    :param img_band: 1-based band index within ``input_img`` to filter
                     (Default = ``1``).
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer data type constant
                     for ``output_img``. When ``None`` (Default), preserves the
                     input band's data type.
    :param no_data_val: Optional no-data pixel value for ``input_img``. When
                        ``None`` (Default), uses the no-data value defined in
                        the input raster metadata (or treats all finite pixels
                        as valid if unset, allowing class ``0``).
    :param keep_centre_on_tie: If ``True`` (Default), retains the existing
                               centre pixel class whenever it is tied with the
                               window maximum count. If ``False``, breaks ties
                               deterministically toward the smallest class ID.
    :param calc_stats: If ``True`` (Default), computes thematic statistics and
                       nearest-neighbour pyramids for integer outputs (or
                       pyramids for floating-point outputs).
    :param copy_colour_table: If ``True`` (Default), copies the input band's
                              colour table to ``output_img`` when present and
                              supported by the output data type (``uint8`` or
                              ``uint16``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping for
                          ``output_img``.
    :param use_pbar: If ``True`` (Default), displays a ``tqdm`` progress bar
                     during processing.

    .. code:: python

        import rsgispy.classification.classpostprocess as classpost

        classpost.apply_majority_filter(
            input_img="classified.tif",
            output_img="classified_maj3x3.tif",
            filter_size=3,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_majority_filter(
        in_path=os.fspath(input_img),
        out_path=os.fspath(output_img),
        filter_size=filter_size,
        img_band=img_band,
        datatype=datatype,
        no_data_val=no_data_val,
        keep_centre_on_tie=keep_centre_on_tie,
        gdalformat=gdalformat,
        calc_stats=calc_stats,
        copy_colour_table=copy_colour_table,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
    )


def apply_sieve_filter(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    size_threshold: int,
    connectivity: int = 4,
    img_band: int = 1,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    in_memory: Optional[bool] = None,
    block_size: Optional[int] = None,
    calc_stats: bool = True,
    copy_colour_table: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Apply connected-component sieve filtering (clump elimination) to remove
    isolated classification regions smaller than ``size_threshold`` pixels.

    Identifies 4- or 8-connected components (clumps) of identical valid class
    values and merges any clump strictly smaller than ``size_threshold`` (the
    Minimum Mapping Unit in pixels) into its surrounding majority neighbour
    class, ranked by total shared boundary contact count (breaking ties by
    largest neighbour clump size, then smallest class ID). Isolated valid
    clumps surrounded solely by no-data pixels are preserved rather than turned
    into no-data.

    Reuses :mod:`rsgispy.segmentation` RLE graph connected-component labelling
    and :mod:`rsgispy.imageprocess` windowed block streaming, supporting both
    single-pass in-memory execution and memory-bounded multi-pass tiled
    execution.

    :param input_img: Path to the input classification raster.
    :param output_img: Path to the sieved output classification raster.
    :param size_threshold: Minimum mapping unit in pixels (``>= 1``). Clumps
                           with pixel area ``< size_threshold`` are merged into
                           neighbouring clumps; ``1`` leaves valid pixels
                           unchanged.
    :param connectivity: Pixel connectivity rule, either ``4`` (orthogonal
                         neighbours only, Default) or ``8`` (orthogonal and
                         diagonal neighbours).
    :param img_band: 1-based band index within ``input_img`` to sieve
                     (Default = ``1``).
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer data type constant
                     for ``output_img``. When ``None`` (Default), preserves the
                     input band's data type.
    :param no_data_val: Optional no-data pixel value for ``input_img``. When
                        ``None`` (Default), uses the no-data value defined in
                        the input raster metadata (or treats all finite pixels
                        as valid if unset, allowing class ``0``).
    :param in_memory: Optional override for single-pass in-memory execution
                      (``True``) versus multi-pass tiled streaming (``False``).
                      When ``None`` (Default), selects in-memory for moderate
                      grids and tiled streaming automatically for large grids.
    :param block_size: Optional tile size in pixels for tiled mode. Defaults to
                       :func:`rsgispy.imageprocess.get_default_block_size`.
    :param calc_stats: If ``True`` (Default), computes thematic statistics and
                       nearest-neighbour pyramids for integer outputs (or
                       pyramids for floating-point outputs).
    :param copy_colour_table: If ``True`` (Default), copies the input band's
                              colour table to ``output_img`` when present and
                              supported by the output data type (``uint8`` or
                              ``uint16``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping for
                          ``output_img``.
    :param use_pbar: If ``True`` (Default), displays a ``tqdm`` progress bar
                     during processing.

    .. code:: python

        import rsgispy.classification.classpostprocess as classpost

        classpost.apply_sieve_filter(
            input_img="classified.tif",
            output_img="classified_sieve10.tif",
            size_threshold=10,
            connectivity=8,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_sieve_filter(
        in_path=os.fspath(input_img),
        out_path=os.fspath(output_img),
        size_threshold=size_threshold,
        connectivity=connectivity,
        img_band=img_band,
        datatype=datatype,
        no_data_val=no_data_val,
        gdalformat=gdalformat,
        in_memory=in_memory,
        block_size=block_size,
        calc_stats=calc_stats,
        copy_colour_table=copy_colour_table,
        creation_opts=creation_opts,
        use_pbar=use_pbar,
    )
