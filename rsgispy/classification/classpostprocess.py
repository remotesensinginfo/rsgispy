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
* :func:`apply_postprocess_pipeline` — Multi-stage classification
  post-processing chaining one or more majority filter passes (halo-fused in
  memory via :class:`~rsgispy._io.TileGraph`) with an optional in-memory sieve
  pass so no intermediate GeoTIFF files are written.
"""

import os
from typing import Any, Mapping, Optional, Sequence, Union

from rsgispy.classification._postprocess_helpers import (
    run_majority_filter,
    run_postprocess_pipeline,
    run_sieve_filter,
)

__all__ = [
    "apply_majority_filter",
    "apply_sieve_filter",
    "apply_postprocess_pipeline",
]


def apply_majority_filter(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    filter_size: int = 3,
    img_band: int = 1,
    keep_centre_on_tie: bool = True,
    copy_colour_table: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
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
    :param keep_centre_on_tie: If ``True`` (Default), retains the existing
                               centre pixel class whenever it is tied with the
                               window maximum count. If ``False``, breaks ties
                               deterministically toward the smallest class ID.
    :param copy_colour_table: If ``True`` (Default), copies the input band's
                              colour table to ``output_img`` when present and
                              supported by the output data type (``uint8`` or
                              ``uint16``).
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer data type constant
                     for ``output_img``. When ``None`` (Default), preserves the
                     input band's data type.
    :param no_data_val: Optional no-data pixel value for ``input_img``. When
                        ``None`` (Default), uses the no-data value defined in
                        the input raster metadata (or treats all finite pixels
                        as valid if unset, allowing class ``0``).
    :param calc_stats: If ``True`` (Default), computes thematic statistics and
                       nearest-neighbour pyramids for integer outputs (or
                       pyramids for floating-point outputs).
    :param use_pbar: If ``True`` (Default), displays a ``tqdm`` progress bar
                     during processing.
    :param creation_opts: Optional GDAL/rasterio creation options mapping for
                          ``output_img``.

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
    in_memory: Optional[bool] = None,
    block_size: Optional[int] = None,
    copy_colour_table: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
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
    :param in_memory: Optional override for single-pass in-memory execution
                      (``True``) versus multi-pass tiled streaming (``False``).
                      When ``None`` (Default), selects in-memory for moderate
                      grids and tiled streaming automatically for large grids.
    :param block_size: Optional tile size in pixels for tiled mode. Defaults to
                       :func:`rsgispy.imageprocess.get_default_block_size`.
    :param copy_colour_table: If ``True`` (Default), copies the input band's
                              colour table to ``output_img`` when present and
                              supported by the output data type (``uint8`` or
                              ``uint16``).
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer data type constant
                     for ``output_img``. When ``None`` (Default), preserves the
                     input band's data type.
    :param no_data_val: Optional no-data pixel value for ``input_img``. When
                        ``None`` (Default), uses the no-data value defined in
                        the input raster metadata (or treats all finite pixels
                        as valid if unset, allowing class ``0``).
    :param calc_stats: If ``True`` (Default), computes thematic statistics and
                       nearest-neighbour pyramids for integer outputs (or
                       pyramids for floating-point outputs).
    :param use_pbar: If ``True`` (Default), displays a ``tqdm`` progress bar
                     during processing.
    :param creation_opts: Optional GDAL/rasterio creation options mapping for
                          ``output_img``.

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


def apply_postprocess_pipeline(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    majority_sizes: Sequence[int] = (),
    sieve_size_threshold: Optional[int] = None,
    sieve_connectivity: int = 4,
    keep_centre_on_tie: bool = True,
    img_band: int = 1,
    in_memory: Optional[bool] = None,
    block_size: Optional[int] = None,
    copy_colour_table: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Execute a multi-stage classification post-processing pipeline in memory
    without writing intermediate GeoTIFF files.

    Chains zero or more moving-window majority (modal) filter passes specified
    by ``majority_sizes`` in a single halo-fused
    :class:`~rsgispy._io.TileGraph` pass, optionally followed by an in-memory
    connected-component sieve pass (when ``sieve_size_threshold`` is provided).

    :param input_img: Path to the input classification raster.
    :param output_img: Path to the final post-processed classification raster.
    :param majority_sizes: Sequence of odd window sizes ``>= 3`` applied in
                           order (e.g., ``(3, 5)`` or ``(3, 3)``; Default = ``()``).
    :param sieve_size_threshold: Optional minimum mapping unit (MMU) in pixels
                                 (``>= 1``) for connected-component sieving
                                 after any majority passes (Default = ``None``).
    :param sieve_connectivity: Pixel connectivity for sieving: ``4`` (Default)
                               or ``8``.
    :param keep_centre_on_tie: Retain centre pixel on modal count ties
                               (Default = ``True``).
    :param img_band: 1-based band index within ``input_img`` (Default = ``1``).
    :param in_memory: Optional override for in-memory (``True``) vs. tiled
                      (``False``) sieving. When ``None`` (Default), selects
                      in-memory automatically for moderate grids.
    :param block_size: Optional processing tile size in pixels.
    :param copy_colour_table: Copy input colour table to ``output_img`` when
                              present and supported (Default = ``True``).
    :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant.
                     When ``None`` (Default), preserves the input band dtype.
    :param no_data_val: Optional input no-data value override.
    :param calc_stats: Compute thematic statistics and pyramids on ``output_img``
                       (Default = ``True``).
    :param use_pbar: Display a ``tqdm`` progress bar (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.

    .. code:: python

        import rsgispy.classification.classpostprocess as classpost

        classpost.apply_postprocess_pipeline(
            input_img="classified.tif",
            output_img="classified_clean.tif",
            majority_sizes=(3, 5),
            sieve_size_threshold=10,
            sieve_connectivity=8,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    run_postprocess_pipeline(
        in_path=os.fspath(input_img),
        out_path=os.fspath(output_img),
        majority_sizes=majority_sizes,
        sieve_size_threshold=sieve_size_threshold,
        sieve_connectivity=sieve_connectivity,
        keep_centre_on_tie=keep_centre_on_tie,
        img_band=img_band,
        in_memory=in_memory,
        block_size=block_size,
        copy_colour_table=copy_colour_table,
        gdalformat=gdalformat,
        datatype=datatype,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
