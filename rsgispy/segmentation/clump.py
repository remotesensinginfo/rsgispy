"""
Connected-component clumping and clump-union functions for raster imagery.

Functions:

* :func:`clump` — Identify 4-connected regions of identical pixel values.
* :func:`union_of_clumps` — Combine multiple clump or thematic images so all
  region boundaries from every input band are preserved.
"""

from contextlib import ExitStack
import os
from typing import List, Optional, Sequence, Union

import rsgispy
from rsgispy.imageprocess import (
    compute_spatial_intersection,
    ensure_file_exists,
    open_multi_image_datasets,
)
from rsgispy.segmentation._helpers import BandSource, run_clump_pipeline

__all__ = [
    "clump",
    "union_of_clumps",
]


def clump(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    in_memory: bool = False,
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Clump an input image (band 1) into 4-connected regions of identical pixel
    values, assigning unique sequential clump IDs (``1 .. N``).

    Uses a vectorized horizontal run-length-encoded (RLE) graph solver
    (``scipy.sparse.csgraph`` when installed, with a pure-Python/NumPy RLE
    union-find fallback) in either single-pass in-memory mode or a
    memory-bounded two-pass tiled streaming mode.

    :param input_img: Path to the input raster file.
    :param output_img: Path to the output clump raster file.
    :param in_memory: If ``True``, process the full image in a single in-memory
                      pass. If ``False`` (Default), use memory-bounded two-pass
                      tiled streaming when the image exceeds ``block_size``.
    :param block_size: Optional tile size in pixels for tiled mode. Defaults to
                       :func:`rsgispy.imageprocess.get_default_block_size`.
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32UINT`).
    :param no_data_val: Optional input pixel value treated as no-data and
                        assigned clump ID ``0`` in ``output_img``
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a ``tqdm`` progress bar during tiled
                     processing (Default = ``True``).

    .. code:: python

        import rsgispy.segmentation.clump

        rsgispy.segmentation.clump.clump(
            input_img="classified.tif",
            output_img="clumps.tif",
            in_memory=True,
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, label="Input image")

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_path])
        grid = compute_spatial_intersection(datasets)
        band_sources = [
            BandSource(dataset=datasets[in_path], path=in_path, band_idx=1)
        ]
        run_clump_pipeline(
            band_sources=band_sources,
            grid=grid,
            output_img=out_path,
            gdalformat=gdalformat,
            in_memory=in_memory,
            no_data_val=no_data_val,
            block_size=block_size,
            datatype=datatype,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )


def union_of_clumps(
    input_imgs: Sequence[Union[str, os.PathLike]],
    output_img: Union[str, os.PathLike],
    in_memory: Optional[bool] = None,
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute the union of multiple clump or thematic images, preserving all
    boundaries from every band of every image in ``input_imgs``.

    Adjacent (4-connected) pixels are merged into the same output clump if and
    only if their values are equal across all input images/bands. When
    ``no_data_val`` is specified, pixels where **every** input band equals
    ``no_data_val`` are assigned output ID ``0``.

    :param input_imgs: Non-empty sequence of input raster file paths.
    :param output_img: Path to the output union-clumps raster file.
    :param in_memory: Optional override for single-pass in-memory execution
                      (``True``) vs two-pass tiled streaming (``False``). When
                      ``None`` (Default), selects in-memory for moderate grids
                      and tiled streaming for large grids automatically.
    :param block_size: Optional tile size in pixels for tiled mode. Defaults to
                       :func:`rsgispy.imageprocess.get_default_block_size`.
    :param gdalformat: GDAL/rasterio driver format for ``output_img``
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type
                     (Default = ``None``, which uses
                     :data:`rsgispy.TYPE_32UINT`).
    :param no_data_val: Optional no-data value across all input bands
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute output image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a ``tqdm`` progress bar during tiled
                     processing (Default = ``True``).

    .. code:: python

        import rsgispy.segmentation.clump

        rsgispy.segmentation.clump.union_of_clumps(
            input_imgs=["clumps_a.tif", "clumps_b.tif"],
            output_img="union_clumps.tif",
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not input_imgs:
        raise rsgispy.RSGISPyException("input_imgs must not be empty.")

    out_path = os.fspath(output_img)
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, input_imgs)
        grid = compute_spatial_intersection(datasets)
        band_sources: List[BandSource] = []
        for path in paths:
            ds = datasets[path]
            for b_idx in range(1, ds.count + 1):
                band_sources.append(
                    BandSource(dataset=ds, path=path, band_idx=b_idx)
                )

        run_clump_pipeline(
            band_sources=band_sources,
            grid=grid,
            output_img=out_path,
            gdalformat=gdalformat,
            in_memory=in_memory,
            no_data_val=no_data_val,
            block_size=block_size,
            datatype=datatype,
            calc_stats=calc_stats,
            use_pbar=use_pbar,
        )
