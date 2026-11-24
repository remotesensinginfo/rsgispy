"""
Raster dataset opening and output-creation context helpers for
:mod:`rsgispy.imageprocess`.
"""

from contextlib import ExitStack
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import rasterio
from rasterio.io import DatasetReader, DatasetWriter

import rsgispy
from rsgispy.imageprocess._arrays import BandDefn, IntersectionGrid
from rsgispy.imageprocess._dtype_nodata import get_gdal_img_creation_opts


def open_input_datasets(
    stack: ExitStack, norm_defs: Sequence[BandDefn]
) -> Dict[str, DatasetReader]:
    """
    Open and validate all unique input raster datasets within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param norm_defs: Validated sequence of BandDefn objects.
    :return: Mapping of file path to opened rasterio DatasetReader.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        band_defs = [rsgispy.imageprocess.BandDefn("b1", "img.tif", 1)]
        with ExitStack() as stack:
            datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
    """
    datasets: Dict[str, DatasetReader] = {}
    for b_def in norm_defs:
        if b_def.input_img is None or b_def.img_band is None:
            raise rsgispy.RSGISPyException(
                "BandDefn must define both input_img and img_band."
            )
        if b_def.input_img not in datasets:
            try:
                datasets[b_def.input_img] = stack.enter_context(
                    rasterio.open(b_def.input_img, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{b_def.input_img}': {err}"
                ) from err

        ds = datasets[b_def.input_img]
        if b_def.img_band > ds.count:
            raise rsgispy.RSGISPyException(
                f"Requested band {b_def.img_band} exceeds band count ({ds.count}) "
                f"for image '{b_def.input_img}'."
            )
    return datasets


def open_multi_image_datasets(
    stack: ExitStack,
    input_imgs: Sequence[Union[str, os.PathLike]],
) -> Tuple[Dict[str, DatasetReader], List[str]]:
    """
    Open and validate all unique input raster datasets from a sequence of
    image paths within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param input_imgs: Sequence of input raster file paths.
    :return: Tuple of ``(datasets, paths)`` where ``datasets`` maps each unique
             file path to an opened rasterio DatasetReader and ``paths`` is the
             resolved list of file paths preserving the input order.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, paths = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["a.tif", "b.tif"]
            )
    """
    if not input_imgs:
        raise rsgispy.RSGISPyException("input_imgs must not be empty.")
    paths: List[str] = []
    datasets: Dict[str, DatasetReader] = {}
    for img in input_imgs:
        img_path = os.fspath(img)
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img_path}'."
            )
        paths.append(img_path)
        if img_path not in datasets:
            try:
                datasets[img_path] = stack.enter_context(
                    rasterio.open(img_path, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{img_path}': {err}"
                ) from err
    return datasets, paths


def open_output_dataset(
    stack: ExitStack,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    grid: IntersectionGrid,
    count: int = 1,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an existing output raster for update or create a new output raster.

    Automatically applies any format creation options configured via
    :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` (such as
    ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>``) as well as any explicit
    ``creation_opts`` dictionary passed to this function.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., 'GTIFF').
    :param out_rio_dtype: Rasterio dtype string (e.g., 'float32').
    :param grid: IntersectionGrid defining output dimensions, transform, and CRS.
    :param count: Number of output bands when creating a new file (Default = 1).
    :param output_exists: If True and the file exists, open in 'r+' mode.
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``).
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, _ = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["in.tif"]
            )
            grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out.tif", "GTIFF", "float32", grid, count=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if output_exists and os.path.exists(output_img):
        open_opts: Dict[str, Any] = {}
        if gdalformat.upper().strip() == "COG":
            open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"
        dst = stack.enter_context(rasterio.open(output_img, "r+", **open_opts))
        if dst.width != grid.width or dst.height != grid.height:
            raise rsgispy.RSGISPyException(
                f"Existing output image '{output_img}' dimensions "
                f"({dst.width}x{dst.height}) do not match calculated "
                f"output dimensions ({grid.width}x{grid.height})."
            )
        return dst

    img_creation_opts: Dict[str, Any] = get_gdal_img_creation_opts(gdalformat)
    if creation_opts is not None:
        img_creation_opts.update(creation_opts)

    profile: Dict[str, Any] = {
        "driver": gdalformat,
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": out_rio_dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        **img_creation_opts,
    }
    return stack.enter_context(rasterio.open(output_img, "w", **profile))


def create_output_from_dataset(
    stack: ExitStack,
    src: DatasetReader,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    count: Optional[int] = None,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an output raster matching the spatial extent, transform, and CRS of an
    opened reference dataset ``src``.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param src: Reference opened rasterio DatasetReader.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., ``'GTIFF'``).
    :param out_rio_dtype: Rasterio dtype string (e.g., ``'float32'``).
    :param count: Number of output bands when creating a new file
                  (defaults to ``src.count`` if ``None``).
    :param output_exists: If ``True`` and the file exists, open in ``'r+'`` mode.
    :param creation_opts: Optional mapping of GDAL creation options.
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rasterio
        import rsgispy.imageprocess

        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open("in.tif", "r"))
            dst = rsgispy.imageprocess.create_output_from_dataset(
                stack, src, "out.tif", "GTIFF", "float32", count=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid = IntersectionGrid.from_dataset(src)
    return open_output_dataset(
        stack=stack,
        output_img=output_img,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        grid=grid,
        count=int(src.count) if count is None else int(count),
        output_exists=output_exists,
        creation_opts=creation_opts,
    )
