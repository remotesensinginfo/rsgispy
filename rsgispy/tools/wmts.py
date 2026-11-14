"""
The tools.wmts module provides utilities for querying and downloading
Web Map Tile Service (WMTS) layers using ``owslib`` and ``rasterio`` without
requiring GDAL's ``osgeo`` bindings.
"""

from contextlib import ExitStack
import io
import math
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy
from PIL import Image
import pyproj
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import IntersectionGrid, open_output_dataset
import rsgispy.imageutils.subsetting
import rsgispy.imageutils.warp
import rsgispy.tools.files
import rsgispy.tools.geometry
import rsgispy.tools.strings

__all__ = [
    "get_wmts_layer_list",
    "get_wmts_as_img",
]


def get_wmts_layer_list(
    wmts_url: str,
    name_filter: Optional[str] = None,
) -> List[str]:
    """
    Retrieve the list of available layer identifiers from a WMTS service.

    :param wmts_url: URL of the WMTS service (GetCapabilities endpoint).
    :param name_filter: Optional substring filter applied to layer identifiers.
    :return: List of matching layer identifiers.

    .. code:: python

        import rsgispy.tools.wmts
        layers = rsgispy.tools.wmts.get_wmts_layer_list(
            "https://example.com/wmts/1.0.0/WMTSCapabilities.xml"
        )
    """
    # pylint: disable=import-outside-toplevel
    import owslib.wmts

    wmts_obj = owslib.wmts.WebMapTileService(wmts_url)
    wmts_lyrs = list(wmts_obj.contents)
    if name_filter is not None:
        wmts_lyrs = [lyr for lyr in wmts_lyrs if name_filter in lyr]
    return wmts_lyrs


def _select_tile_matrix_set(
    wmts_obj: Any,
    lyr_obj: Any,
    wmts_epsg: Optional[int],
) -> Tuple[str, Any, pyproj.CRS, int]:
    """Select the best matching TileMatrixSet and resolve its CRS/EPSG."""
    candidate_sets = list(getattr(lyr_obj, "tilematrixsets", []) or [])
    if not candidate_sets and getattr(lyr_obj, "tilematrixsetlinks", None):
        candidate_sets = list(lyr_obj.tilematrixsetlinks.keys())
    if not candidate_sets:
        candidate_sets = list(wmts_obj.tilematrixsets.keys())
    if not candidate_sets:
        raise rsgispy.RSGISPyException(
            "No TileMatrixSet definitions found for the requested WMTS layer."
        )

    chosen_name = candidate_sets[0]
    if wmts_epsg is not None:
        for name in candidate_sets:
            tms = wmts_obj.tilematrixsets.get(name)
            if tms is not None:
                try:
                    crs = pyproj.CRS.from_user_input(tms.crs)
                    if crs.to_epsg() == int(wmts_epsg):
                        chosen_name = name
                        break
                except Exception:  # pylint: disable=broad-exception-caught
                    continue

    tms_obj = wmts_obj.tilematrixsets[chosen_name]
    tms_crs = pyproj.CRS.from_user_input(tms_obj.crs)
    resolved_epsg = tms_crs.to_epsg() or wmts_epsg
    if resolved_epsg is None:
        raise rsgispy.RSGISPyException(
            "Could not determine EPSG code for WMTS TileMatrixSet; please supply wmts_epsg."
        )
    return chosen_name, tms_obj, tms_crs, int(resolved_epsg)


def _select_tile_matrix(
    tms_obj: Any,
    zoom_level: Optional[Union[int, str]],
    wmts_bbox: Sequence[float],
    tms_crs: pyproj.CRS,
) -> Any:
    """Select the WMTS TileMatrix for ``zoom_level`` or auto-select by extent."""
    matrices = sorted(
        tms_obj.tilematrix.values(),
        key=lambda tm: float(tm.scaledenominator),
        reverse=True,
    )
    if not matrices:
        raise rsgispy.RSGISPyException("TileMatrixSet contains no TileMatrix levels.")

    if zoom_level is not None:
        zoom_str = str(zoom_level)
        for tm in matrices:
            if str(tm.identifier) == zoom_str:
                return tm
        if isinstance(zoom_level, int) and 0 <= zoom_level < len(matrices):
            return matrices[zoom_level]
        raise rsgispy.RSGISPyException(
            f"Zoom level '{zoom_level}' was not found in TileMatrixSet."
        )

    m_per_unit = _metres_per_crs_unit(tms_crs)
    bbox_w = max(1e-9, float(wmts_bbox[1]) - float(wmts_bbox[0]))
    selected = matrices[0]
    for tm in matrices:
        pxl_size = (float(tm.scaledenominator) * 0.00028) / m_per_unit
        n_px_x = bbox_w / pxl_size
        selected = tm
        if n_px_x >= float(tm.tilewidth):
            break
    return selected


def _metres_per_crs_unit(crs_obj: pyproj.CRS) -> float:
    """Return approximate metres per linear/angular CRS axis unit."""
    if crs_obj.is_geographic:
        return 6378137.0 * math.pi / 180.0
    factor = crs_obj.axis_info[0].unit_conversion_factor if crs_obj.axis_info else 1.0
    return float(factor or 1.0)


def _resolve_origin_xy(
    topleftcorner: Sequence[float],
    tms_crs: pyproj.CRS,
) -> Tuple[float, float]:
    """Return ``(origin_x, origin_y)`` accounting for EPSG:4326 axis order."""
    c0, c1 = float(topleftcorner[0]), float(topleftcorner[1])
    if (
        tms_crs.is_geographic
        and tms_crs.axis_info
        and tms_crs.axis_info[0].direction == "north"
    ):
        return c1, c0
    return c0, c1


def get_wmts_as_img(
    wmts_url: str,
    wmts_lyr: str,
    bbox: Sequence[float],
    bbox_epsg: int,
    output_img: str,
    gdalformat: str = "GTIFF",
    zoom_level: Optional[Union[int, str]] = None,
    tmp_dir: Optional[str] = None,
    wmts_epsg: Optional[int] = None,
) -> None:
    """
    Download and mosaic WMTS tiles covering ``bbox`` to an output raster image.

    :param wmts_url: URL of the WMTS service (GetCapabilities endpoint).
    :param wmts_lyr: Layer identifier within the WMTS service.
    :param bbox: Bounding box ``[MinX, MaxX, MinY, MaxY]`` in ``bbox_epsg``.
    :param bbox_epsg: EPSG code of ``bbox`` and the output image.
    :param output_img: Output raster file path.
    :param gdalformat: Output GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param zoom_level: Optional TileMatrix identifier or zero-based zoom index.
    :param tmp_dir: Optional temporary directory for intermediate files.
    :param wmts_epsg: Optional EPSG code override for the WMTS layer.

    .. code:: python

        import rsgispy.tools.wmts
        rsgispy.tools.wmts.get_wmts_as_img(
            "https://example.com/wmts/1.0.0/WMTSCapabilities.xml",
            wmts_lyr="imagery",
            bbox=[-4.5, -4.0, 52.0, 52.5],
            bbox_epsg=4326,
            output_img="wmts_tile.tif",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    import owslib.wmts

    wmts_obj = owslib.wmts.WebMapTileService(wmts_url)
    if wmts_lyr not in wmts_obj.contents:
        raise rsgispy.RSGISPyException(
            f"Layer '{wmts_lyr}' was not found in WMTS service '{wmts_url}'."
        )
    lyr_obj = wmts_obj.contents[wmts_lyr]

    tms_name, tms_obj, tms_crs, resolved_wmts_epsg = _select_tile_matrix_set(
        wmts_obj, lyr_obj, wmts_epsg
    )
    if resolved_wmts_epsg != int(bbox_epsg):
        wmts_bbox = rsgispy.tools.geometry.reproj_bbox_epsg(
            tuple(bbox[:4]), int(bbox_epsg), resolved_wmts_epsg
        )
    else:
        wmts_bbox = tuple(float(v) for v in bbox[:4])

    tm = _select_tile_matrix(tms_obj, zoom_level, wmts_bbox, tms_crs)
    pxl_res = (float(tm.scaledenominator) * 0.00028) / _metres_per_crs_unit(tms_crs)
    tile_w = int(tm.tilewidth)
    tile_h = int(tm.tileheight)
    tile_span_x = tile_w * pxl_res
    tile_span_y = tile_h * pxl_res
    orig_x, orig_y = _resolve_origin_xy(tm.topleftcorner, tms_crs)

    max_col = max(0, int(tm.matrixwidth) - 1)
    max_row = max(0, int(tm.matrixheight) - 1)
    col_min = max(0, min(max_col, int(math.floor((wmts_bbox[0] - orig_x) / tile_span_x))))
    col_max = max(0, min(max_col, int(math.floor((wmts_bbox[1] - orig_x) / tile_span_x))))
    row_min = max(0, min(max_row, int(math.floor((orig_y - wmts_bbox[3]) / tile_span_y))))
    row_max = max(0, min(max_row, int(math.floor((orig_y - wmts_bbox[2]) / tile_span_y))))

    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    img_formats = list(getattr(lyr_obj, "formats", []) or ["image/png"])
    tile_fmt = "image/png" if "image/png" in img_formats else img_formats[0]

    mosaic_ul_x = orig_x + col_min * tile_span_x
    mosaic_ul_y = orig_y - row_min * tile_span_y
    mosaic_transform = from_origin(mosaic_ul_x, mosaic_ul_y, pxl_res, pxl_res)

    uid_str = rsgispy.tools.strings.uid_generator()
    with rsgispy.tools.files.work_dir_context(tmp_dir) as work_dir:
        mosaic_img = os.path.join(work_dir, f"wmts_mosaic_{uid_str}.tif")
        mosaic_grid = IntersectionGrid(
            crs=rasterio.crs.CRS.from_epsg(resolved_wmts_epsg),
            transform=mosaic_transform,
            width=n_cols * tile_w,
            height=n_rows * tile_h,
            offsets={},
        )

        dst_stack = ExitStack()
        dst = None
        n_bands: Optional[int] = None
        out_rio_dtype: Optional[str] = None
        try:
            for r_idx, row in enumerate(range(row_min, row_max + 1)):
                for c_idx, col in enumerate(range(col_min, col_max + 1)):
                    tile_resp = wmts_obj.gettile(
                        layer=wmts_lyr,
                        tilematrixset=tms_name,
                        tilematrix=str(tm.identifier),
                        row=row,
                        column=col,
                        format=tile_fmt,
                    )
                    tile_bytes = tile_resp.read()
                    with Image.open(io.BytesIO(tile_bytes)) as pil_tile:
                        if pil_tile.mode in ("P", "LA"):
                            pil_tile = pil_tile.convert("RGBA")
                        tile_arr = numpy.asarray(pil_tile)
                    if tile_arr.ndim == 2:
                        tile_arr = tile_arr[numpy.newaxis, :, :]
                    else:
                        tile_arr = numpy.moveaxis(tile_arr, -1, 0)

                    if dst is None:
                        n_bands = int(tile_arr.shape[0])
                        out_rio_dtype = str(tile_arr.dtype)
                        dst = open_output_dataset(
                            stack=dst_stack,
                            output_img=mosaic_img,
                            gdalformat="GTIFF",
                            out_rio_dtype=out_rio_dtype,
                            grid=mosaic_grid,
                            count=n_bands,
                        )
                    dst.write(
                        tile_arr,
                        window=Window(c_idx * tile_w, r_idx * tile_h, tile_w, tile_h),
                    )
        finally:
            dst_stack.close()

        clipped_img = os.path.join(work_dir, f"wmts_clip_{uid_str}.tif")
        if resolved_wmts_epsg != int(bbox_epsg):
            rsgispy.imageutils.subsetting.subset_bbox(
                input_img=mosaic_img,
                output_img=clipped_img,
                gdalformat="GTIFF",
                datatype=rsgispy.get_rsgis_datatype(out_rio_dtype or "uint8"),
                min_x=wmts_bbox[0],
                max_x=wmts_bbox[1],
                min_y=wmts_bbox[2],
                max_y=wmts_bbox[3],
            )
            rsgispy.imageutils.warp.gdal_warp(
                input_img=clipped_img,
                output_img=output_img,
                out_epsg=int(bbox_epsg),
                interp_method=rsgispy.INTERP_CUBIC,
                gdalformat=gdalformat,
                use_multi_threaded=False,
            )
        else:
            rsgispy.imageutils.subsetting.subset_bbox(
                input_img=mosaic_img,
                output_img=output_img,
                gdalformat=gdalformat,
                datatype=rsgispy.get_rsgis_datatype(out_rio_dtype or "uint8"),
                min_x=wmts_bbox[0],
                max_x=wmts_bbox[1],
                min_y=wmts_bbox[2],
                max_y=wmts_bbox[3],
            )
