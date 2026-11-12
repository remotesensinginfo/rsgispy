"""
The imagelut module provides utilities for creating and querying spatial
image look-up tables (LUTs) using rasterio and geopandas.
"""

import os
from typing import List, Optional, Sequence, Tuple
import uuid

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

import rsgispy
import rsgispy.imageutils

__all__ = [
    "create_img_extent_lut",
    "query_img_lut",
    "get_all_lut_imgs",
    "get_raster_lyr",
    "query_file_lut",
]


def create_img_extent_lut(
    input_imgs: Sequence[str],
    vec_file: str,
    vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    ignore_none_imgs: bool = False,
    out_proj_wgs84: bool = False,
    overwrite_lut_file: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Create a vector layer look-up table (LUT) storing the spatial extent,
    filename, and directory path for a sequence of raster images.

    :param input_imgs: Sequence of input image file paths. Unless ``out_proj_wgs84``
                       is True, all valid input images must share the same CRS.
    :param vec_file: Output vector file path (e.g., ``'ImgExtents.gpkg'``).
    :param vec_lyr: Output vector layer name (used for multi-layer formats such
                    as GeoPackage). If ``None``, the format default layer is used.
    :param out_format: Output OGR/GeoPandas driver name (Default = ``'GPKG'``).
    :param ignore_none_imgs: If True, images without a defined CRS are skipped;
                             otherwise a :class:`rsgispy.RSGISPyException` is raised
                             (Default = False).
    :param out_proj_wgs84: If True, image bounding boxes are reprojected to
                           ``EPSG:4326`` (Default = False).
    :param overwrite_lut_file: If True, any existing ``vec_file`` is removed
                               before writing (Default = False).
    :param use_pbar: If True, display a tqdm progress bar while scanning images
                     (Default = True).

    .. code:: python

        import glob
        import rsgispy.imageutils.imagelut

        img_list = glob.glob("/path/to/images/*.tif")
        rsgispy.imageutils.imagelut.create_img_extent_lut(
            img_list,
            "ImgExtents.gpkg",
            "ImgExtents",
            "GPKG",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    if not input_imgs:
        raise rsgispy.RSGISPyException(
            "input_imgs must contain at least one image path."
        )

    geoms = []
    filenames: List[str] = []
    paths: List[str] = []
    ref_crs = None
    out_crs = "EPSG:4326" if out_proj_wgs84 else None

    tqdm_cls = rsgispy.get_tqdm()
    for img in tqdm_cls(input_imgs, disable=not use_pbar):
        if not os.path.exists(img):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img}'."
            )

        with rasterio.open(img, "r") as src:
            img_crs = src.crs
            if img_crs is None:
                if not ignore_none_imgs:
                    raise rsgispy.RSGISPyException(
                        f"The CRS/EPSG code is None: '{img}'"
                    )
                continue

            left, bottom, right, top = src.bounds

        if out_proj_wgs84:
            min_x, min_y, max_x, max_y = transform_bounds(
                img_crs, "EPSG:4326", left, bottom, right, top
            )
        else:
            if ref_crs is None:
                ref_crs = img_crs
                out_crs = img_crs
            elif img_crs != ref_crs:
                raise rsgispy.RSGISPyException(
                    f"The image projections ({ref_crs} & {img_crs}) do not match "
                    f"for '{img}'."
                )
            min_x, min_y, max_x, max_y = left, bottom, right, top

        geoms.append(box(min_x, min_y, max_x, max_y))
        filenames.append(os.path.basename(img))
        paths.append(os.path.dirname(img))

    if not geoms:
        raise rsgispy.RSGISPyException(
            "No valid images with a defined CRS were found to create the LUT."
        )

    if overwrite_lut_file and os.path.exists(vec_file):
        os.remove(vec_file)

    parent_dir = os.path.dirname(os.path.abspath(vec_file))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    lut_gdf = gpd.GeoDataFrame(
        {"filename": filenames, "path": paths},
        geometry=geoms,
        crs=out_crs,
    )
    lut_gdf.to_file(vec_file, layer=vec_lyr, driver=out_format)


def query_img_lut(
    scn_bbox: Tuple[float, float, float, float],
    lut_db_file: str,
    lyr_name: Optional[str] = None,
) -> List[str]:
    """
    Query an image LUT spatially using a bounding box ``(minX, maxX, minY, maxY)``
    in the same CRS as the LUT.

    :param scn_bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` defining the area
                     of interest in the LUT projection.
    :param lut_db_file: Path to the LUT vector file (e.g., ``'lut.gpkg'``).
    :param lyr_name: Optional layer name within ``lut_db_file``.
    :return: List of image file paths intersecting ``scn_bbox``.
    """
    if not os.path.exists(lut_db_file):
        raise rsgispy.RSGISPyException(
            f"LUT vector file does not exist: '{lut_db_file}'."
        )
    if len(scn_bbox) < 4:
        raise rsgispy.RSGISPyException(
            "scn_bbox must contain 4 coordinates: (MinX, MaxX, MinY, MaxY)."
        )

    min_x, max_x, min_y, max_y = (float(v) for v in scn_bbox[:4])
    lut_gdf = gpd.read_file(
        lut_db_file,
        layer=lyr_name,
        bbox=(min_x, min_y, max_x, max_y),
    )
    if lut_gdf.empty:
        return []

    return [
        os.path.join(str(p), str(f))
        for p, f in zip(lut_gdf["path"], lut_gdf["filename"])
    ]


def get_all_lut_imgs(
    lut_db_file: str,
    lyr_name: Optional[str] = None,
) -> List[str]:
    """
    Return a list of all image file paths recorded in an image LUT.

    :param lut_db_file: Path to the LUT vector file (e.g., ``'lut.gpkg'``).
    :param lyr_name: Optional layer name within ``lut_db_file``.
    :return: List of all image file paths stored in the LUT.
    """
    if not os.path.exists(lut_db_file):
        raise rsgispy.RSGISPyException(
            f"LUT vector file does not exist: '{lut_db_file}'."
        )

    lut_df = gpd.read_file(
        lut_db_file,
        layer=lyr_name,
        ignore_geometry=True,
    )
    if lut_df.empty:
        return []

    return [
        os.path.join(str(p), str(f))
        for p, f in zip(lut_df["path"], lut_df["filename"])
    ]


def get_raster_lyr(
    scn_bbox: Tuple[float, float, float, float],
    lut_db_file: str,
    lyr_name: Optional[str],
    tmp_dir: str,
) -> Optional[str]:
    """
    Retrieve a single raster layer path for ``scn_bbox`` from an image LUT.

    If a single image intersects ``scn_bbox``, its path is returned directly.
    If multiple images intersect, a mosaic VRT file is generated in ``tmp_dir``
    using :func:`rsgispy.imageutils.create_mosaic_images_vrt` and its path is
    returned. If no images intersect, ``None`` is returned.

    :param scn_bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` in the LUT CRS.
    :param lut_db_file: Path to the LUT vector file (e.g., ``'lut.gpkg'``).
    :param lyr_name: Layer name within ``lut_db_file`` (or ``None``).
    :param tmp_dir: Directory where temporary VRT files will be written when
                    multiple images intersect ``scn_bbox``.
    :return: Path to the single image or generated VRT file, or ``None``.
    """
    imgs = query_img_lut(
        scn_bbox=scn_bbox,
        lut_db_file=lut_db_file,
        lyr_name=lyr_name,
    )
    if len(imgs) > 1:
        os.makedirs(tmp_dir, exist_ok=True)
        img_uid = uuid.uuid4().hex[:8]
        out_vrt_file = os.path.join(tmp_dir, f"imglyr_{img_uid}_tmp.vrt")
        rsgispy.imageutils.create_mosaic_images_vrt(
            input_imgs=imgs,
            out_vrt_file=out_vrt_file,
        )
        return out_vrt_file
    if len(imgs) == 1:
        return imgs[0]
    return None


def query_file_lut(
    lut_db_file: str,
    lyr_name: Optional[str],
    roi_file: str,
    roi_lyr: Optional[str] = None,
    out_dest: Optional[str] = None,
    targz_out: bool = False,
    cp_cmds: bool = False,
) -> List[str]:
    """
    Query a file LUT by spatial intersection with a region-of-interest (ROI)
    vector layer and optionally format shell commands for archiving or copying
    the intersecting files.

    Set at most one of ``targz_out`` or ``cp_cmds`` to True. If both are False,
    the list of intersecting file paths is returned directly.

    :param lut_db_file: Path to the LUT vector file.
    :param lyr_name: Layer name within ``lut_db_file`` (or ``None`` for default layer).
    :param roi_file: Path to the region-of-interest vector file.
    :param roi_lyr: Layer name within ``roi_file`` (or ``None`` for default layer).
    :param out_dest: Destination path for output archive or copy directory when
                     ``targz_out`` or ``cp_cmds`` is True.
    :param targz_out: If True, return a single ``tar -czf`` command string containing
                      all intersecting files (Default = False).
    :param cp_cmds: If True, return a list of ``cp <src> <out_dest>`` command
                    strings (Default = False).
    :return: List of intersecting file paths or shell command strings.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if targz_out and cp_cmds:
        raise rsgispy.RSGISPyException(
            "Cannot set both targz_out and cp_cmds to True."
        )
    if (targz_out or cp_cmds) and not out_dest:
        raise rsgispy.RSGISPyException(
            "out_dest must be provided when targz_out or cp_cmds is True."
        )
    if not os.path.exists(lut_db_file):
        raise rsgispy.RSGISPyException(
            f"LUT vector file does not exist: '{lut_db_file}'."
        )
    if not os.path.exists(roi_file):
        raise rsgispy.RSGISPyException(
            f"ROI vector file does not exist: '{roi_file}'."
        )

    roi_gdf = gpd.read_file(roi_file, layer=roi_lyr)
    if roi_gdf.empty:
        return []

    lut_hdr = gpd.read_file(lut_db_file, layer=lyr_name, rows=0)
    if (
        lut_hdr.crs is not None
        and roi_gdf.crs is not None
        and lut_hdr.crs != roi_gdf.crs
    ):
        roi_gdf = roi_gdf.to_crs(lut_hdr.crs)

    roi_bounds = tuple(float(v) for v in roi_gdf.total_bounds)
    lut_gdf = gpd.read_file(lut_db_file, layer=lyr_name, bbox=roi_bounds)
    if lut_gdf.empty:
        return []

    roi_union = roi_gdf.geometry.union_all()
    matched_gdf = lut_gdf[lut_gdf.intersects(roi_union)]
    if matched_gdf.empty:
        return []

    file_paths = [
        os.path.join(str(p), str(f))
        for p, f in zip(matched_gdf["path"], matched_gdf["filename"])
    ]

    if targz_out:
        return [f"tar -czf {out_dest} " + " ".join(file_paths)]
    if cp_cmds:
        return [f"cp {file_path} {out_dest}" for file_path in file_paths]
    return file_paths
