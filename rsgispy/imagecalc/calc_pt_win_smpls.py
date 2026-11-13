"""
Provides classes and functions to calculate image summary values for a window
around each point in a vector point dataset (e.g., DEM roughness metrics).
"""

from abc import ABCMeta, abstractmethod
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset, ensure_file_exists
import rsgispy.tools.geometry

__all__ = [
    "RSGISCalcSumVals",
    "RSGISDebugExportImg",
    "calc_pt_smpl_img_vals",
]


class RSGISCalcSumVals(metaclass=ABCMeta):
    """
    Abstract base class for calculating summary values for a ``rasterio``
    image dataset window provided to :func:`calc_pt_smpl_img_vals`.

    :param self.n_out_vals: Number of output values (length of ``out_val_names``).
    :param self.out_val_names: List of output attribute column names.
    """

    def __init__(self) -> None:
        self.n_out_vals: int = 1
        self.out_val_names: List[str] = ["value"]

    @abstractmethod
    def calcVals(  # pylint: disable=invalid-name
        self, smpl_idx: int, in_img_ds_obj: rasterio.DatasetReader
    ) -> Dict[str, float]:
        """
        Abstract method to calculate summary values from an extracted window.

        :param smpl_idx: Unique index for the point sample being processed.
        :param in_img_ds_obj: Open ``rasterio.DatasetReader`` for the sample window.
        :return: Dictionary mapping output attribute names to float values.
        """

    def getNOutVals(self) -> int:  # pylint: disable=invalid-name
        """
        Return the number of output summary values produced by this calculator.
        """
        return self.n_out_vals

    def getOutValNames(self) -> List[str]:  # pylint: disable=invalid-name
        """
        Return the list of output attribute names produced by this calculator.
        """
        return self.out_val_names


def _extract_rotated_window(
    src_ds: rasterio.DatasetReader,
    img_bbox: Tuple[float, float, float, float],
    pt_xy: Tuple[float, float],
    pt_xbox: float,
    pt_ybox: float,
    pt_angle: float,
    rio_interp: Resampling,
    no_data_val: Optional[float],
) -> Tuple[np.ndarray, Affine, int, int]:
    """
    Extract and optionally rotate a raster window centred on ``pt_xy``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    img_x_res = float(src_ds.transform.a)
    img_y_res = float(src_ds.transform.e)
    img_y_res_abs = abs(img_y_res)

    pt_grid = rsgispy.tools.geometry.find_point_on_whole_num_grid(
        pt_xy, img_bbox, img_x_res, img_y_res_abs
    )

    img_xbox = float(pt_xbox) * img_x_res
    img_ybox = float(pt_ybox) * img_y_res_abs
    img_max_box = max(img_xbox, img_ybox)

    pt_img_bbox = (
        pt_grid[0] - img_xbox,
        pt_grid[0] + img_xbox,
        pt_grid[1] - img_ybox,
        pt_grid[1] + img_ybox,
    )
    pt_img_buf_bbox = rsgispy.tools.geometry.buffer_bbox(
        pt_img_bbox, buf=(img_max_box * 2.0)
    )
    pt_img_bound_bbox = rsgispy.tools.geometry.bbox_intersection(
        pt_img_bbox, img_bbox
    )
    pt_img_bound_buf_bbox = rsgispy.tools.geometry.bbox_intersection(
        pt_img_buf_bbox, img_bbox
    )

    sub_width = int(
        math.ceil((pt_img_bound_buf_bbox[1] - pt_img_bound_buf_bbox[0]) / img_x_res)
    )
    sub_height = int(
        math.ceil(
            (pt_img_bound_buf_bbox[3] - pt_img_bound_buf_bbox[2]) / img_y_res_abs
        )
    )
    if sub_width < 1 or sub_height < 1:
        raise rsgispy.RSGISPyException(
            f"Subset window size is less than 1: {sub_width} x {sub_height}."
        )

    img_pxl_x_diff = int(
        math.floor(((pt_img_bound_buf_bbox[0] - img_bbox[0]) / img_x_res) + 0.5)
    )
    img_pxl_y_diff = int(
        math.floor(((img_bbox[3] - pt_img_bound_buf_bbox[3]) / img_y_res_abs) + 0.5)
    )

    fill_val = 0 if no_data_val is None else no_data_val
    sub_win = Window(img_pxl_x_diff, img_pxl_y_diff, sub_width, sub_height)
    sub_arr = src_ds.read(window=sub_win, boundless=True, fill_value=fill_val)

    pt_img_affine_src = Affine.from_gdal(
        pt_img_bound_buf_bbox[0],
        img_x_res,
        0.0,
        pt_img_bound_buf_bbox[3],
        0.0,
        img_y_res,
    )
    pt_img_affine_dst = pt_img_affine_src @ Affine.rotation(
        pt_angle, pivot=(sub_width / 2.0, sub_height / 2.0)
    )

    roi_x_coords = pt_img_bound_bbox[1] - pt_img_bound_bbox[0]
    roi_y_coords = pt_img_bound_bbox[3] - pt_img_bound_bbox[2]
    roi_width = int(math.ceil((roi_x_coords / img_x_res) + 0.5))
    roi_height = int(math.ceil((roi_y_coords / img_y_res_abs) + 0.5))
    if roi_width < 1 or roi_height < 1:
        raise rsgispy.RSGISPyException(
            f"ROI window size is less than 1: {roi_width} x {roi_height}."
        )

    roi_transform = Affine.from_gdal(
        pt_img_bound_bbox[0],
        img_x_res,
        0.0,
        pt_img_bound_bbox[3],
        0.0,
        img_y_res,
    )
    roi_arr = np.full(
        (src_ds.count, roi_height, roi_width),
        fill_val,
        dtype=src_ds.dtypes[0],
    )

    warp_crs = src_ds.crs if src_ds.crs is not None else "EPSG:32630"
    reproject(
        source=sub_arr,
        destination=roi_arr,
        src_transform=pt_img_affine_dst,
        src_crs=warp_crs,
        src_nodata=no_data_val,
        dst_transform=roi_transform,
        dst_crs=warp_crs,
        dst_nodata=no_data_val,
        resampling=rio_interp,
    )
    return roi_arr, roi_transform, roi_width, roi_height


def calc_pt_smpl_img_vals(
    input_img: str,
    vec_file: str,
    vec_lyr: Optional[str],
    calc_objs: Sequence[RSGISCalcSumVals],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    interp_method: Union[int, Resampling] = rsgispy.INTERP_CUBIC,
    angle_col: Optional[str] = None,
    x_box_col: str = "xbox",
    y_box_col: str = "ybox",
    no_data_val: Optional[float] = None,
) -> None:
    """
    Calculate summary values for an image window around each point in an input
    point vector layer using one or more :class:`RSGISCalcSumVals` calculators.

    The window size is taken from ``x_box_col`` and ``y_box_col`` (half-widths in
    image pixels) and can optionally be rotated using ``angle_col`` (degrees
    relative to north).

    :param input_img: Path to the input image file.
    :param vec_file: Path to the input point vector file.
    :param vec_lyr: Input vector layer name (or ``None`` for default layer).
    :param calc_objs: Sequence of :class:`RSGISCalcSumVals` instances.
    :param out_vec_file: Path to the output vector file.
    :param out_vec_lyr: Output vector layer name (used for multi-layer formats
                        such as GeoPackage).
    :param out_format: Output vector format/driver (e.g., ``'GPKG'``, ``'GeoJSON'``).
    :param interp_method: Interpolation method used when resampling/rotating the
                          window. Default: ``rsgispy.INTERP_CUBIC``.
    :param angle_col: Optional column name specifying rotation angle in degrees.
    :param x_box_col: Column name specifying half-window width in X pixels.
    :param y_box_col: Column name specifying half-window height in Y pixels.
    :param no_data_val: Image no-data value. If ``None``, taken from image header.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    ensure_file_exists(vec_file, label="Input vector")

    out_col_names: List[str] = []
    out_vals_dict: Dict[str, List[float]] = {}
    for calc_obj in calc_objs:
        col_names_tmp = calc_obj.getOutValNames()
        for out_col in col_names_tmp:
            out_vals_dict[out_col] = []
        out_col_names.extend(col_names_tmp)

    rio_interp = rsgispy.get_rasterio_interp_type(interp_method)
    read_kwargs = {"layer": vec_lyr} if vec_lyr is not None else {}
    pts_gdf = geopandas.read_file(vec_file, **read_kwargs)
    tqdm_cls = rsgispy.get_tqdm()

    with rasterio.open(input_img, "r") as src_ds:
        eff_nodata = no_data_val if no_data_val is not None else src_ds.nodata
        img_bbox = bbox_from_dataset(src_ds)
        img_dtype = src_ds.dtypes[0]
        img_crs = src_ds.crs
        img_count = src_ds.count

        for i, row in tqdm_cls(pts_gdf.iterrows(), total=len(pts_gdf)):
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                raise rsgispy.RSGISPyException(
                    f"Feature {i} has an empty geometry."
                )

            pt_angle = float(row[angle_col]) if angle_col is not None else 0.0
            if pt_angle < 0.0:
                raise rsgispy.RSGISPyException("The angle needs to be >= 0.0.")

            roi_arr, roi_transform, roi_width, roi_height = (
                _extract_rotated_window(
                    src_ds=src_ds,
                    img_bbox=img_bbox,
                    pt_xy=(float(geom.x), float(geom.y)),
                    pt_xbox=float(row[x_box_col]),
                    pt_ybox=float(row[y_box_col]),
                    pt_angle=pt_angle,
                    rio_interp=rio_interp,
                    no_data_val=eff_nodata,
                )
            )

            with MemoryFile() as memfile:
                with memfile.open(
                    driver="GTiff",
                    height=roi_height,
                    width=roi_width,
                    count=img_count,
                    dtype=img_dtype,
                    crs=img_crs,
                    transform=roi_transform,
                    nodata=eff_nodata,
                ) as tmp_dst:
                    tmp_dst.write(roi_arr)
                with memfile.open() as roi_ds:
                    for calc_obj in calc_objs:
                        rtn_vals = calc_obj.calcVals(i, roi_ds)
                        for out_col in calc_obj.getOutValNames():
                            out_vals_dict[out_col].append(rtn_vals[out_col])

    for out_col in out_col_names:
        pts_gdf[out_col] = out_vals_dict[out_col]

    if out_format.upper() == "GPKG" and out_vec_lyr is not None:
        pts_gdf.to_file(out_vec_file, layer=out_vec_lyr, driver=out_format)
    else:
        pts_gdf.to_file(out_vec_file, driver=out_format)


class RSGISDebugExportImg(RSGISCalcSumVals):
    """
    Debug calculator class that exports each sampled raster window as a GeoTIFF.
    """

    def __init__(self, out_path: str, img_name: str) -> None:
        super().__init__()
        self.n_out_vals = 1
        self.out_val_names = ["debug"]
        self.out_path = str(out_path)
        self.img_name = img_name

    def calcVals(  # pylint: disable=invalid-name
        self, smpl_idx: int, in_img_ds_obj: rasterio.DatasetReader
    ) -> Dict[str, float]:
        """
        Write ``in_img_ds_obj`` to ``{out_path}/{smpl_idx}_{img_name}.tif``.

        :param smpl_idx: Unique index for the point sample being processed.
        :param in_img_ds_obj: Open ``rasterio.DatasetReader`` for the sample window.
        :return: Dictionary containing ``{'debug': 0.0}``.
        """
        out_img = os.path.join(self.out_path, f"{smpl_idx}_{self.img_name}.tif")
        profile = in_img_ds_obj.profile.copy()
        profile["driver"] = "GTiff"
        with rasterio.open(out_img, "w", **profile) as dst:
            dst.write(in_img_ds_obj.read())
        return {"debug": 0.0}
