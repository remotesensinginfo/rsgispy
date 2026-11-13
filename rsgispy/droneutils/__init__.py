"""
The droneutils module provides utilities for organising drone imagery,
building flightline point and line vector layers from EXIF or external GPS
metadata, and writing external GPS/orientation tags into image EXIF headers.

Functions requiring ``exiftool`` are only defined and exported when
``exiftool`` is installed.
"""

import datetime
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas
import pandas
import shapely.geometry
import tqdm
from PIL import Image
from PIL.ExifTags import TAGS

import rsgispy.tools.files

try:
    import exiftool  # pylint: disable=import-error

    HAVE_EXIFTOOL = True
except ImportError:
    HAVE_EXIFTOOL = False

__all__ = [
    "ImageTimeCluster",
    "split_photos_by_time",
]

if HAVE_EXIFTOOL:
    __all__.extend(
        [
            "create_flightline_vec",
            "create_flightline_extern_gps_vec",
            "define_extern_gps_in_imgs",
        ]
    )


class ImageTimeCluster:
    """
    A class which supports the ``split_photos_by_time`` function by grouping
    image acquisitions within a temporal tolerance.
    """

    def __init__(
        self,
        cluster_id: Optional[int] = None,
        init_time: Optional[datetime.datetime] = None,
    ):
        """
        Initialise a time cluster.

        :param cluster_id: Unique ID for the cluster.
        :param init_time: Initial timestamp of the first image.
        """
        self.cluster_id = cluster_id
        self.start_time = init_time
        self.end_time = init_time
        self.images: List[str] = []

    def within_cluster(
        self,
        date_time_obj: datetime.datetime,
        time_split_secs: float = 60,
    ) -> bool:
        """
        Check whether ``date_time_obj`` falls within ``time_split_secs`` of
        this cluster's start or end time (or inside its span).
        """
        if self.start_time is None or self.end_time is None:
            return False
        if self.start_time <= date_time_obj <= self.end_time:
            return True
        if abs((self.start_time - date_time_obj).total_seconds()) < time_split_secs:
            return True
        if abs((self.end_time - date_time_obj).total_seconds()) < time_split_secs:
            return True
        return False

    def add_image_to_cluster(
        self,
        image_file: str,
        date_time_obj: datetime.datetime,
    ) -> None:
        """Add an image file and update the cluster time bounds."""
        self.images.append(image_file)
        if self.start_time is None or date_time_obj < self.start_time:
            self.start_time = date_time_obj
        if self.end_time is None or date_time_obj > self.end_time:
            self.end_time = date_time_obj

    def does_cluster_overlap(
        self,
        cluster_obj: "ImageTimeCluster",
        time_split_secs: float = 60,
    ) -> bool:
        """
        Check whether another :class:`ImageTimeCluster` overlaps or lies within
        ``time_split_secs`` of this cluster.
        """
        if (
            self.start_time is None
            or self.end_time is None
            or cluster_obj.start_time is None
            or cluster_obj.end_time is None
        ):
            return False

        endpoints = (
            self.start_time - cluster_obj.start_time,
            self.end_time - cluster_obj.end_time,
            self.start_time - cluster_obj.end_time,
            self.end_time - cluster_obj.start_time,
        )
        if any(abs(diff.total_seconds()) < time_split_secs for diff in endpoints):
            return True

        if self.start_time < cluster_obj.start_time < self.end_time:
            return True
        if self.start_time < cluster_obj.end_time < self.end_time:
            return True
        if cluster_obj.start_time < self.start_time < cluster_obj.end_time:
            return True
        return False

    def merge_cluster(self, cluster_obj: "ImageTimeCluster") -> None:
        """Merge another cluster's images and time bounds into this cluster."""
        self.images = self.images + cluster_obj.images
        if cluster_obj.start_time is not None and (
            self.start_time is None or cluster_obj.start_time < self.start_time
        ):
            self.start_time = cluster_obj.start_time
        if cluster_obj.end_time is not None and (
            self.end_time is None or cluster_obj.end_time > self.end_time
        ):
            self.end_time = cluster_obj.end_time

    def __str__(self) -> str:
        return (
            f"{self.cluster_id}: {self.start_time} - {self.end_time}"
            f" = {len(self.images)}"
        )

    def __repr__(self) -> str:
        return self.__str__()


def _merge_overlapping_clusters(
    img_clusters: List[ImageTimeCluster],
    time_split_secs: float,
) -> None:
    """Iteratively merge overlapping or adjacent clusters in-place."""
    merged = True
    while merged:
        merged = False
        for i, cluster_out in enumerate(img_clusters):
            for j in range(i + 1, len(img_clusters)):
                cluster_in = img_clusters[j]
                if cluster_out.does_cluster_overlap(cluster_in, time_split_secs):
                    cluster_out.merge_cluster(cluster_in)
                    img_clusters.pop(j)
                    merged = True
                    break
            if merged:
                break


# pylint: disable=too-many-arguments,too-many-positional-arguments
def split_photos_by_time(
    input_imgs: Sequence[str],
    output_base_dir: str,
    time_split_secs: float = 60,
    date_time_tag: str = "DateTime",
    date_time_format: str = "%Y:%m:%d %H:%M:%S",
    out_date_time_format: str = "%Y%m%d_%H%M%S",
) -> None:
    """
    Split a set of images (e.g., from drone surveys) into subdirectories by time.

    Photos taken within the specified time window are clustered together
    (default: 60 seconds) and copied into a new directory named using the start
    time of each cluster. Photos are copied rather than moved.

    :param input_imgs: List of input image file paths.
    :param output_base_dir: Output base directory where cluster folders are created.
    :param time_split_secs: Maximum time gap in seconds for images to belong to
                            the same cluster (default: 60).
    :param date_time_tag: EXIF tag name containing the acquisition timestamp
                          (default: ``"DateTime"``).
    :param date_time_format: ``datetime.strptime`` format for the EXIF timestamp.
    :param out_date_time_format: ``datetime.strftime`` format for cluster folder names.
    """
    img_dt_lut: Dict[str, str] = {}
    for img_file in input_imgs:
        with Image.open(img_file) as pil_image:
            pil_exif_data = pil_image.getexif()
            for tag_id in pil_exif_data:
                tag = TAGS.get(tag_id, tag_id)
                try:
                    data = pil_exif_data.get(tag_id)
                    if isinstance(data, bytes):
                        data = data.decode()
                    if tag == date_time_tag:
                        img_dt_lut[img_file] = data
                except Exception:  # pylint: disable=broad-exception-caught
                    continue

    img_clusters: List[ImageTimeCluster] = []
    next_cluster_id = 0
    for img_file, dt_str in img_dt_lut.items():
        date_time_obj = datetime.datetime.strptime(dt_str, date_time_format)

        matched = False
        for img_cluster in img_clusters:
            if img_cluster.within_cluster(date_time_obj, time_split_secs):
                img_cluster.add_image_to_cluster(img_file, date_time_obj)
                matched = True
                break
        if not matched:
            img_cluster_obj = ImageTimeCluster(next_cluster_id, date_time_obj)
            img_cluster_obj.images.append(img_file)
            img_clusters.append(img_cluster_obj)
            next_cluster_id += 1

        _merge_overlapping_clusters(img_clusters, time_split_secs)

    for img_cluster in img_clusters:
        if img_cluster.start_time is None:
            continue
        print(f"Copying Files for: {img_cluster}")
        out_dir_name = img_cluster.start_time.strftime(out_date_time_format)
        out_dir_path = os.path.join(output_base_dir, out_dir_name)
        print(f"Output Directory: {out_dir_path}")
        os.makedirs(out_dir_path, exist_ok=True)
        for img in tqdm.tqdm(img_cluster.images):
            shutil.copy(img, out_dir_path)


if HAVE_EXIFTOOL:

    def _first_valid(
        exif: Dict[str, Any], keys: Sequence[str], default: Any = None
    ) -> Any:
        """Return the first non-None / non-empty value for a sequence of EXIF keys."""
        for key in keys:
            val = exif.get(key)
            if val not in (None, ""):
                return val
        return default

    def _init_img_info_table(include_relative_alt: bool) -> Dict[str, List[Any]]:
        """Create an empty metadata column dictionary for flightline point export."""
        img_info: Dict[str, List[Any]] = {
            "lat": [],
            "lon": [],
        }
        if include_relative_alt:
            img_info["relative_altitude"] = []
        img_info.update(
            {
                "absolute_altitude": [],
                "gimbal_roll_degree": [],
                "gimbal_pitch_degree": [],
                "gimbal_yaw_degree": [],
                "flight_pitch_degree": [],
                "flight_roll_degree": [],
                "flight_yaw_degree": [],
                "focal_length": [],
                "focal_length35mm": [],
                "image_width": [],
                "image_height": [],
                "max_aperture_value": [],
                "datetime_original": [],
                "sensor_model_data": [],
                "sensor_index": [],
                "image_filename": [],
            }
        )
        return img_info

    def _append_common_exif_fields(
        img_info: Dict[str, List[Any]],
        img_path: str,
        img_exif: Dict[str, Any],
        gimbal_default: Optional[float] = None,
    ) -> None:
        """Populate orientation, lens, and sensor EXIF fields into ``img_info``."""
        img_info["image_filename"].append(os.path.split(img_path)[1])

        img_info["gimbal_roll_degree"].append(
            float(
                _first_valid(
                    img_exif,
                    ("XMP:GimbalRollDegree", "MakerNotes:CameraRoll", "XMP:Roll"),
                    gimbal_default,
                )
            )
        )
        img_info["gimbal_pitch_degree"].append(
            float(
                _first_valid(
                    img_exif,
                    ("XMP:GimbalPitchDegree", "MakerNotes:CameraPitch", "XMP:Pitch"),
                    gimbal_default,
                )
            )
        )
        img_info["gimbal_yaw_degree"].append(
            float(
                _first_valid(
                    img_exif,
                    ("XMP:GimbalYawDegree", "MakerNotes:CameraYaw", "XMP:Yaw"),
                    gimbal_default,
                )
            )
        )

        focal_len = _first_valid(img_exif, ("EXIF:FocalLength",))
        focal_35 = _first_valid(img_exif, ("EXIF:FocalLengthIn35mmFormat",))
        img_w = _first_valid(img_exif, ("EXIF:ImageWidth", "EXIF:ExifImageWidth"))
        img_h = _first_valid(img_exif, ("EXIF:ImageHeight", "EXIF:ExifImageHeight"))

        img_info["focal_length"].append(float(focal_len))
        img_info["focal_length35mm"].append(float(focal_35))
        img_info["image_width"].append(int(img_w))
        img_info["image_height"].append(int(img_h))
        img_info["max_aperture_value"].append(img_exif.get("EXIF:MaxApertureValue"))
        img_info["datetime_original"].append(img_exif.get("EXIF:DateTimeOriginal"))
        img_info["sensor_model_data"].append(img_exif.get("EXIF:Model"))
        img_info["sensor_index"].append(
            str(_first_valid(img_exif, ("XMP:RigCameraIndex", "XMP:SensorIndex")))
        )

    def _write_vector_layer(
        gdf: geopandas.GeoDataFrame,
        out_file: str,
        out_lyr: str,
        out_format: str,
    ) -> None:
        """Write a GeoDataFrame to disk, respecting GPKG layer naming."""
        if out_format == "GPKG":
            gdf.to_file(out_file, layer=out_lyr, driver=out_format)
        else:
            gdf.to_file(out_file, driver=out_format)

    def _write_flightline_outputs(
        img_info: Dict[str, List[Any]],
        pts_vec_file: str,
        pts_vec_lyr: str,
        line_vec_file: str,
        line_vec_lyr: str,
        out_format: str,
    ) -> None:
        """Create and write both the photo point layer and flightline segment layer."""
        imgs_df = pandas.DataFrame(img_info)
        imgs_gdf = geopandas.GeoDataFrame(
            imgs_df,
            geometry=geopandas.points_from_xy(imgs_df.lon, imgs_df.lat),
            crs="EPSG:4326",
        )

        if len(imgs_gdf) == 0:
            return

        _write_vector_layer(imgs_gdf, pts_vec_file, pts_vec_lyr, out_format)

        flightline_ids: List[int] = []
        flightline_geoms: List[shapely.geometry.LineString] = []
        coords = list(zip(imgs_df.lon, imgs_df.lat))
        for seg_id, (start_pt, end_pt) in enumerate(
            zip(coords[:-1], coords[1:]), start=1
        ):
            flightline_ids.append(seg_id)
            flightline_geoms.append(
                shapely.geometry.LineString(
                    [shapely.geometry.Point(start_pt), shapely.geometry.Point(end_pt)]
                )
            )

        if flightline_geoms:
            flightline_gdf = geopandas.GeoDataFrame(
                data={"ID": flightline_ids, "geometry": flightline_geoms},
                geometry="geometry",
                crs="EPSG:4326",
            )
            _write_vector_layer(
                flightline_gdf, line_vec_file, line_vec_lyr, out_format
            )

    def _parse_exif_datetime(date_str: str) -> datetime.datetime:
        return datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_flightline_vec(
        input_imgs: List[str],
        pts_vec_file: str,
        pts_vec_lyr: str,
        line_vec_file: str,
        line_vec_lyr: str,
        out_format: str = "GPKG",
    ) -> None:
        """
        Create point and flightline vector layers from drone images containing GPS EXIF.

        If the output vector format supports multiple layers (e.g., ``"GPKG"``),
        the same file can be provided for ``pts_vec_file`` and ``line_vec_file``.

        :param input_imgs: List of input image paths.
        :param pts_vec_file: Output vector file for the photo points.
        :param pts_vec_lyr: Output vector layer name for the photo points.
        :param line_vec_file: Output vector file for the flightline.
        :param line_vec_lyr: Output vector layer name for the flightline.
        :param out_format: Output vector format driver (default: ``"GPKG"``).
        """
        sorted_imgs = rsgispy.tools.files.sort_file_by_datetime(input_imgs)
        print(f"There are {len(sorted_imgs)} input images")

        exif_entries: List[Tuple[str, Dict[str, Any]]] = []
        with exiftool.ExifToolHelper() as et:
            for img in sorted_imgs:
                exif_entries.append((img, et.get_metadata(img)[0]))
        exif_entries.sort(
            key=lambda item: _parse_exif_datetime(item[1]["EXIF:DateTimeOriginal"])
        )

        img_info = _init_img_info_table(include_relative_alt=True)
        for img, img_exif in exif_entries:
            img_info["lat"].append(
                float(_first_valid(img_exif, ("Composite:GPSLatitude", "EXIF:GPSLatitude")))
            )
            img_info["lon"].append(
                float(
                    _first_valid(img_exif, ("Composite:GPSLongitude", "EXIF:GPSLongitude"))
                )
            )
            img_info["relative_altitude"].append(
                float(
                    _first_valid(img_exif, ("XMP:RelativeAltitude", "Composite:GPSAltitude"))
                )
            )
            img_info["absolute_altitude"].append(
                float(
                    _first_valid(img_exif, ("XMP:AbsoluteAltitude", "Composite:GPSAltitude"))
                )
            )

            _append_common_exif_fields(img_info, img, img_exif)

            img_info["flight_pitch_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightPitchDegree", "MakerNotes:Pitch"),
                        999,
                    )
                )
            )
            img_info["flight_roll_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightRollDegree", "MakerNotes:Roll"),
                        999,
                    )
                )
            )
            img_info["flight_yaw_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightYawDegree", "MakerNotes:Yaw"),
                        999,
                    )
                )
            )

        _write_flightline_outputs(
            img_info,
            pts_vec_file,
            pts_vec_lyr,
            line_vec_file,
            line_vec_lyr,
            out_format,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_flightline_extern_gps_vec(
        gps_data_df: pandas.DataFrame,
        input_imgs_dir: str,
        pts_vec_file: str,
        pts_vec_lyr: str,
        line_vec_file: str,
        line_vec_lyr: str,
        out_format: str = "GPKG",
    ) -> None:
        """
        Create point and flightline vector layers from external GPS data and drone images.

        The input pandas DataFrame must have columns:
        ``image``, ``lat``, ``lon``, ``alt``, ``roll``, ``pitch``, ``yaw``.

        :param gps_data_df: DataFrame containing external GPS and attitude records.
        :param input_imgs_dir: Directory containing the input image files.
        :param pts_vec_file: Output vector file for the photo points.
        :param pts_vec_lyr: Output vector layer name for the photo points.
        :param line_vec_file: Output vector file for the flightline.
        :param line_vec_lyr: Output vector layer name for the flightline.
        :param out_format: Output vector format driver (default: ``"GPKG"``).
        """
        exif_entries: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        with exiftool.ExifToolHelper() as et:
            for _, row in gps_data_df.iterrows():
                input_img = os.path.join(input_imgs_dir, row["image"])
                if os.path.exists(input_img):
                    gps_data = {
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "alt": row["alt"],
                        "roll": row["roll"],
                        "pitch": row["pitch"],
                        "yaw": row["yaw"],
                    }
                    exif_entries.append((input_img, gps_data, et.get_metadata(input_img)[0]))

        exif_entries.sort(
            key=lambda item: _parse_exif_datetime(item[2]["EXIF:DateTimeOriginal"])
        )
        print(f"There were {len(exif_entries)} images found matching the GPS data.")

        img_info = _init_img_info_table(include_relative_alt=False)
        for img, img_gps_data, img_exif in exif_entries:
            img_info["lat"].append(float(img_gps_data["lat"]))
            img_info["lon"].append(float(img_gps_data["lon"]))
            img_info["absolute_altitude"].append(float(img_gps_data["alt"]))

            _append_common_exif_fields(img_info, img, img_exif, gimbal_default=0.0)

            img_info["flight_pitch_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightPitchDegree", "MakerNotes:Pitch"),
                        img_gps_data["pitch"],
                    )
                )
            )
            img_info["flight_roll_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightRollDegree", "MakerNotes:Roll"),
                        img_gps_data["roll"],
                    )
                )
            )
            img_info["flight_yaw_degree"].append(
                float(
                    _first_valid(
                        img_exif,
                        ("XMP:FlightYawDegree", "MakerNotes:Yaw"),
                        img_gps_data["yaw"],
                    )
                )
            )

        _write_flightline_outputs(
            img_info,
            pts_vec_file,
            pts_vec_lyr,
            line_vec_file,
            line_vec_lyr,
            out_format,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def define_extern_gps_in_imgs(
        gps_data_df: pandas.DataFrame,
        input_imgs_dir: str,
        flying_height: float,
        gimbal_roll: float,
        gimbal_pitch: float,
        gimbal_yaw: float,
        overwrite: bool = False,
    ) -> None:
        """
        Write external GPS and attitude metadata into image EXIF/XMP tags.

        The input pandas DataFrame must have columns:
        ``image``, ``lat``, ``lon``, ``alt``, ``roll``, ``pitch``, ``yaw``.

        :param gps_data_df: DataFrame containing external GPS and attitude records.
        :param input_imgs_dir: Directory containing the input image files.
        :param flying_height: Flying height (metres) written to ``XMP:RelativeAltitude``.
        :param gimbal_roll: Gimbal roll angle in degrees.
        :param gimbal_pitch: Gimbal pitch angle in degrees.
        :param gimbal_yaw: Gimbal yaw angle in degrees.
        :param overwrite: If ``True``, overwrite the original image files in place.
        """
        n_imgs = 0
        with exiftool.ExifToolHelper() as et:
            for _, row in tqdm.tqdm(gps_data_df.iterrows()):
                input_img = os.path.join(input_imgs_dir, row["image"])
                if not os.path.exists(input_img):
                    continue

                lat_val = float(row["lat"])
                lon_val = float(row["lon"])
                alt_val = float(row["alt"])
                pitch_val = float(row["pitch"])
                roll_val = float(row["roll"])
                yaw_val = float(row["yaw"])

                lat_ref, lat_ref_comp = ("S", "South") if lat_val < 0 else ("N", "North")
                lon_ref, lon_ref_comp = ("W", "West") if lon_val < 0 else ("E", "East")

                out_tags: Dict[str, Any] = {
                    "Composite:GPSLatitude": lat_val,
                    "EXIF:GPSLatitude": lat_val,
                    "EXIF:GPSLatitudeRef": lat_ref,
                    "Composite:GPSLatitudeRef": lat_ref_comp,
                    "Composite:GPSLongitude": lon_val,
                    "EXIF:GPSLongitude": lon_val,
                    "EXIF:GPSLongitudeRef": lon_ref,
                    "Composite:GPSLongitudeRef": lon_ref_comp,
                    "XMP:AbsoluteAltitude": alt_val,
                    "Composite:GPSAltitude": alt_val,
                    "XMP:RelativeAltitude": float(flying_height),
                    "XMP:FlightPitchDegree": pitch_val,
                    "MakerNotes:Pitch": pitch_val,
                    "XMP:FlightRollDegree": roll_val,
                    "MakerNotes:Roll": roll_val,
                    "XMP:FlightYawDegree": yaw_val,
                    "MakerNotes:Yaw": yaw_val,
                    "XMP:GimbalRollDegree": float(gimbal_roll),
                    "MakerNotes:CameraRoll": float(gimbal_roll),
                    "XMP:Roll": float(gimbal_roll),
                    "XMP:GimbalPitchDegree": float(gimbal_pitch),
                    "MakerNotes:CameraPitch": float(gimbal_pitch),
                    "XMP:Pitch": float(gimbal_pitch),
                    "XMP:GimbalYawDegree": float(gimbal_yaw),
                    "MakerNotes:CameraYaw": float(gimbal_yaw),
                    "XMP:Yaw": float(gimbal_yaw),
                }

                params = ["-P", "-overwrite_original"] if overwrite else []
                et.set_tags(input_img, tags=out_tags, params=params)
                n_imgs += 1
        print(f"Updated EXIF for {n_imgs} images.")
