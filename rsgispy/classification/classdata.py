# pylint: disable=too-many-lines
"""
Classification data structures, configuration I/O, sample extraction and
train/test/validation splitting (defaulting to Snappy-compressed Parquet with
support for other pandas-supported tabular formats), and unsigned class ID
recoding utilities.
"""

import math
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy.classification._helpers import resolve_rng, select_out_int_dtype
from rsgispy.imagecalc.encoding import recode_int_raster
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageutils.colourtable import (
    define_colour_table,
    set_img_thematic,
)
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.imageutils.metadata import set_img_no_data_value
from rsgispy.tools._ml_data import (
    allocate_split_counts,
    check_poly_id_col,
    extract_vector_samples_from_gdf,
    get_group_labels,
    normalise_img_band_info_list as _normalise_img_band_info_list,
    read_ml_samples,
    resolve_extract_split_group_col,
    resolve_img_band_info as _shared_resolve_img_band_info,
    resolve_split_proportions,
    split_groups_stratified,
    write_ml_samples,
)
from rsgispy.tools.colours import hex_to_rgb, rgb_to_hex
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import read_json_to_dict, write_dict_to_json
from rsgispy.vectorprocess import read_vec_gdf

__all__ = [
    "ImageBandInfo",
    "ClassInfoObj",
    "ClassificationConfig",
    "normalise_class_info",
    "resolve_img_band_info",
    "write_cls_config",
    "read_cls_config",
    "write_cls_samples",
    "read_cls_samples",
    "extract_class_training_samples",
    "split_class_samples",
    "extract_and_split_class_samples",
    "get_cls_id_luts",
    "recode_internal_to_out_ids",
    "recode_cls_img_to_out_ids",
]


def _validate_int_value(val: Any, label: str) -> int:
    """Validate that ``val`` is a non-boolean integer and return ``int(val)``."""
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise rsgispy.RSGISPyException(
            f"{label} must be an integer, got {val!r}."
        )
    return int(val)


def _validate_uint_value(val: Any, label: str) -> int:
    """Validate that ``val`` is an unsigned (non-negative) integer."""
    int_val = _validate_int_value(val, label)
    if int_val < 0:
        raise rsgispy.RSGISPyException(
            f"{label} must be an unsigned (non-negative) integer, "
            f"got {int_val}."
        )
    return int_val


def _validate_rgb_component(val: Any, label: str) -> int:
    """Validate an RGB colour component integer in ``[0, 255]``."""
    int_val = _validate_int_value(val, label)
    if int_val < 0 or int_val > 255:
        raise rsgispy.RSGISPyException(
            f"{label} must be between 0 and 255, got {int_val}."
        )
    return int_val


def _parse_colour_value(
    colour: Union[str, Sequence[int]],
) -> Tuple[int, int, int]:
    """Parse a hex colour string (``'#RRGGBB'`` / ``'RRGGBB'``) or RGB sequence."""
    if isinstance(colour, str):
        return hex_to_rgb(colour)
    if isinstance(colour, (tuple, list)) and len(colour) == 3:
        return (
            _validate_rgb_component(colour[0], "red"),
            _validate_rgb_component(colour[1], "green"),
            _validate_rgb_component(colour[2], "blue"),
        )
    raise rsgispy.RSGISPyException(
        "colour must be a hex string (e.g. '#ff8800') or "
        f"3-element (R, G, B) sequence, got {colour!r}."
    )


class ClassInfoObj:
    """
    Definition of a single classification class and its training vector samples.

    Supports any user-specified unsigned integer :attr:`out_id` (e.g., ``0``,
    ``1``, ``10``, ``255``, ``1000``) alongside a consecutive 0-based
    :attr:`internal_id` (``0, 1, ..., N - 1``) required by classifiers such as
    XGBoost.

    Class display colours may be specified either via ``red``, ``green``,
    ``blue`` integer components (``0..255``) or via ``colour`` as a hexadecimal
    string (``'#RRGGBB'`` or ``'RRGGBB'``) or ``(R, G, B)`` tuple.

    :param out_id: User-defined unsigned integer output class ID (``>= 0``).
    :param vec_file: Path to the training vector file (points or polygons).
    :param vec_lyr: Optional layer name within ``vec_file``.
    :param red: Red colour value in ``0..255`` (Default = ``0`` when ``colour`` is omitted).
    :param green: Green colour value in ``0..255`` (Default = ``0`` when ``colour`` is omitted).
    :param blue: Blue colour value in ``0..255`` (Default = ``0`` when ``colour`` is omitted).
    :param internal_id: Optional consecutive 0-based internal class ID. When
                        ``None`` (default), consecutive IDs ``0..N-1`` are
                        assigned automatically during normalisation.
    :param name: Optional logical class name.
    :param colour: Optional hex colour string (e.g. ``'#228b22'``) or ``(R, G, B)`` tuple.
                   Cannot conflict with explicitly supplied ``red``, ``green``, ``blue`` values.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        out_id: int,
        vec_file: Optional[Union[str, os.PathLike]] = None,
        vec_lyr: Optional[str] = None,
        red: Optional[int] = None,
        green: Optional[int] = None,
        blue: Optional[int] = None,
        internal_id: Optional[int] = None,
        name: Optional[str] = None,
        colour: Optional[Union[str, Sequence[int]]] = None,
    ):
        self.out_id: int = _validate_uint_value(out_id, "out_id")
        self.vec_file: Optional[str] = (
            os.fspath(vec_file) if vec_file is not None else None
        )
        self.vec_lyr: Optional[str] = vec_lyr
        if colour is not None:
            c_red, c_green, c_blue = _parse_colour_value(colour)
            if red is not None and _validate_rgb_component(red, "red") != c_red:
                raise rsgispy.RSGISPyException(
                    f"Specified red={red} conflicts with colour={colour!r}."
                )
            if (
                green is not None
                and _validate_rgb_component(green, "green") != c_green
            ):
                raise rsgispy.RSGISPyException(
                    f"Specified green={green} conflicts with colour={colour!r}."
                )
            if blue is not None and _validate_rgb_component(blue, "blue") != c_blue:
                raise rsgispy.RSGISPyException(
                    f"Specified blue={blue} conflicts with colour={colour!r}."
                )
            self.red: int = c_red
            self.green: int = c_green
            self.blue: int = c_blue
        else:
            self.red = _validate_rgb_component(0 if red is None else red, "red")
            self.green = _validate_rgb_component(
                0 if green is None else green, "green"
            )
            self.blue = _validate_rgb_component(0 if blue is None else blue, "blue")
        if internal_id is not None:
            self.internal_id: Optional[int] = _validate_uint_value(
                internal_id, "internal_id"
            )
        else:
            self.internal_id = None
        self.name: Optional[str] = name

    @property
    def colour(self) -> str:
        """Hexadecimal colour string (``'#rrggbb'``) synchronised with RGB values."""
        return rgb_to_hex(self.red, self.green, self.blue)

    @colour.setter
    def colour(self, value: Union[str, Sequence[int]]) -> None:
        r_val, g_val, b_val = _parse_colour_value(value)
        self.red = r_val
        self.green = g_val
        self.blue = b_val

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "name": self.name,
            "out_id": self.out_id,
            "internal_id": self.internal_id,
            "vec_file": self.vec_file,
            "vec_lyr": self.vec_lyr,
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "colour": self.colour,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassInfoObj":
        """Construct a :class:`ClassInfoObj` from a dictionary."""
        if not isinstance(data, Mapping) or "out_id" not in data:
            raise rsgispy.RSGISPyException(
                "ClassInfoObj data must be a mapping containing 'out_id'."
            )
        return cls(
            out_id=data["out_id"],
            vec_file=data.get("vec_file"),
            vec_lyr=data.get("vec_lyr"),
            red=data.get("red"),
            green=data.get("green"),
            blue=data.get("blue"),
            internal_id=data.get("internal_id"),
            name=data.get("name"),
            colour=data.get("colour"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassInfoObj):
            return NotImplemented
        return (
            self.out_id == other.out_id
            and self.internal_id == other.internal_id
            and self.vec_file == other.vec_file
            and self.vec_lyr == other.vec_lyr
            and self.red == other.red
            and self.green == other.green
            and self.blue == other.blue
            and self.name == other.name
        )

    def __repr__(self) -> str:
        return (
            f"ClassInfoObj(out_id={self.out_id!r}, "
            f"internal_id={self.internal_id!r}, "
            f"vec_file={self.vec_file!r}, vec_lyr={self.vec_lyr!r}, "
            f"red={self.red!r}, green={self.green!r}, blue={self.blue!r}, "
            f"colour={self.colour!r}, name={self.name!r})"
        )


def normalise_class_info(
    class_info: Mapping[str, Union[ClassInfoObj, Mapping[str, Any]]],
) -> Dict[str, ClassInfoObj]:
    """
    Validate and normalise a mapping of class names to :class:`ClassInfoObj`
    instances, ensuring unique output class IDs and consecutive 0-based
    ``internal_id`` values (``0, 1, ..., N - 1``).

    :param class_info: Mapping of ``{class_name: ClassInfoObj}``.
    :return: Normalised dictionary of :class:`ClassInfoObj` copies with
             populated ``name`` and consecutive 0-based ``internal_id`` values.

    .. code:: python

        import rsgispy.classification.classdata as cd

        classes = cd.normalise_class_info({
            "Forest": cd.ClassInfoObj(out_id=10, vec_file="forest.gpkg"),
            "Water": cd.ClassInfoObj(out_id=25, vec_file="water.gpkg"),
        })
    """
    # pylint: disable=too-many-branches
    if not isinstance(class_info, Mapping) or not class_info:
        raise rsgispy.RSGISPyException(
            "class_info must be a non-empty mapping of class names to "
            "ClassInfoObj instances."
        )

    entries: List[Tuple[str, ClassInfoObj]] = []
    seen_out_ids: Dict[int, str] = {}

    for raw_name, raw_obj in class_info.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise rsgispy.RSGISPyException(
                "Class names in class_info must be non-empty strings."
            )
        cls_name = raw_name.strip()
        if isinstance(raw_obj, ClassInfoObj):
            obj = ClassInfoObj(
                out_id=raw_obj.out_id,
                vec_file=raw_obj.vec_file,
                vec_lyr=raw_obj.vec_lyr,
                red=raw_obj.red,
                green=raw_obj.green,
                blue=raw_obj.blue,
                internal_id=raw_obj.internal_id,
                name=raw_obj.name if raw_obj.name is not None else cls_name,
            )
        elif isinstance(raw_obj, Mapping):
            obj = ClassInfoObj.from_dict(raw_obj)
            if obj.name is None:
                obj.name = cls_name
        else:
            raise rsgispy.RSGISPyException(
                f"Class '{cls_name}' must be a ClassInfoObj instance."
            )

        if obj.out_id in seen_out_ids:
            raise rsgispy.RSGISPyException(
                f"Duplicate out_id {obj.out_id} found for classes "
                f"'{seen_out_ids[obj.out_id]}' and '{cls_name}'."
            )
        seen_out_ids[obj.out_id] = cls_name
        entries.append((cls_name, obj))

    internal_ids = [obj.internal_id for _, obj in entries]
    if all(iid is None for iid in internal_ids):
        for idx, (_, obj) in enumerate(entries):
            obj.internal_id = idx
    elif any(iid is None for iid in internal_ids):
        raise rsgispy.RSGISPyException(
            "Either all ClassInfoObj entries must specify internal_id or none."
        )
    else:
        int_list = [int(iid) for iid in internal_ids if iid is not None]
        if sorted(int_list) != list(range(len(entries))):
            raise rsgispy.RSGISPyException(
                "Explicit internal_id values must start from 0 and be "
                f"consecutive (expected 0..{len(entries) - 1}), got {int_list}."
            )

    return dict(entries)


def resolve_img_band_info(
    img_band_info: Sequence[ImageBandInfo],
) -> List[Tuple[str, List[int], List[str]]]:
    """
    Validate input raster paths and resolve 1-based band lists and unique
    output feature column names for each :class:`ImageBandInfo`.

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects.
    :return: List of ``(file_name, bands_list, band_col_names)`` tuples.

    .. code:: python

        import rsgispy.classification.classdata as cd

        resolved = cd.resolve_img_band_info([
            cd.ImageBandInfo("s2.tif", name="s2", bands=[1, 2]),
        ])
    """
    return _shared_resolve_img_band_info(img_band_info)


def _coerce_seg_info(seg_info: Optional[Any]) -> Optional[Any]:
    """Coerce a ``seg_info`` mapping into a ``SegmentFeatureConfig`` (or ``None``)."""
    if seg_info is None:
        return None
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.segments import coerce_seg_info

    return coerce_seg_info(seg_info)


class ClassificationConfig:
    """
    Container storing image/band definitions (:class:`ImageBandInfo`) and
    per-class training definitions (:class:`ClassInfoObj`), with class-ID
    lookup helpers and JSON serialisation.

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects defining
                          input raster images and bands.
    :param class_info: Mapping of ``{class_name: ClassInfoObj}`` defining
                       training classes, output IDs, and internal IDs.
    :param out_id_col: Column name for user output class IDs in sample tables
                       (Default = ``'out_cls_id'``).
    :param internal_id_col: Column name for consecutive 0-based internal class
                            IDs in sample tables
                            (Default = ``'internal_cls_id'``).
    :param class_name_col: Column name for class names in sample tables
                           (Default = ``'class_name'``).
    :param seg_info: Optional segmentation settings for object-based
                     (segment) classification
                     (:class:`~rsgispy.ml.segments.SegmentFeatureConfig` or
                     its dictionary form). Omitted from the JSON when ``None``.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        img_band_info: Sequence[Union[ImageBandInfo, Mapping[str, Any]]],
        class_info: Mapping[str, Union[ClassInfoObj, Mapping[str, Any]]],
        out_id_col: str = "out_cls_id",
        internal_id_col: str = "internal_cls_id",
        class_name_col: str = "class_name",
        seg_info: Optional[Any] = None,
    ):
        self.img_band_info: List[ImageBandInfo] = _normalise_img_band_info_list(
            img_band_info
        )
        self.class_info: Dict[str, ClassInfoObj] = normalise_class_info(
            class_info
        )
        if not out_id_col or not internal_id_col or not class_name_col:
            raise rsgispy.RSGISPyException(
                "out_id_col, internal_id_col, and class_name_col must be "
                "non-empty strings."
            )
        if len({out_id_col, internal_id_col, class_name_col}) != 3:
            raise rsgispy.RSGISPyException(
                "out_id_col, internal_id_col, and class_name_col must be "
                "distinct."
            )
        self.out_id_col: str = str(out_id_col)
        self.internal_id_col: str = str(internal_id_col)
        self.class_name_col: str = str(class_name_col)
        self.seg_info: Optional[Any] = _coerce_seg_info(seg_info)

    def get_internal_to_out_lut(self) -> Dict[int, int]:
        """Return mapping ``{internal_id: out_id}`` across all classes."""
        return {
            int(obj.internal_id): int(obj.out_id)  # type: ignore[arg-type]
            for obj in self.class_info.values()
        }

    def get_out_to_internal_lut(self) -> Dict[int, int]:
        """Return mapping ``{out_id: internal_id}`` across all classes."""
        return {
            int(obj.out_id): int(obj.internal_id)  # type: ignore[arg-type]
            for obj in self.class_info.values()
        }

    def get_class_names_lut(
        self, use_internal_id: bool = False
    ) -> Dict[int, str]:
        """
        Return ``{class_id: class_name}`` keyed by ``out_id`` (default) or
        ``internal_id``.
        """
        if use_internal_id:
            return {
                int(obj.internal_id): cls_name  # type: ignore[arg-type]
                for cls_name, obj in self.class_info.items()
            }
        return {
            int(obj.out_id): cls_name
            for cls_name, obj in self.class_info.items()
        }

    def get_colour_table(
        self, use_internal_id: bool = False
    ) -> Dict[int, List[int]]:
        """
        Return ``{class_id: [red, green, blue, 255]}`` suitable for
        :func:`rsgispy.imageutils.colourtable.define_colour_table`.
        """
        out_lut: Dict[int, List[int]] = {}
        for obj in self.class_info.values():
            key = (
                int(obj.internal_id)  # type: ignore[arg-type]
                if use_internal_id
                else int(obj.out_id)
            )
            out_lut[key] = [obj.red, obj.green, obj.blue, 255]
        return out_lut

    def to_feature_stack(self, scaler: Optional[Any] = None) -> Any:
        """
        Convert this configuration's ``img_band_info`` into a unified
        :class:`~rsgispy.ml.stack.ImageFeatureStack`.

        :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler`.
        :return: :class:`~rsgispy.ml.stack.ImageFeatureStack`.
        """
        # pylint: disable=import-outside-toplevel
        from rsgispy.ml.stack import ImageFeatureStack

        return ImageFeatureStack.from_image_band_infos(
            self.img_band_info, scaler=scaler
        )

    def get_band_names(self) -> List[str]:
        """
        Resolve and return the ordered list of feature column names across all
        configured :class:`ImageBandInfo` items.
        """
        resolved = resolve_img_band_info(self.img_band_info)
        names: List[str] = []
        for _, _, col_names in resolved:
            names.extend(col_names)
        return names

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        out: Dict[str, Any] = {
            "version": 1,
            "out_id_col": self.out_id_col,
            "internal_id_col": self.internal_id_col,
            "class_name_col": self.class_name_col,
            "img_band_info": [info.to_dict() for info in self.img_band_info],
            "class_info": {
                cls_name: obj.to_dict()
                for cls_name, obj in self.class_info.items()
            },
        }
        if self.seg_info is not None:
            out["seg_info"] = self.seg_info.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationConfig":
        """Construct a :class:`ClassificationConfig` from a dictionary."""
        if not isinstance(data, Mapping):
            raise rsgispy.RSGISPyException(
                "Classification configuration data must be a dictionary."
            )
        if "img_band_info" not in data or "class_info" not in data:
            raise rsgispy.RSGISPyException(
                "Classification configuration must contain 'img_band_info' "
                "and 'class_info'."
            )
        return cls(
            img_band_info=data["img_band_info"],
            class_info=data["class_info"],
            out_id_col=data.get("out_id_col", "out_cls_id"),
            internal_id_col=data.get("internal_id_col", "internal_cls_id"),
            class_name_col=data.get("class_name_col", "class_name"),
            seg_info=data.get("seg_info"),
        )

    def save(self, out_json_file: Union[str, os.PathLike]) -> None:
        """Save this configuration to ``out_json_file``."""
        write_cls_config(out_json_file, self)

    @classmethod
    def load(
        cls, in_json_file: Union[str, os.PathLike]
    ) -> "ClassificationConfig":
        """Load a :class:`ClassificationConfig` from ``in_json_file``."""
        return read_cls_config(in_json_file)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassificationConfig):
            return NotImplemented
        return (
            self.img_band_info == other.img_band_info
            and self.class_info == other.class_info
            and self.out_id_col == other.out_id_col
            and self.internal_id_col == other.internal_id_col
            and self.class_name_col == other.class_name_col
            and self.seg_info == other.seg_info
        )

    def __repr__(self) -> str:
        return (
            f"ClassificationConfig(img_band_info={self.img_band_info!r}, "
            f"class_info={self.class_info!r}, "
            f"out_id_col={self.out_id_col!r}, "
            f"internal_id_col={self.internal_id_col!r}, "
            f"class_name_col={self.class_name_col!r}, "
            f"seg_info={self.seg_info!r})"
        )


def _coerce_config(
    img_band_info: Union[ClassificationConfig, Sequence[ImageBandInfo]],
    class_info: Optional[Mapping[str, ClassInfoObj]],
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    class_name_col: Optional[str] = None,
) -> ClassificationConfig:
    """Coerce either a ClassificationConfig or separate structures into one."""
    if isinstance(img_band_info, ClassificationConfig):
        if class_info is not None:
            raise rsgispy.RSGISPyException(
                "Do not pass class_info separately when img_band_info is a "
                "ClassificationConfig instance."
            )
        return ClassificationConfig(
            img_band_info=img_band_info.img_band_info,
            class_info=img_band_info.class_info,
            out_id_col=(
                out_id_col
                if out_id_col is not None
                else img_band_info.out_id_col
            ),
            internal_id_col=(
                internal_id_col
                if internal_id_col is not None
                else img_band_info.internal_id_col
            ),
            class_name_col=(
                class_name_col
                if class_name_col is not None
                else img_band_info.class_name_col
            ),
            seg_info=img_band_info.seg_info,
        )
    if class_info is None:
        raise rsgispy.RSGISPyException(
            "class_info must be provided when img_band_info is not a "
            "ClassificationConfig instance."
        )
    return ClassificationConfig(
        img_band_info=img_band_info,
        class_info=class_info,
        out_id_col="out_cls_id" if out_id_col is None else out_id_col,
        internal_id_col=(
            "internal_cls_id" if internal_id_col is None else internal_id_col
        ),
        class_name_col=(
            "class_name" if class_name_col is None else class_name_col
        ),
    )


def write_cls_config(
    out_json_file: Union[str, os.PathLike],
    img_band_info: Union[ClassificationConfig, Sequence[ImageBandInfo]],
    class_info: Optional[Mapping[str, ClassInfoObj]] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    class_name_col: Optional[str] = None,
) -> ClassificationConfig:
    """
    Save the classification image/band and class training configuration to a
    JSON file on disk.

    :param out_json_file: Path to the output JSON configuration file.
    :param img_band_info: Either a :class:`ClassificationConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param class_info: Mapping of ``{class_name: ClassInfoObj}`` (required when
                       ``img_band_info`` is a sequence of :class:`ImageBandInfo`).
    :param out_id_col: Optional override for the output class ID column name.
    :param internal_id_col: Optional override for the internal class ID column
                            name.
    :param class_name_col: Optional override for the class name column name.
    :return: The validated :class:`ClassificationConfig` written to disk.

    .. code:: python

        import rsgispy.classification.classdata as cd

        img_info = [cd.ImageBandInfo("s2.tif", name="s2", bands=[1, 2, 3])]
        cls_info = {
            "Forest": cd.ClassInfoObj(out_id=10, vec_file="forest.gpkg", green=200),
            "Water": cd.ClassInfoObj(out_id=25, vec_file="water.gpkg", blue=255),
        }
        cd.write_cls_config("cls_config.json", img_info, cls_info)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg = _coerce_config(
        img_band_info,
        class_info,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        class_name_col=class_name_col,
    )
    out_path = ensure_parent_dir(out_json_file)
    write_dict_to_json(cfg.to_dict(), out_path)
    return cfg


def read_cls_config(
    in_json_file: Union[str, os.PathLike],
) -> ClassificationConfig:
    """
    Read a classification configuration JSON file from disk and return a
    validated :class:`ClassificationConfig` object.

    :param in_json_file: Path to the input JSON configuration file.
    :return: Loaded :class:`ClassificationConfig` instance.

    .. code:: python

        import rsgispy.classification.classdata as cd

        cfg = cd.read_cls_config("cls_config.json")
    """
    in_path = os.fspath(in_json_file)
    ensure_file_exists(in_path, label="Classification config")
    raw_dict = read_json_to_dict(in_path)
    return ClassificationConfig.from_dict(raw_dict)


def write_cls_samples(
    samples_df: pd.DataFrame,
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
) -> None:
    """
    Write a classification sample :class:`pandas.DataFrame` to disk in Parquet
    (using Snappy compression by default) or any other pandas-supported format
    handled by :func:`rsgispy.tools.tables.write_dataframe_table` (e.g., CSV,
    TSV, Feather, Excel, JSON, Pickle, HDF5). Every column is written,
    including a group column (``poly_id_col`` of
    :func:`extract_class_training_samples`, e.g. ``'train_grp'``), so the
    grouped split and group cross-validation also work on the files.

    :param samples_df: Sample :class:`pandas.DataFrame` to save.
    :param out_file: Output table file path (e.g., ``'train.parquet'``,
                     ``'train.csv'``, ``'train.feather'``).
    :param out_format: Optional output table format overriding extension
                       detection (e.g., ``'parquet'``, ``'csv'``, ``'tsv'``,
                       ``'feather'``, ``'excel'``, ``'json'``, ``'pickle'``,
                       ``'hdf'``).
    :param out_compress: Compression setting passed to
                         :func:`rsgispy.tools.tables.write_dataframe_table`
                         (Default = ``'snappy'`` for Parquet; ignored for
                         non-Parquet formats when left at ``'snappy'``).

    .. code:: python

        import pandas as pd
        import rsgispy.classification.classdata as cd

        df = pd.DataFrame({"out_cls_id": [10], "internal_cls_id": [0], "b1": [0.2]})
        cd.write_cls_samples(df, "train.parquet")
        cd.write_cls_samples(df, "train.csv")
    """
    write_ml_samples(
        samples_df=samples_df,
        out_file=out_file,
        out_format=out_format,
        out_compress=out_compress,
    )


def read_cls_samples(
    in_file: Union[str, os.PathLike],
    in_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a classification sample table file (Parquet or any other format
    supported by :func:`rsgispy.tools.tables.read_dataframe_table`) into a
    :class:`pandas.DataFrame`.

    :param in_file: Path to the input sample file.
    :param in_format: Optional input format overriding extension detection.
    :return: Loaded :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.classification.classdata as cd

        train_df = cd.read_cls_samples("train.parquet")
    """
    return read_ml_samples(in_file, in_format=in_format)


def extract_class_training_samples(
    img_band_info: Union[ClassificationConfig, Sequence[ImageBandInfo]],
    class_info: Optional[Mapping[str, ClassInfoObj]] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    out_id_col: Optional[str] = None,
    internal_id_col: Optional[str] = None,
    class_name_col: Optional[str] = None,
    use_pbar: bool = True,
    *,
    poly_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract pixel training samples for every class defined in ``class_info`` across
    all bands defined in ``img_band_info``, returning a combined
    :class:`pandas.DataFrame` and optionally saving it to Parquet (Snappy
    compressed by default) or another pandas-supported table format.

    Point training layers are extracted using
    :func:`rsgispy.zonalstats.point.ext_point_band_values_file` (in memory,
    without modifying the source vector file) and polygon training layers are
    extracted using
    :func:`rsgispy.zonalstats.polygon.export_poly_pxls_to_df`. The images may
    have bands of different data types (e.g. one VRT stacking ``uint16``
    reflectance and ``float32`` indices).

    With ``poly_id_col``, the source of each pixel is recorded in that column,
    as in the segment workflow's ``train_grp`` column:
    ``'<class name>:poly<n>'`` for the ``n``-th (0-based) polygon of the
    class's training layer and ``'<class name>:pt<n>'`` for points. The pixels
    of one polygon are strongly correlated, so pass the column as
    ``group_col`` to :func:`split_class_samples` (so that a polygon is never
    in two sets) and as ``groups`` to
    :func:`~rsgispy.classification.classsklearn.perform_sklearn_classifier_param_search`
    with :class:`~rsgispy.ml.cv.PolygonGroupKFold`. It is a string column, so
    it is never used as a feature, and is kept by :func:`write_cls_samples`
    and :func:`read_cls_samples` in every table format.

    :param img_band_info: Either a :class:`ClassificationConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param class_info: Mapping of ``{class_name: ClassInfoObj}`` (required when
                       ``img_band_info`` is a sequence of :class:`ImageBandInfo`).
    :param out_file: Optional output table file path (e.g., ``'samples.parquet'``
                     or ``'samples.csv'``).
    :param out_format: Optional output table format overriding extension
                       detection (Default = ``None``).
    :param out_compress: Compression setting passed to :func:`write_cls_samples`
                         (Default = ``'snappy'`` for Parquet).
    :param min_thres: Optional inclusive lower valid pixel threshold.
    :param max_thres: Optional inclusive upper valid pixel threshold.
    :param out_no_data_val: Optional fill/nodata sentinel value excluded when
                            ``mask_nodata=True``.
    :param mask_nodata: If ``True`` (default), drop rows containing nodata or
                        non-finite band values.
    :param all_touched: For polygon vectors, passed to
                        :func:`rsgispy.zonalstats.polygon.export_poly_pxls_to_df`
                        (Default = ``False``).
    :param include_coords: If ``True``, include ``x`` and ``y`` sample
                           coordinates (Default = ``False``).
    :param reproj_vec: Reproject vector geometries on the fly if CRSs differ
                       (Default = ``False``).
    :param vec_def_epsg: Optional EPSG code overriding undefined vector CRS.
    :param out_id_col: Optional column name for user output class IDs.
    :param internal_id_col: Optional column name for 0-based internal class IDs.
    :param class_name_col: Optional column name for class names.
    :param use_pbar: If ``True``, display progress bars during extraction
                     (Default = ``True``).
    :param poly_id_col: Optional name of a column recording the source polygon
                        (or point) of each sample (e.g. ``'train_grp'``; it
                        follows the class ID columns). When ``None`` (default)
                        no column is added.
    :return: Combined :class:`pandas.DataFrame` containing class ID/name
             columns, the optional ``poly_id_col`` and coordinate columns and
             the extracted band feature columns.

    .. code:: python

        import rsgispy.classification.classdata as cd

        cfg = cd.read_cls_config("cls_config.json")
        samples_df = cd.extract_class_training_samples(
            cfg, out_file="all_samples.parquet", poly_id_col="train_grp"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg = _coerce_config(
        img_band_info,
        class_info,
        out_id_col=out_id_col,
        internal_id_col=internal_id_col,
        class_name_col=class_name_col,
    )
    resolved_imgs = resolve_img_band_info(cfg.img_band_info)
    all_feature_cols = [col for _, _, cols in resolved_imgs for col in cols]
    meta_cols = [cfg.class_name_col, cfg.out_id_col, cfg.internal_id_col]
    check_poly_id_col(poly_id_col, meta_cols + ["x", "y"] + all_feature_cols)

    chunks: List[pd.DataFrame] = []
    for cls_name, cls_obj in cfg.class_info.items():
        if not cls_obj.vec_file:
            raise rsgispy.RSGISPyException(
                f"Class '{cls_name}' must specify 'vec_file' to extract "
                "training samples."
            )
        ensure_file_exists(
            cls_obj.vec_file, label=f"Class '{cls_name}' training vector"
        )
        vec_gdf = read_vec_gdf(
            cls_obj.vec_file, vec_lyr=cls_obj.vec_lyr, copy=True
        )
        cls_df = extract_vector_samples_from_gdf(
            vec_gdf=vec_gdf,
            resolved_imgs=resolved_imgs,
            min_thres=min_thres,
            max_thres=max_thres,
            out_no_data_val=out_no_data_val,
            mask_nodata=mask_nodata,
            all_touched=all_touched,
            include_coords=include_coords,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
            label=f"class '{cls_name}'",
            group_col=poly_id_col,
            group_prefix=f"{cls_name}:",
        )

        n_rows = len(cls_df)
        if n_rows > 0:
            int_cls_id = int(cls_obj.internal_id)  # type: ignore[arg-type]
            cls_df.insert(0, cfg.internal_id_col, int_cls_id)
            cls_df.insert(0, cfg.out_id_col, int(cls_obj.out_id))
            cls_df.insert(0, cfg.class_name_col, cls_name)
            chunks.append(cls_df)

    grp_cols = [poly_id_col] if poly_id_col is not None else []
    coord_cols = ["x", "y"] if include_coords else []
    ordered_cols = meta_cols + grp_cols + coord_cols + all_feature_cols

    if chunks:
        out_df = pd.concat(chunks, ignore_index=True)[ordered_cols]
    else:
        empty_data: Dict[str, Any] = {
            cfg.class_name_col: pd.Series(dtype="object"),
            cfg.out_id_col: pd.Series(dtype="int64"),
            cfg.internal_id_col: pd.Series(dtype="int64"),
        }
        for g_col in grp_cols:
            empty_data[g_col] = pd.Series(dtype="object")
        for c_col in coord_cols:
            empty_data[c_col] = pd.Series(dtype="float64")
        for f_col in all_feature_cols:
            empty_data[f_col] = pd.Series(dtype="float64")
        out_df = pd.DataFrame(empty_data)

    if out_file is not None:
        write_cls_samples(
            out_df,
            out_file,
            out_format=out_format,
            out_compress=out_compress,
        )

    return out_df


def _resolve_count_for_class(
    spec: Optional[Union[int, Mapping[Any, int]]],
    cls_key: Any,
    cls_name: Optional[str],
    label: str,
) -> Optional[int]:
    """Resolve a scalar or per-class sample count limit."""
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        if cls_key in spec:
            val = spec[cls_key]
        elif cls_name is not None and cls_name in spec:
            val = spec[cls_name]
        else:
            raise rsgispy.RSGISPyException(
                f"Missing class '{cls_key}' in {label} count mapping."
            )
    else:
        val = spec
    int_val = _validate_int_value(val, label)
    if int_val < 0:
        raise rsgispy.RSGISPyException(f"{label} must be >= 0, got {int_val}.")
    return int_val


def split_class_samples(
    samples_df: pd.DataFrame,
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    stratify_col: str = "internal_cls_id",
    class_name_col: str = "class_name",
    max_cls_smpls: Optional[Union[int, Mapping[Any, int]]] = None,
    n_train: Optional[Union[int, Mapping[Any, int]]] = None,
    n_test: Optional[Union[int, Mapping[Any, int]]] = None,
    n_valid: Optional[Union[int, Mapping[Any, int]]] = None,
    rnd_seed: Optional[int] = None,
    *,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a classification sample :class:`pandas.DataFrame` stratified per class
    into training, testing, and validation DataFrames, optionally saving each
    set in Parquet (Snappy compressed by default) or another pandas-supported
    tabular format.

    By default the samples (pixels) of each class are split at random, so the
    pixels of one training polygon usually end up in all three sets and the
    test accuracy is optimistic. With ``group_col`` (e.g. the ``poly_id_col``
    column from :func:`extract_class_training_samples`), whole groups are
    assigned to one set, as in
    :func:`~rsgispy.classification.classsegments.split_class_segment_samples`:
    for each class the groups are taken in random order and each is put in
    the set furthest below its target number of samples, after making sure
    that every set gets at least one group of the class if there are enough
    groups. The proportions are therefore met as closely as the group sizes
    allow, and a warning is issued for a class with fewer groups than sets
    (some sets then have no samples of that class). ``max_cls_smpls``
    subsamples pixels at random before the split, and ``n_train`` /
    ``n_test`` / ``n_valid`` become upper limits (sets above them are randomly
    subsampled). A group whose samples have several classes is assigned with
    the class of most of its samples.

    :param samples_df: Input sample :class:`pandas.DataFrame` containing a class
                       identifier column (``stratify_col``).
    :param train_prop: Proportion of samples per class for training. When
                       ``None`` (default), computed as
                       ``1.0 - test_prop - valid_prop`` (``0.6`` by default).
    :param test_prop: Proportion of samples per class for testing
                      (Default = ``0.2``).
    :param valid_prop: Proportion of samples per class for validation
                       (Default = ``0.2``).
    :param out_train_file: Optional output file path for training samples.
    :param out_test_file: Optional output file path for testing samples.
    :param out_valid_file: Optional output file path for validation samples.
    :param out_format: Optional output table format overriding extension
                       detection (Default = ``None``).
    :param out_compress: Compression setting passed to :func:`write_cls_samples`
                         (Default = ``'snappy'`` for Parquet).
    :param stratify_col: Column name identifying each class for stratified
                         splitting (Default = ``'internal_cls_id'``).
    :param class_name_col: Optional class name column used when per-class
                           count dictionaries are keyed by class name.
    :param max_cls_smpls: Optional per-class cap (scalar or mapping) applied by
                          random subsampling before splitting.
    :param n_train: Optional explicit training sample count (scalar or mapping).
    :param n_test: Optional explicit testing sample count (scalar or mapping).
    :param n_valid: Optional explicit validation sample count (scalar or
                    mapping).
    :param rnd_seed: Optional random seed for reproducible splits.
    :param group_col: Optional column of group labels (e.g. the source
                      polygon of each pixel, ``'train_grp'``); all samples of
                      a group are put in the same set. When ``None``
                      (default) the samples are split at random.
    :return: Tuple ``(train_df, test_df, valid_df)`` of split DataFrames.

    .. code:: python

        import rsgispy.classification.classdata as cd

        train_df, test_df, valid_df = cd.split_class_samples(
            samples_df,
            train_prop=0.6,
            test_prop=0.2,
            valid_prop=0.2,
            out_train_file="train.parquet",
            out_test_file="test.parquet",
            out_valid_file="valid.parquet",
            rnd_seed=42,
            group_col="train_grp",  # whole training polygons per set
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if not isinstance(samples_df, pd.DataFrame):
        raise rsgispy.RSGISPyException(
            "samples_df must be a pandas DataFrame."
        )
    if stratify_col not in samples_df.columns:
        raise rsgispy.RSGISPyException(
            f"Stratification column '{stratify_col}' was not found in "
            f"samples_df columns {list(samples_df.columns)}."
        )
    grp_labels = (
        get_group_labels(samples_df, group_col) if group_col is not None else None
    )

    train_p, test_p, valid_p = resolve_split_proportions(
        train_prop, test_prop, valid_prop
    )
    rng = resolve_rng(rnd_seed)

    train_idx: List[int] = []
    test_idx: List[int] = []
    valid_idx: List[int] = []
    capped_pos: List[np.ndarray] = []

    has_cls_name = class_name_col in samples_df.columns

    def _cls_name(cls_pos: np.ndarray) -> Optional[str]:
        if has_cls_name and len(cls_pos) > 0:
            return str(samples_df[class_name_col].iloc[int(cls_pos[0])])
        return None

    def _cls_counts(
        cls_key: Any, n_rows: int, cls_name: Optional[str]
    ) -> Tuple[int, int, int, bool]:
        c_tr = _resolve_count_for_class(n_train, cls_key, cls_name, "n_train")
        c_te = _resolve_count_for_class(n_test, cls_key, cls_name, "n_test")
        c_va = _resolve_count_for_class(n_valid, cls_key, cls_name, "n_valid")
        n_tr_c, n_te_c, n_va_c = allocate_split_counts(
            n_rows,
            train_p,
            test_p,
            valid_p,
            c_tr,
            c_te,
            c_va,
            group_label="class samples",
        )
        return n_tr_c, n_te_c, n_va_c, any(
            c is not None for c in (c_tr, c_te, c_va)
        )

    grouped_pos = samples_df.groupby(stratify_col, sort=False).indices
    for cls_key, raw_pos in grouped_pos.items():
        cls_pos = np.asarray(raw_pos, dtype=np.intp).copy()
        rng.shuffle(cls_pos)
        max_cap = _resolve_count_for_class(
            max_cls_smpls, cls_key, _cls_name(cls_pos), "max_cls_smpls"
        )
        if max_cap is not None and len(cls_pos) > max_cap:
            cls_pos = cls_pos[:max_cap]
        if grp_labels is not None:
            capped_pos.append(cls_pos)
            continue

        n_tr_c, n_te_c, n_va_c, _ = _cls_counts(
            cls_key, len(cls_pos), _cls_name(cls_pos)
        )
        test_end = n_tr_c + n_te_c
        valid_end = test_end + n_va_c
        train_idx.extend(cls_pos[:n_tr_c].tolist())
        test_idx.extend(cls_pos[n_tr_c:test_end].tolist())
        valid_idx.extend(cls_pos[test_end:valid_end].tolist())

    if grp_labels is not None and capped_pos:
        all_pos = np.concatenate(capped_pos)
        strata = samples_df[stratify_col].to_numpy()[all_pos]
        parts = split_groups_stratified(
            strata,
            grp_labels[all_pos],
            lambda key, rows: _cls_counts(
                key, len(rows), _cls_name(all_pos[strata == key])
            ),
            rng,
            describe_stratum=lambda key: (
                f"Class '{_cls_name(all_pos[strata == key])}' "
                f"({stratify_col} {key})"
                if has_cls_name
                else f"Class {key} ({stratify_col})"
            ),
        )
        train_idx, test_idx, valid_idx = (
            all_pos[np.asarray(part, dtype=np.intp)].tolist() for part in parts
        )

    train_df = samples_df.iloc[train_idx].reset_index(drop=True)
    test_df = samples_df.iloc[test_idx].reset_index(drop=True)
    valid_df = samples_df.iloc[valid_idx].reset_index(drop=True)

    for df_part, out_path in (
        (train_df, out_train_file),
        (test_df, out_test_file),
        (valid_df, out_valid_file),
    ):
        if out_path is not None:
            write_cls_samples(
                df_part,
                out_path,
                out_format=out_format,
                out_compress=out_compress,
            )

    return train_df, test_df, valid_df


def extract_and_split_class_samples(
    img_band_info: Union[ClassificationConfig, Sequence[ImageBandInfo]],
    class_info: Optional[Mapping[str, ClassInfoObj]] = None,
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    out_all_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    max_cls_smpls: Optional[Union[int, Mapping[Any, int]]] = None,
    n_train: Optional[Union[int, Mapping[Any, int]]] = None,
    n_test: Optional[Union[int, Mapping[Any, int]]] = None,
    n_valid: Optional[Union[int, Mapping[Any, int]]] = None,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
    *,
    poly_id_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract training samples across configured raster bands and vector classes,
    split them per class into training, testing, and validation DataFrames, and
    optionally write Parquet files (Snappy compressed by default) or another
    pandas-supported tabular format.

    With ``group_col``, the source polygon (or point) of each pixel is recorded
    in that column (see :func:`extract_class_training_samples`) and whole
    polygons are assigned to the training, testing or validation set, still
    stratified by class as far as the number of polygons allows (see
    :func:`split_class_samples`). This avoids pixels of one polygon being used
    both to train and to test a model. ``poly_id_col`` alone records the
    polygons but keeps the random pixel split.

    :param img_band_info: Either a :class:`ClassificationConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param class_info: Mapping of ``{class_name: ClassInfoObj}`` (required when
                       ``img_band_info`` is a sequence of :class:`ImageBandInfo`).
    :param train_prop: Proportion of samples per class for training (defaults to
                       ``1.0 - test_prop - valid_prop``).
    :param test_prop: Proportion of samples per class for testing
                      (Default = ``0.2``).
    :param valid_prop: Proportion of samples per class for validation
                       (Default = ``0.2``).
    :param out_train_file: Optional output file path for training samples.
    :param out_test_file: Optional output file path for testing samples.
    :param out_valid_file: Optional output file path for validation samples.
    :param out_all_file: Optional output file path for all extracted samples
                         prior to splitting.
    :param out_format: Optional output table format overriding extension
                       detection (Default = ``None``).
    :param out_compress: Compression setting passed to :func:`write_cls_samples`
                         (Default = ``'snappy'`` for Parquet).
    :param min_thres: Optional inclusive lower valid pixel threshold.
    :param max_thres: Optional inclusive upper valid pixel threshold.
    :param out_no_data_val: Optional fill/nodata sentinel value.
    :param mask_nodata: If ``True`` (default), exclude nodata/non-finite rows.
    :param all_touched: Passed to
                        :func:`rsgispy.zonalstats.polygon.export_poly_pxls_to_df`.
    :param include_coords: Include ``x`` and ``y`` columns (Default = ``False``).
    :param reproj_vec: Reproject vectors on the fly when CRSs differ.
    :param vec_def_epsg: Optional EPSG code overriding undefined vector CRS.
    :param max_cls_smpls: Optional per-class sample cap before splitting.
    :param n_train: Optional explicit training sample count per class.
    :param n_test: Optional explicit testing sample count per class.
    :param n_valid: Optional explicit validation sample count per class.
    :param rnd_seed: Optional random seed for reproducible splits.
    :param use_pbar: Display progress bars during extraction (Default = ``True``).
    :param poly_id_col: Optional column recording the source polygon (or
                        point) of each sample (Default = ``None``; set to
                        ``group_col`` when only that is given).
    :param group_col: Optional column by which the samples are grouped in the
                      split (the polygons are recorded in it). When ``None``
                      (default) the pixels are split at random.
    :return: Tuple ``(train_df, test_df, valid_df)``.

    .. code:: python

        import rsgispy.classification.classdata as cd

        cfg = cd.read_cls_config("cls_config.json")
        train_df, test_df, valid_df = cd.extract_and_split_class_samples(
            cfg,
            out_train_file="train.parquet",
            out_test_file="test.parquet",
            out_valid_file="valid.parquet",
            group_col="train_grp",  # whole polygons per set
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg = _coerce_config(img_band_info, class_info)
    rec_col = resolve_extract_split_group_col(poly_id_col, group_col)
    samples_df = extract_class_training_samples(
        cfg,
        out_file=out_all_file,
        out_format=out_format,
        out_compress=out_compress,
        min_thres=min_thres,
        max_thres=max_thres,
        out_no_data_val=out_no_data_val,
        mask_nodata=mask_nodata,
        all_touched=all_touched,
        include_coords=include_coords,
        reproj_vec=reproj_vec,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
        poly_id_col=rec_col,
    )
    return split_class_samples(
        samples_df=samples_df,
        train_prop=train_prop,
        test_prop=test_prop,
        valid_prop=valid_prop,
        out_train_file=out_train_file,
        out_test_file=out_test_file,
        out_valid_file=out_valid_file,
        out_format=out_format,
        out_compress=out_compress,
        stratify_col=cfg.internal_id_col,
        class_name_col=cfg.class_name_col,
        max_cls_smpls=max_cls_smpls,
        n_train=n_train,
        n_test=n_test,
        n_valid=n_valid,
        rnd_seed=rnd_seed,
        group_col=group_col,
    )


def _extract_normalised_classes(
    class_info: Union[ClassificationConfig, Mapping[str, ClassInfoObj]],
) -> Dict[str, ClassInfoObj]:
    """Return normalised ClassInfoObj mapping from config or dict."""
    if isinstance(class_info, ClassificationConfig):
        return class_info.class_info
    return normalise_class_info(class_info)


def get_cls_id_luts(
    class_info: Union[ClassificationConfig, Mapping[str, ClassInfoObj]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build forward and reverse lookup dictionaries mapping consecutive 0-based
    internal class IDs to user-specified output class IDs.

    :param class_info: A :class:`ClassificationConfig` or mapping of
                       ``{class_name: ClassInfoObj}``.
    :return: Tuple ``(internal_to_out_lut, out_to_internal_lut)``.

    .. code:: python

        import rsgispy.classification.classdata as cd

        int_to_out, out_to_int = cd.get_cls_id_luts(cls_info)
    """
    norm_classes = _extract_normalised_classes(class_info)
    internal_to_out = {
        int(obj.internal_id): int(obj.out_id)  # type: ignore[arg-type]
        for obj in norm_classes.values()
    }
    out_to_internal = {
        int(obj.out_id): int(obj.internal_id)  # type: ignore[arg-type]
        for obj in norm_classes.values()
    }
    return internal_to_out, out_to_internal


def recode_internal_to_out_ids(
    preds: Union[np.ndarray, pd.Series, Sequence[int]],
    class_info: Union[ClassificationConfig, Mapping[str, ClassInfoObj]],
    no_data_val: Optional[int] = None,
    out_dtype: Optional[Any] = None,
    keep_unmapped: bool = False,
) -> Union[np.ndarray, pd.Series]:
    """
    Recode predicted consecutive 0-based internal class IDs (e.g., from XGBoost
    or scikit-learn classifiers) back to the user-specified output class IDs.

    :param preds: Array, :class:`pandas.Series`, or sequence of integer
                  internal class predictions.
    :param class_info: A :class:`ClassificationConfig` or mapping of
                       ``{class_name: ClassInfoObj}``.
    :param no_data_val: Optional nodata/background value preserved unchanged.
    :param out_dtype: Optional NumPy output dtype. When ``None``, selects an
                      integer dtype large enough for all output class IDs.
    :param keep_unmapped: If ``True``, leave values absent from ``class_info``
                          unchanged; if ``False`` (default), raise
                          :class:`rsgispy.RSGISPyException`.
    :return: Recoded :class:`pandas.Series` (if ``preds`` was a Series) or
             :class:`numpy.ndarray`.

    .. code:: python

        import numpy as np
        import rsgispy.classification.classdata as cd

        out_preds = cd.recode_internal_to_out_ids(
            np.array([0, 1, 0]), cls_info
        )
    """
    # pylint: disable=too-many-locals
    internal_to_out, _ = get_cls_id_luts(class_info)
    target_vals = list(internal_to_out.values())
    if no_data_val is not None:
        target_vals.append(int(no_data_val))

    if out_dtype is None:
        _, resolved_dtype = select_out_int_dtype(target_vals)
    else:
        resolved_dtype = np.dtype(out_dtype)

    is_series = isinstance(preds, pd.Series)
    arr = preds.to_numpy() if is_series else np.asarray(preds)
    arr_int = arr.astype(np.int64, copy=False)

    out_arr = (
        arr_int.astype(resolved_dtype, copy=True)
        if keep_unmapped or no_data_val is not None
        else np.empty(arr_int.shape, dtype=resolved_dtype)
    )

    mapped_mask = np.zeros(arr_int.shape, dtype=bool)
    if no_data_val is not None:
        nd_mask = arr_int == int(no_data_val)
        out_arr[nd_mask] = int(no_data_val)
        mapped_mask |= nd_mask

    for int_id, out_id in internal_to_out.items():
        cls_mask = arr_int == int_id
        out_arr[cls_mask] = out_id
        mapped_mask |= cls_mask

    if not keep_unmapped and not np.all(mapped_mask):
        unknown_vals = np.unique(arr_int[~mapped_mask]).tolist()
        raise rsgispy.RSGISPyException(
            f"Unexpected internal class ID(s) not present in class_info: "
            f"{unknown_vals}."
        )

    if is_series:
        return pd.Series(out_arr, index=preds.index, name=preds.name)
    return out_arr


def recode_cls_img_to_out_ids(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    class_info: Union[ClassificationConfig, Mapping[str, ClassInfoObj]],
    set_thematic: bool = True,
    apply_colour_table: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[int] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Recode a classified raster image from consecutive 0-based internal class IDs
    back to the user-specified output class IDs, optionally assigning thematic
    metadata and class RGB colour tables.

    :param input_img: Path to the input classified raster containing internal
                      class IDs (``0..N-1``).
    :param output_img: Path to the recoded output classified raster.
    :param class_info: A :class:`ClassificationConfig` or mapping of
                       ``{class_name: ClassInfoObj}``.
    :param set_thematic: If ``True`` (default), mark output bands as thematic.
    :param apply_colour_table: If ``True`` (default), write the class RGB colour
                               table when output dtype is ``uint8`` or
                               ``uint16`` and all output IDs are ``>= 0``.
    :param gdalformat: Output GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer output datatype. When
                     ``None``, automatically selects the smallest integer type
                     that fits all output class IDs and ``no_data_val``.
    :param no_data_val: Optional nodata value to preserve/assign on output.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.classification.classdata as cd

        cfg = cd.read_cls_config("cls_config.json")
        cd.recode_cls_img_to_out_ids(
            input_img="cls_internal.tif",
            output_img="cls_output.tif",
            class_info=cfg,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path)
    norm_classes = _extract_normalised_classes(class_info)
    internal_to_out, _ = get_cls_id_luts(norm_classes)
    recode_dict = dict(internal_to_out)

    with rasterio.open(in_path, "r") as src:
        src_nodata = src.nodata

    val_candidates = list(internal_to_out.values())
    if no_data_val is not None:
        nd_int = int(no_data_val)
        val_candidates.append(nd_int)
        if src_nodata is not None and not (
            isinstance(src_nodata, float) and math.isnan(src_nodata)
        ):
            src_nd_num = float(src_nodata)
            if src_nd_num.is_integer() and int(src_nd_num) not in recode_dict:
                recode_dict[int(src_nd_num)] = nd_int
    elif src_nodata is not None and not (
        isinstance(src_nodata, float) and math.isnan(src_nodata)
    ):
        src_nd_num = float(src_nodata)
        if src_nd_num.is_integer() and int(src_nd_num) not in recode_dict:
            val_candidates.append(int(src_nd_num))

    if datatype is None:
        out_rsgis_dtype, out_np_dtype = select_out_int_dtype(val_candidates)
    else:
        out_rsgis_dtype = datatype
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    recode_int_raster(
        input_img=in_path,
        output_img=out_path,
        recode_dict=recode_dict,
        keep_vals_not_in_dict=True,
        gdalformat=gdalformat,
        datatype=out_rsgis_dtype,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )

    if no_data_val is not None:
        set_img_no_data_value(out_path, int(no_data_val), img_band=1)

    if set_thematic:
        set_img_thematic(out_path)

    if (
        apply_colour_table
        and out_np_dtype in (np.dtype(np.uint8), np.dtype(np.uint16))
        and all(obj.out_id >= 0 for obj in norm_classes.values())
    ):
        clr_lut = {
            int(obj.out_id): [obj.red, obj.green, obj.blue, 255]
            for obj in norm_classes.values()
        }
        define_colour_table(out_path, clr_lut, img_band=1)
