"""
The imageutils.colourtable module provides functions for inspecting and modifying
raster thematic metadata, band colour interpretation, and colour tables using
``rasterio``.
"""

import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import rasterio
from rasterio.enums import ColorInterp

import rsgispy
from rsgispy.imageprocess import is_cog_layout as _is_cog_layout
from rsgispy.imageprocess import update_cog as _update_cog
from rsgispy.imageprocess import validate_band_index as _validate_band_index
from rsgispy.imageutils._helpers import ensure_img_exists as _ensure_img_exists
from rsgispy.tools.colours import hex_to_rgb

__all__ = [
    "set_img_thematic",
    "set_img_not_thematic",
    "is_img_thematic",
    "get_img_band_colour_interp",
    "set_img_band_colour_interp",
    "get_colour_tab_info",
    "get_colour_tab_info_for_def",
    "define_colour_table",
    "set_thematic_classes",
    "get_thematic_classes",
    "CLASS_NAMES_TAG",
]

#: Band tag holding the class names written by :func:`set_thematic_classes`:
#: a JSON object mapping each pixel value (as a string) to its class name.
CLASS_NAMES_TAG = "RSGISPY_CLASS_NAMES"


def set_img_thematic(input_img: str, ignore_cog: bool = False) -> None:
    """
    Set all image bands in ``input_img`` to have ``LAYER_TYPE=thematic`` metadata.

    :param input_img: Path to the image file to update.
    :param ignore_cog: If ``True``, pass ``IGNORE_COG_LAYOUT_BREAK=YES`` when
                       updating a COG file in place (Default = ``False``).

    .. code:: python

        import rsgispy.imageutils.colourtable

        rsgispy.imageutils.colourtable.set_img_thematic("cls_img.tif")
    """
    _ensure_img_exists(input_img)
    open_opts: Dict[str, str] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"
    try:
        with rasterio.open(input_img, "r+", **open_opts) as dst:
            for bidx in range(1, dst.count + 1):
                dst.update_tags(bidx, LAYER_TYPE="thematic")
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not update thematic metadata on '{input_img}': {err}"
        ) from err


def set_img_not_thematic(input_img: str, ignore_cog: bool = False) -> None:
    """
    Set all image bands in ``input_img`` to have ``LAYER_TYPE=athematic`` metadata.

    :param input_img: Path to the image file to update.
    :param ignore_cog: If ``True``, pass ``IGNORE_COG_LAYOUT_BREAK=YES`` when
                       updating a COG file in place (Default = ``False``).

    .. code:: python

        import rsgispy.imageutils.colourtable

        rsgispy.imageutils.colourtable.set_img_not_thematic("img.tif")
    """
    _ensure_img_exists(input_img)
    open_opts: Dict[str, str] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"
    try:
        with rasterio.open(input_img, "r+", **open_opts) as dst:
            for bidx in range(1, dst.count + 1):
                dst.update_tags(bidx, LAYER_TYPE="athematic")
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not update thematic metadata on '{input_img}': {err}"
        ) from err


def is_img_thematic(input_img: str, img_band: int = 1) -> bool:
    """
    Check whether band ``img_band`` of ``input_img`` has ``LAYER_TYPE=thematic``
    metadata.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index to inspect (Default = ``1``).
    :return: ``True`` if the band is marked thematic, ``False`` otherwise.

    .. code:: python

        import rsgispy.imageutils.colourtable

        thematic = rsgispy.imageutils.colourtable.is_img_thematic("cls_img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        tags = src.tags(img_band)
        return tags.get("LAYER_TYPE") == "thematic"


def get_img_band_colour_interp(input_img: str, img_band: int) -> int:
    """
    Retrieve the colour interpretation integer code for a band of ``input_img``.

    The integer corresponds to GDAL / ``rasterio.enums.ColorInterp`` values
    (e.g., ``ColorInterp.undefined == 0``, ``ColorInterp.gray == 1``,
    ``ColorInterp.palette == 2``, ``ColorInterp.red == 3``, etc.).

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index.
    :return: Colour interpretation integer code.

    .. code:: python

        import rsgispy.imageutils.colourtable

        ci = rsgispy.imageutils.colourtable.get_img_band_colour_interp("img.tif", 1)
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        return int(src.colorinterp[img_band - 1].value)


def set_img_band_colour_interp(
    input_img: str, img_band: int, clr_itrp_val: Union[int, ColorInterp]
) -> None:
    """
    Set the colour interpretation for a specific band of ``input_img``.

    :param input_img: Path to the image file to update.
    :param img_band: 1-based band index.
    :param clr_itrp_val: Integer colour interpretation code or
                         ``rasterio.enums.ColorInterp`` member.

    .. code:: python

        from rasterio.enums import ColorInterp
        import rsgispy.imageutils.colourtable

        rsgispy.imageutils.colourtable.set_img_band_colour_interp(
            "img.tif", 1, ColorInterp.red
        )
    """
    _ensure_img_exists(input_img)
    try:
        ci_enum = (
            clr_itrp_val
            if isinstance(clr_itrp_val, ColorInterp)
            else ColorInterp(int(clr_itrp_val))
        )
    except (TypeError, ValueError) as err:
        raise rsgispy.RSGISPyException(
            f"Invalid colour interpretation value: {clr_itrp_val!r}."
        ) from err

    with rasterio.open(input_img, "r+") as dst:
        _validate_band_index(img_band, dst.count, input_img)
        clr_interps = list(dst.colorinterp)
        clr_interps[img_band - 1] = ci_enum
        dst.colorinterp = clr_interps


def get_colour_tab_info(
    input_img: str, img_band: int = 1
) -> Dict[int, Dict[str, int]]:
    """
    Retrieve the colour table associated with ``img_band`` of ``input_img``.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index (Default = ``1``).
    :return: Dictionary mapping pixel values to ``{'red', 'green', 'blue', 'alpha'}`` dicts.

    .. code:: python

        import rsgispy.imageutils.colourtable

        clr_tab = rsgispy.imageutils.colourtable.get_colour_tab_info("cls_img.tif")
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        try:
            cmap = src.colormap(img_band)
        except ValueError as err:
            raise rsgispy.RSGISPyException(
                f"Band {img_band} of image '{input_img}' does not have a colour table."
            ) from err

    if not cmap:
        raise rsgispy.RSGISPyException(
            f"Band {img_band} of image '{input_img}' does not have a colour table."
        )

    clr_tab: Dict[int, Dict[str, int]] = {}
    for pxl_val, rgba in cmap.items():
        clr_tab[int(pxl_val)] = {
            "red": int(rgba[0]),
            "green": int(rgba[1]),
            "blue": int(rgba[2]),
            "alpha": int(rgba[3]) if len(rgba) > 3 else 255,
        }
    return clr_tab


def get_colour_tab_info_for_def(
    input_img: str, img_band: int = 1
) -> Dict[int, List[int]]:
    """
    Retrieve the colour table associated with ``img_band`` of ``input_img`` in
    a format suitable for :func:`define_colour_table`.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index (Default = ``1``).
    :return: Dictionary mapping pixel values to ``[red, green, blue, alpha]`` lists.

    .. code:: python

        import rsgispy.imageutils.colourtable

        clr_lut = rsgispy.imageutils.colourtable.get_colour_tab_info_for_def(
            "cls_img.tif"
        )
    """
    clr_tab = get_colour_tab_info(input_img, img_band=img_band)
    return {
        pxl_val: [
            entry["red"],
            entry["green"],
            entry["blue"],
            entry["alpha"],
        ]
        for pxl_val, entry in clr_tab.items()
    }


def _parse_colour_entry(
    clr_spec: Union[str, Sequence[int], Mapping[str, int]]
) -> Tuple[int, int, int, int]:
    """
    Parse a hex string, an RGB/RGBA sequence or a ``{'red', 'green', 'blue',
    'alpha'}`` dict (as returned by :func:`get_colour_tab_info`) into a
    validated ``(r, g, b, a)`` tuple.
    """
    if isinstance(clr_spec, str):
        hex_str = clr_spec.strip().lstrip("#")
        if len(hex_str) == 8:
            # '#rrggbbaa', as returned by get_thematic_classes.
            red, green, blue = hex_to_rgb(hex_str[:6])
            try:
                alpha = int(hex_str[6:], 16)
            except ValueError as err:
                raise rsgispy.RSGISPyException(
                    f"Invalid hexadecimal colour string: {clr_spec!r}."
                ) from err
        else:
            red, green, blue = hex_to_rgb(clr_spec)
            alpha = 255
    elif isinstance(clr_spec, Mapping) and {"red", "green", "blue"} <= set(clr_spec):
        # {'red', 'green', 'blue'[, 'alpha']}, as returned by get_colour_tab_info.
        red = int(clr_spec["red"])
        green = int(clr_spec["green"])
        blue = int(clr_spec["blue"])
        alpha = int(clr_spec.get("alpha", 255))
    elif isinstance(clr_spec, (list, tuple)) and len(clr_spec) in (3, 4):
        red = int(clr_spec[0])
        green = int(clr_spec[1])
        blue = int(clr_spec[2])
        alpha = int(clr_spec[3]) if len(clr_spec) == 4 else 255
    else:
        raise rsgispy.RSGISPyException(
            "Colour entry must be a hex string, a sequence of 3 (RGB) or 4 (RGBA) ints, "
            "or a dict with 'red', 'green', 'blue' (and 'alpha') keys."
        )

    for comp_name, comp_val in (
        ("red", red),
        ("green", green),
        ("blue", blue),
        ("alpha", alpha),
    ):
        if comp_val < 0 or comp_val > 255:
            raise rsgispy.RSGISPyException(
                f"Colour component '{comp_name}' must be in 0..255, got {comp_val}."
            )
    return (red, green, blue, alpha)


def define_colour_table(
    input_img: str,
    clr_lut: Mapping[int, Union[str, Sequence[int]]],
    img_band: int = 1,
) -> None:
    """
    Define and write a colour table onto ``img_band`` of ``input_img``.

    :param input_img: Path to the image file to update.
    :param clr_lut: Mapping from integer pixel values to hex strings (e.g., ``'#FF0000'``),
                    ``[R, G, B]`` sequences, ``[R, G, B, A]`` sequences, or
                    ``{'red', 'green', 'blue', 'alpha'}`` dicts, so the output of
                    :func:`get_colour_tab_info` can be copied to another image.
    :param img_band: 1-based band index (Default = ``1``).

    .. code:: python

        import rsgispy.imageutils.colourtable

        clr_lut = {1: "#FF0000", 2: [0, 255, 0, 255]}
        rsgispy.imageutils.colourtable.define_colour_table("cls_img.tif", clr_lut)
    """
    _ensure_img_exists(input_img)
    if not clr_lut:
        raise rsgispy.RSGISPyException("clr_lut must be a non-empty mapping.")

    cmap: Dict[int, Tuple[int, int, int, int]] = {}
    for pxl_val, clr_spec in clr_lut.items():
        idx = int(pxl_val)
        if idx < 0:
            raise rsgispy.RSGISPyException(
                f"Colour table pixel values must be non-negative integers, got {idx}."
            )
        cmap[idx] = _parse_colour_entry(clr_spec)

    with rasterio.open(input_img, "r+") as dst:
        _validate_band_index(img_band, dst.count, input_img)
        if dst.dtypes[img_band - 1] not in ("uint8", "uint16"):
            raise rsgispy.RSGISPyException(
                f"Colour tables require an unsigned integer (uint8/uint16) band, "
                f"got '{dst.dtypes[img_band - 1]}'."
            )
        dst.write_colormap(img_band, cmap)
        clr_interps = list(dst.colorinterp)
        clr_interps[img_band - 1] = ColorInterp.palette
        dst.colorinterp = clr_interps


ThematicClassesType = Mapping[int, Tuple[str, Union[str, Sequence[int]]]]


def _parse_thematic_classes(
    classes: ThematicClassesType,
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int, int, int]]]:
    """Split ``{id: (name, colour)}`` into validated name and RGBA mappings."""
    if not classes:
        raise rsgispy.RSGISPyException("classes must be a non-empty mapping.")
    names: Dict[int, str] = {}
    colours: Dict[int, Tuple[int, int, int, int]] = {}
    for cls_id, cls_def in classes.items():
        try:
            idx = int(cls_id)
        except (TypeError, ValueError) as err:
            raise rsgispy.RSGISPyException(
                f"Class ids must be integers, got {cls_id!r}."
            ) from err
        if idx < 0:
            raise rsgispy.RSGISPyException(
                f"Class ids must be non-negative integers, got {idx}."
            )
        if not isinstance(cls_def, (list, tuple)) or len(cls_def) != 2:
            raise rsgispy.RSGISPyException(
                f"Class {idx} must be given as (name, colour), got {cls_def!r}."
            )
        cls_name, cls_clr = cls_def
        if not isinstance(cls_name, str):
            raise rsgispy.RSGISPyException(
                f"The name of class {idx} must be a string, got {cls_name!r}."
            )
        names[idx] = cls_name
        colours[idx] = _parse_colour_entry(cls_clr)
    return names, colours


def _read_band_colour_tables(
    input_img: str,
) -> Dict[int, Optional[Dict[int, Tuple[int, ...]]]]:
    """Return each band's colour table (``None`` where a band has none)."""
    clr_tabs: Dict[int, Optional[Dict[int, Tuple[int, ...]]]] = {}
    with rasterio.open(input_img, "r") as src:
        for bidx in range(1, src.count + 1):
            try:
                cmap = src.colormap(bidx)
            except ValueError:
                cmap = None
            clr_tabs[bidx] = (
                {int(k): tuple(int(c) for c in v) for k, v in cmap.items()}
                if cmap
                else None
            )
    return clr_tabs


def _read_class_names_tag(names_json: Optional[str], err_msg: str) -> Dict[int, str]:
    """Parse the :data:`CLASS_NAMES_TAG` JSON into ``{pixel value: name}``."""
    if not names_json:
        return {}
    try:
        return {int(k): str(v) for k, v in json.loads(names_json).items()}
    except (ValueError, TypeError, AttributeError) as err:
        raise rsgispy.RSGISPyException(
            f"{err_msg}: the {CLASS_NAMES_TAG} tag is not a valid JSON object ({err})."
        ) from err


def _apply_thematic_classes(
    input_img: str,
    names: Dict[int, str],
    colours: Dict[int, Tuple[int, int, int, int]],
    *,
    img_band: int,
    ignore_zero: bool,
    calc_stats: bool,
    calc_pyramids: bool,
    ignore_cog: bool,
    use_pbar: bool,
) -> None:
    """Write the classes onto a (non-COG) image; see :func:`set_thematic_classes`."""
    # pylint: disable=too-many-arguments,too-many-locals
    # Imported here: rsgispy.imageutils.stats imports this module.
    from rsgispy.imageutils.stats import (  # pylint: disable=import-outside-toplevel
        pop_thmt_img_stats,
    )

    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        band_dtype = src.dtypes[img_band - 1]
        old_names = _read_class_names_tag(
            src.tags(img_band).get(CLASS_NAMES_TAG), f"Band {img_band} of '{input_img}'"
        )
    if band_dtype not in ("uint8", "uint16"):
        raise rsgispy.RSGISPyException(
            f"Colour tables require an unsigned integer (uint8/uint16) band, "
            f"got '{band_dtype}'."
        )
    max_id = 255 if band_dtype == "uint8" else 65535
    if max(colours) > max_id:
        raise rsgispy.RSGISPyException(
            f"Class id {max(colours)} does not fit in a {band_dtype} band."
        )

    # Read the colour tables before the statistics are calculated, because
    # pop_thmt_img_stats writes a new (random) colour table to every band.
    old_clr_tabs = _read_band_colour_tables(input_img)

    if calc_stats:
        pop_thmt_img_stats(
            input_img,
            add_clr_tab=True,
            calc_pyramids=calc_pyramids,
            ignore_zero=ignore_zero,
            use_pbar=use_pbar,
            ignore_cog=ignore_cog,
        )
    else:
        set_img_thematic(input_img, ignore_cog=ignore_cog)

    clr_tab = dict(old_clr_tabs[img_band] or {})
    if ignore_zero and 0 not in colours:
        clr_tab[0] = (0, 0, 0, 0)
    clr_tab.update(colours)

    all_names = dict(old_names)
    all_names.update(names)
    names_json = json.dumps({str(k): all_names[k] for k in sorted(all_names)})

    open_opts: Dict[str, str] = {}
    if ignore_cog:
        open_opts["IGNORE_COG_LAYOUT_BREAK"] = "YES"
    with rasterio.open(input_img, "r+", **open_opts) as dst:
        for bidx, old_tab in old_clr_tabs.items():
            if bidx != img_band and old_tab is not None:
                dst.write_colormap(bidx, old_tab)
        dst.write_colormap(img_band, clr_tab)
        clr_interps = list(dst.colorinterp)
        clr_interps[img_band - 1] = ColorInterp.palette
        dst.colorinterp = clr_interps
        if all_names:
            dst.update_tags(img_band, **{CLASS_NAMES_TAG: names_json})


def set_thematic_classes(
    input_img: str,
    classes: ThematicClassesType,
    *,
    img_band: int = 1,
    ignore_zero: bool = True,
    calc_stats: bool = True,
    ignore_cog: bool = False,
    use_pbar: bool = False,
) -> None:
    """
    Set the class names and colours of a thematic (class) image in one call.

    This replaces the sequence of :func:`set_img_thematic`,
    :func:`define_colour_table` and
    :func:`rsgispy.imageutils.stats.pop_thmt_img_stats`. It:

    1. marks every band of the image thematic (``LAYER_TYPE=thematic``);
    2. writes the class colours into the colour table of ``img_band``,
       merged with any colour table the band already has (the entries for
       other pixel values are kept; with ``ignore_zero`` and no class ``0``,
       pixel value ``0`` is made transparent);
    3. stores the class names, merged with any names stored before, as a JSON
       band tag (:data:`CLASS_NAMES_TAG`, ``RSGISPY_CLASS_NAMES``) mapping each
       pixel value to its name; read them back with
       :func:`get_thematic_classes`;
    4. when ``calc_stats``, calculates the thematic statistics and builds the
       overview pyramids, as :func:`~rsgispy.imageutils.stats.pop_thmt_img_stats`
       does, without replacing the colour table.

    The names are stored as a band tag because rasterio has no API for GDAL
    category names. The tag is kept inside the GeoTIFF (and through COG
    conversion), so no ``.aux.xml`` sidecar is needed. GeoTIFF colour tables
    have no alpha channel and only band 1 can have one: an alpha given for a
    class is ignored by the GeoTIFF driver.

    :param input_img: Path to the thematic image to update. The band must be
                      ``uint8`` or ``uint16`` (colour tables need an unsigned
                      integer band).
    :param classes: Mapping of pixel value to ``(name, colour)``, where the
                    colour is a hex string (``'#rrggbb'``, e.g. ``'#1f78b4'``,
                    or ``'#rrggbbaa'``) or an
                    ``(r, g, b)`` / ``(r, g, b, a)`` sequence of ``0..255``
                    ints.
    :param img_band: 1-based band to which the classes belong (Default = ``1``).
    :param ignore_zero: Whether pixel value ``0`` is no data: it is set as the
                        image no-data value, left out of the statistics and
                        made transparent unless it is one of ``classes``
                        (Default = ``True``).
    :param calc_stats: Whether to calculate the thematic statistics and build
                       the pyramids (Default = ``True``).
    :param ignore_cog: If ``True``, pass ``IGNORE_COG_LAYOUT_BREAK=YES`` when
                       updating the file in place. A Cloud Optimised GeoTIFF
                       (see :func:`rsgispy.imageprocess.is_cog_layout`) is
                       never edited in place: it is rebuilt with
                       :func:`rsgispy.imageprocess.update_cog`, as
                       :func:`~rsgispy.imageutils.stats.pop_thmt_img_stats`
                       does, and keeps its COG layout (Default = ``False``).
    :param use_pbar: Whether to show a progress bar while the statistics are
                     calculated (Default = ``False``).

    .. code:: python

        import rsgispy.imageutils.colourtable

        rsgispy.imageutils.colourtable.set_thematic_classes(
            "cls_img.tif",
            {
                1: ("Water", "#1f78b4"),
                2: ("Forest", (51, 160, 44)),
                3: ("Urban", (227, 26, 28, 255)),
            },
        )
    """
    # pylint: disable=too-many-arguments
    _ensure_img_exists(input_img)
    names, colours = _parse_thematic_classes(classes)

    if _is_cog_layout(input_img):
        with rasterio.open(input_img, "r") as src:
            _validate_band_index(img_band, src.count, input_img)
            ovr_resamp = str(
                src.tags(ns="IMAGE_STRUCTURE").get("OVERVIEW_RESAMPLING", "")
            ).upper()
        _update_cog(
            input_img,
            lambda tmp_img: _apply_thematic_classes(
                tmp_img,
                names,
                colours,
                img_band=img_band,
                ignore_zero=ignore_zero,
                calc_stats=calc_stats,
                calc_pyramids=False,
                ignore_cog=False,
                use_pbar=use_pbar,
            ),
            overview_resampling=(
                None if ovr_resamp in ("NEAREST", "MODE") else "NEAREST"
            ),
        )
        return

    _apply_thematic_classes(
        input_img,
        names,
        colours,
        img_band=img_band,
        ignore_zero=ignore_zero,
        calc_stats=calc_stats,
        calc_pyramids=calc_stats,
        ignore_cog=ignore_cog,
        use_pbar=use_pbar,
    )


def get_thematic_classes(
    input_img: str, img_band: int = 1
) -> Dict[int, Tuple[str, Optional[str]]]:
    """
    Read the class names and colours stored by :func:`set_thematic_classes`.

    :param input_img: Path to the thematic image.
    :param img_band: 1-based band index (Default = ``1``).
    :return: Mapping of pixel value to ``(name, colour)``, sorted by pixel
             value. The colour comes from the band's colour table as a
             ``'#rrggbb'`` hex string (``'#rrggbbaa'`` when the entry is not
             fully opaque), or is ``None`` when the band has no colour table.
             The mapping is empty when no class names are stored. It can be
             passed back to :func:`set_thematic_classes` (when every class
             has a colour) or used as the ``classes_info`` of the
             :mod:`rsgispy.mapping.legend` functions.

    .. code:: python

        import rsgispy.imageutils.colourtable

        classes = rsgispy.imageutils.colourtable.get_thematic_classes("cls_img.tif")
        for pxl_val, (name, colour) in classes.items():
            print(pxl_val, name, colour)
    """
    _ensure_img_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        _validate_band_index(img_band, src.count, input_img)
        names = _read_class_names_tag(
            src.tags(img_band).get(CLASS_NAMES_TAG), f"Band {img_band} of '{input_img}'"
        )
        try:
            cmap = src.colormap(img_band) if names else {}
        except ValueError:
            cmap = {}

    classes: Dict[int, Tuple[str, Optional[str]]] = {}
    for pxl_val in sorted(names):
        rgba = cmap.get(pxl_val)
        hex_clr: Optional[str] = None
        if rgba is not None:
            hex_clr = f"#{int(rgba[0]):02x}{int(rgba[1]):02x}{int(rgba[2]):02x}"
            alpha = int(rgba[3]) if len(rgba) > 3 else 255
            if alpha != 255:
                hex_clr += f"{alpha:02x}"
        classes[pxl_val] = (names[pxl_val], hex_clr)
    return classes
