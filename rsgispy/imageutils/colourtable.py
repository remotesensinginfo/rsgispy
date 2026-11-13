"""
The imageutils.colourtable module provides functions for inspecting and modifying
raster thematic metadata, band colour interpretation, and colour tables using
``rasterio``.
"""

import os
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import rasterio
from rasterio.enums import ColorInterp

import rsgispy
from rsgispy.imageprocess import validate_band_index as _validate_band_index
from rsgispy.imageutils._helpers import ensure_img_exists as _ensure_img_exists
from rsgispy.tools.utils import hex_to_rgb

__all__ = [
    "set_img_thematic",
    "set_img_not_thematic",
    "is_img_thematic",
    "get_img_band_colour_interp",
    "set_img_band_colour_interp",
    "get_colour_tab_info",
    "get_colour_tab_info_for_def",
    "define_colour_table",
]


def set_img_thematic(input_img: str) -> None:
    """
    Set all image bands in ``input_img`` to have ``LAYER_TYPE=thematic`` metadata.

    :param input_img: Path to the image file to update.
    """
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    try:
        with rasterio.open(input_img, "r+") as dst:
            for bidx in range(1, dst.count + 1):
                dst.update_tags(bidx, LAYER_TYPE="thematic")
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not update thematic metadata on '{input_img}': {err}"
        ) from err


def set_img_not_thematic(input_img: str) -> None:
    """
    Set all image bands in ``input_img`` to have ``LAYER_TYPE=athematic`` metadata.

    :param input_img: Path to the image file to update.
    """
    if not os.path.exists(input_img):
        raise rsgispy.RSGISPyException(
            f"Input image file does not exist: '{input_img}'."
        )
    try:
        with rasterio.open(input_img, "r+") as dst:
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
        colorinterps = list(dst.colorinterp)
        colorinterps[img_band - 1] = ci_enum
        dst.colorinterp = colorinterps


def get_colour_tab_info(
    input_img: str, img_band: int = 1
) -> Dict[int, Dict[str, int]]:
    """
    Retrieve the colour table associated with ``img_band`` of ``input_img``.

    :param input_img: Path to the input image file.
    :param img_band: 1-based band index (Default = ``1``).
    :return: Dictionary mapping pixel values to ``{'red', 'green', 'blue', 'alpha'}`` dicts.
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
    clr_spec: Union[str, Sequence[int]]
) -> Tuple[int, int, int, int]:
    """Parse a hex string or RGB/RGBA sequence into a validated ``(r, g, b, a)`` tuple."""
    if isinstance(clr_spec, str):
        red, green, blue = hex_to_rgb(clr_spec)
        alpha = 255
    elif isinstance(clr_spec, (list, tuple)) and len(clr_spec) in (3, 4):
        red = int(clr_spec[0])
        green = int(clr_spec[1])
        blue = int(clr_spec[2])
        alpha = int(clr_spec[3]) if len(clr_spec) == 4 else 255
    else:
        raise rsgispy.RSGISPyException(
            "Colour entry must be a hex string or a sequence of 3 (RGB) or 4 (RGBA) ints."
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
                    ``[R, G, B]`` sequences, or ``[R, G, B, A]`` sequences.
    :param img_band: 1-based band index (Default = ``1``).
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
        colorinterps = list(dst.colorinterp)
        colorinterps[img_band - 1] = ColorInterp.palette
        dst.colorinterp = colorinterps
