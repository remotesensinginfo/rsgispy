"""
The tools.sensors module provides metadata header parsers for Earth observation
sensors (Landsat Collection 1/2 MTL and Sentinel-2 L2A MTD XML).
"""

import datetime
import os
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import rsgispy
from rsgispy.imageprocess import ensure_file_exists

__all__ = [
    "read_landsat_mtl_to_dict",
    "read_sen2_l2a_mtd_to_dict",
]


def _local_tag(tag: str) -> str:
    """Strip any XML namespace URI prefix from ``tag``."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """Find the first direct child element whose local tag equals ``name``."""
    for child in parent:
        if _local_tag(child.tag) == name:
            return child
    return None


def _require_child(
    parent: ET.Element, name: str, err_msg: str
) -> ET.Element:
    """Return direct child ``name`` or raise :class:`rsgispy.RSGISPyException`."""
    child = _find_child(parent, name)
    if child is None:
        raise rsgispy.RSGISPyException(err_msg)
    return child


def _child_text(
    parent: ET.Element, name: str, default: str = ""
) -> str:
    """Return stripped text of direct child ``name``, or ``default``."""
    child = _find_child(parent, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_iso_datetime(raw_val: str) -> datetime.datetime:
    """Parse an ISO-8601 UTC timestamp string from a Sentinel-2 header."""
    cleaned = raw_val.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise rsgispy.RSGISPyException(
        f"Could not parse Sentinel-2 timestamp '{raw_val}'."
    )


def read_landsat_mtl_to_dict(in_mtl_file: str) -> Dict[str, Dict[str, str]]:
    """
    Read a Landsat MTL metadata header file (``*MTL.txt``) into a nested
    dictionary keyed by MTL group name.

    :param in_mtl_file: File path to the Landsat MTL header file.
    :return: Nested dictionary mapping group names to key-value metadata dicts.

    .. code:: python

        import rsgispy.tools.sensors

        ls_header = rsgispy.tools.sensors.read_landsat_mtl_to_dict("LC08_MTL.txt")
    """
    ensure_file_exists(in_mtl_file, label="Landsat MTL header file")

    header_params: Dict[str, Dict[str, str]] = {}
    group_stack: List[Dict[str, str]] = []
    with open(in_mtl_file, "r", encoding="utf-8") as h_file:
        for raw_line in h_file:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().replace('"', "")
            if key == "GROUP":
                grp_dict: Dict[str, str] = {}
                header_params[val] = grp_dict
                group_stack.append(grp_dict)
            elif key == "END_GROUP":
                if group_stack:
                    group_stack.pop()
            elif group_stack:
                group_stack[-1][key] = val

    if not header_params:
        raise rsgispy.RSGISPyException(
            f"No metadata groups found in Landsat MTL file '{in_mtl_file}'."
        )
    return header_params


def read_sen2_l2a_mtd_to_dict(mtd_header_file: str) -> Dict[str, Any]:
    """
    Parse a Sentinel-2 Level-2A XML metadata header file (``MTD_MSIL2A.xml``)
    into a structured dictionary.

    Handles all Sentinel-2 L2A PSD schema namespaces (e.g., ``psd-14`` and
    ``psd-15``) and optionally extracts ``BOA_ADD_OFFSET_VALUES_LIST`` when
    present (Processing Baseline >= 04.00).

    :param mtd_header_file: File path to ``MTD_MSIL2A.xml``.
    :return: Structured dictionary containing parsed Sentinel-2 L2A metadata.

    .. code:: python

        import rsgispy.tools.sensors

        s2_header = rsgispy.tools.sensors.read_sen2_l2a_mtd_to_dict("MTD_MSIL2A.xml")
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if not os.path.exists(mtd_header_file):
        raise rsgispy.RSGISPyException(
            f"MTD header file does not exist: '{mtd_header_file}'."
        )

    try:
        tree = ET.parse(mtd_header_file)
    except ET.ParseError as err:
        raise rsgispy.RSGISPyException(
            f"Failed to parse Sentinel-2 MTD XML '{mtd_header_file}': {err}"
        ) from err
    root = tree.getroot()

    general_info_tag = _require_child(
        root,
        "General_Info",
        "Cannot open top level section 'General_Info' - "
        "is this really a Sentinel-2 image file?",
    )
    product_info_tag = _require_child(
        general_info_tag,
        "Product_Info",
        "Product Info Tag is None - is this a Sentinel-2 L2A header.",
    )

    out_header_dict: Dict[str, Any] = {}
    prod_info: Dict[str, Any] = {}
    out_header_dict["Product_Info"] = prod_info

    for key in ("PRODUCT_START_TIME", "PRODUCT_STOP_TIME", "GENERATION_TIME"):
        raw_ts = _child_text(product_info_tag, key)
        prod_info[key] = _parse_iso_datetime(raw_ts) if raw_ts else None

    for key in (
        "PRODUCT_URI",
        "PROCESSING_LEVEL",
        "PRODUCT_TYPE",
        "PROCESSING_BASELINE",
        "PREVIEW_IMAGE_URL",
        "PREVIEW_GEO_INFO",
    ):
        prod_info[key] = _child_text(product_info_tag, key)

    data_take_tag = _require_child(
        product_info_tag,
        "Datatake",
        "Data Take Tag is None - is this a Sentinel-2 L2A header.",
    )
    sense_str = _child_text(data_take_tag, "DATATAKE_SENSING_START")
    prod_info["datatake"] = {
        "datatakeIdentifier": data_take_tag.attrib.get(
            "datatakeIdentifier", ""
        ).strip(),
        "SPACECRAFT_NAME": _child_text(data_take_tag, "SPACECRAFT_NAME"),
        "DATATAKE_TYPE": _child_text(data_take_tag, "DATATAKE_TYPE"),
        "DATATAKE_SENSING_START": (
            _parse_iso_datetime(sense_str) if sense_str else None
        ),
        "SENSING_ORBIT_NUMBER": _child_text(
            data_take_tag, "SENSING_ORBIT_NUMBER"
        ),
        "SENSING_ORBIT_DIRECTION": _child_text(
            data_take_tag, "SENSING_ORBIT_DIRECTION"
        ),
    }

    query_options_tag = _find_child(product_info_tag, "Query_Options")
    prod_info["Query_Options"] = {
        "PRODUCT_FORMAT": (
            _child_text(query_options_tag, "PRODUCT_FORMAT")
            if query_options_tag is not None
            else ""
        )
    }

    product_org_tag = _require_child(
        product_info_tag,
        "Product_Organisation",
        "Product Organisation Tag is None - is this a Sentinel-2 L2A header.",
    )
    granule_list_tag = _require_child(
        product_org_tag,
        "Granule_List",
        "Granule list Tag is None - is this a Sentinel-2 L2A header.",
    )

    granule_list: List[Dict[str, Any]] = []
    prod_info["Product_Organisation"] = {"Granule_List": granule_list}

    for granule_child in granule_list_tag:
        if _local_tag(granule_child.tag) == "Granule":
            img_files = [
                child.text.strip()
                for child in granule_child
                if _local_tag(child.tag) == "IMAGE_FILE" and child.text
            ]
            granule_list.append(
                {
                    "datastripIdentifier": granule_child.attrib.get(
                        "datastripIdentifier", ""
                    ).strip(),
                    "granuleIdentifier": granule_child.attrib.get(
                        "granuleIdentifier", ""
                    ).strip(),
                    "imageFormat": granule_child.attrib.get(
                        "imageFormat", ""
                    ).strip(),
                    "IMAGE_FILES": img_files,
                }
            )

    product_img_char_tag = _require_child(
        general_info_tag,
        "Product_Image_Characteristics",
        "Product Image Characteristics Tag is None - is this a Sentinel-2 L2A header.",
    )

    img_char: Dict[str, Any] = {
        "Special_Values": {},
        "Image_Display_Order": {},
        "Quantification_values": {},
        "BOA_Add_Offsets": {},
        "Reflectance_Conversion": {"Solar_Irradiance": {}},
        "Spectral_Information_List": {},
        "Physical_Gains": {},
        "Scene_Classification": {},
    }
    out_header_dict["Product_Image_Characteristics"] = img_char
    out_header_dict["Band_LUT"] = {"Physical_Band": {}, "Band_ID": {}}

    for child in product_img_char_tag:
        tag_name = _local_tag(child.tag)
        if tag_name == "Special_Values":
            sv_name = _child_text(child, "SPECIAL_VALUE_TEXT")
            sv_idx = _child_text(child, "SPECIAL_VALUE_INDEX")
            if sv_name and sv_idx:
                img_char["Special_Values"][sv_name] = int(sv_idx)
        elif tag_name == "Image_Display_Order":
            for ch_name in ("RED_CHANNEL", "GREEN_CHANNEL", "BLUE_CHANNEL"):
                val_txt = _child_text(child, ch_name)
                if val_txt:
                    img_char["Image_Display_Order"][ch_name] = int(val_txt)
        elif tag_name == "QUANTIFICATION_VALUES_LIST":
            boa_txt = _child_text(child, "BOA_QUANTIFICATION_VALUE")
            aot_txt = _child_text(child, "AOT_QUANTIFICATION_VALUE")
            wvp_txt = _child_text(child, "WVP_QUANTIFICATION_VALUE")
            if boa_txt:
                img_char["Quantification_values"]["BOA_QUANTIFICATION_VALUE"] = int(
                    boa_txt
                )
            if aot_txt:
                img_char["Quantification_values"]["AOT_QUANTIFICATION_VALUE"] = (
                    float(aot_txt)
                )
            if wvp_txt:
                img_char["Quantification_values"]["WVP_QUANTIFICATION_VALUE"] = (
                    float(wvp_txt)
                )
        elif tag_name == "BOA_ADD_OFFSET_VALUES_LIST":
            for offset_child in child:
                if _local_tag(offset_child.tag) == "BOA_ADD_OFFSET" and offset_child.text:
                    band_id = int(offset_child.attrib["band_id"].strip())
                    img_char["BOA_Add_Offsets"][band_id] = float(
                        offset_child.text.strip()
                    )
        elif tag_name == "Reflectance_Conversion":
            u_txt = _child_text(child, "U")
            if u_txt:
                img_char["Reflectance_Conversion"]["U"] = float(u_txt)
            irr_list = _find_child(child, "Solar_Irradiance_List")
            if irr_list is not None:
                for irr_child in irr_list:
                    if _local_tag(irr_child.tag) == "SOLAR_IRRADIANCE" and irr_child.text:
                        irr_band = int(irr_child.attrib["bandId"].strip())
                        img_char["Reflectance_Conversion"]["Solar_Irradiance"][
                            irr_band
                        ] = float(irr_child.text.strip())
        elif tag_name == "Spectral_Information_List":
            for spec_child in child:
                if _local_tag(spec_child.tag) != "Spectral_Information":
                    continue
                band_id = int(spec_child.attrib["bandId"].strip())
                phys_band = spec_child.attrib["physicalBand"].strip()
                out_header_dict["Band_LUT"]["Physical_Band"][phys_band] = band_id
                out_header_dict["Band_LUT"]["Band_ID"][band_id] = phys_band
                spec_info: Dict[str, Any] = {}
                res_txt = _child_text(spec_child, "RESOLUTION")
                if res_txt:
                    spec_info["RESOLUTION"] = int(res_txt)
                wl_tag = _find_child(spec_child, "Wavelength")
                if wl_tag is not None:
                    spec_info["Wavelength"] = {
                        "MIN": float(_child_text(wl_tag, "MIN", "0")),
                        "MAX": float(_child_text(wl_tag, "MAX", "0")),
                        "CENTRAL": float(_child_text(wl_tag, "CENTRAL", "0")),
                    }
                sr_tag = _find_child(spec_child, "Spectral_Response")
                if sr_tag is not None:
                    vals_str = _child_text(sr_tag, "VALUES")
                    spec_info["Spectral_Response"] = {
                        "STEP": float(_child_text(sr_tag, "STEP", "0")),
                        "Values": [float(v) for v in vals_str.split()] if vals_str else [],
                    }
                img_char["Spectral_Information_List"][band_id] = spec_info
        elif tag_name == "PHYSICAL_GAINS" and child.text:
            band_id = int(child.attrib["bandId"].strip())
            img_char["Physical_Gains"][band_id] = float(child.text.strip())
        elif tag_name == "Scene_Classification_List":
            for scn_child in child:
                if _local_tag(scn_child.tag) == "Scene_Classification_ID":
                    cls_name = _child_text(scn_child, "SCENE_CLASSIFICATION_TEXT")
                    cls_val = _child_text(scn_child, "SCENE_CLASSIFICATION_INDEX")
                    if cls_name:
                        img_char["Scene_Classification"][cls_name] = cls_val

    return out_header_dict
