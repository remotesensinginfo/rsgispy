"""
The imageutils.envopts module provides functions for configuring GDAL/rasterio
image creation options via environment variables.
"""

import os
from typing import Any, Mapping

from rsgispy.imageprocess import get_gdal_img_creation_opts

__all__ = [
    "get_gdal_img_creation_opts",
    "set_img_creation_opts",
    "set_env_vars_lzw_gtiff_outs",
    "set_env_vars_deflate_gtiff_outs",
    "set_env_vars_zstd_gtiff_outs",
    "set_env_vars_lzw_cog_outs",
    "set_env_vars_deflate_cog_outs",
]


def set_img_creation_opts(
    gdalformat: str, creation_opts: Mapping[str, Any]
) -> None:
    """
    Set the environment variables ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` and
    ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` from a mapping of creation options.

    :param gdalformat: GDAL format string (e.g., 'GTIFF', 'COG').
    :param creation_opts: Mapping of GDAL creation option keys and values.

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_img_creation_opts(
            "GTIFF", {"TILED": "YES", "COMPRESS": "LZW"}
        )
    """
    fmt_upper = gdalformat.upper().strip()
    opts_str = ":".join(f"{k}={v}" for k, v in creation_opts.items())
    os.environ[f"RSGISLIB_IMG_CRT_OPTS_{fmt_upper}"] = opts_str
    os.environ[f"RSGISPY_IMG_CRT_OPTS_{fmt_upper}"] = opts_str


def set_env_vars_lzw_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the LZW algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts
        rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"TILED=YES:COMPRESS=LZW:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_deflate_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the Deflate (zlib) algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts
        rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = (
        f"TILED=YES:COMPRESS=DEFLATE:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    )
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_zstd_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the ZSTD algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"TILED=YES:COMPRESS=ZSTD:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_lzw_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the LZW algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_lzw_cog_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=LZW:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str


def set_env_vars_deflate_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the Deflate algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_deflate_cog_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=DEFLATE:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str
