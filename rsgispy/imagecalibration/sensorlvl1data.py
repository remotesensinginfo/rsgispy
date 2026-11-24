"""
The imagecalibration.sensorlvl1data module provides utilities for ingesting
Level-1 satellite data archives into analysis-ready Top-of-Atmosphere (TOA)
reflectance products.
"""

from typing import Any, Mapping, Optional, Sequence, Union

from rsgispy.imagecalibration._landsat_c2 import (
    LS_OLI_LV1_SPEC,
    process_landsat_c2_archive,
)

__all__ = [
    "create_stacked_toa_ls_oli_cl2_lv1_img",
]


def create_stacked_toa_ls_oli_cl2_lv1_img(
    input_file: str,
    out_dir: str,
    tmp_dir: str,
    scale_factor: int = 10000,
    gdalformat: str = "GTIFF",
    delete_inter_data: bool = True,
    inc_full_qa_pxl_img: bool = False,
    *,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Extract USGS Landsat Collection 2 Level-1 (Landsat 8/9 OLI) data from an
    input archive and create a single stacked multi-band Top-of-Atmosphere (TOA)
    reflectance raster (32-bit integer, nodata = ``-9999``), along with cloud
    and valid-pixel masks derived from ``QA_PIXEL``.

    :param input_file: File path to the input Landsat Collection 2 archive
                       (``.tar``, ``.tar.gz``, ``.tar.bz2``, or ``.zip``).
    :param out_dir: Output directory where calibrated rasters and masks will be
                    written.
    :param tmp_dir: Temporary working directory for archive extraction.
    :param scale_factor: Multiplicative scale factor applied to TOA reflectance
                         values (Default = ``10000``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param delete_inter_data: Delete extracted intermediate files when done
                              (Default = ``True``).
    :param inc_full_qa_pxl_img: If ``True``, also write the decoded 12-band
                                pixel QA raster (Default = ``False``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.sensorlvl1data as l1

        l1.create_stacked_toa_ls_oli_cl2_lv1_img(
            "LC08_L1TP_135046_20210319_20210328_02_T1.tar",
            "outputs",
            "tmp",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    process_landsat_c2_archive(
        input_file=input_file,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        spec=LS_OLI_LV1_SPEC,
        scale_factor=scale_factor,
        gdalformat=gdalformat,
        delete_inter_data=delete_inter_data,
        inc_full_qa_pxl_img=inc_full_qa_pxl_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
