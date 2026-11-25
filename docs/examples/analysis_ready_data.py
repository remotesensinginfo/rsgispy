"""
Worked example: analysis-ready Sentinel-2 data with rsgispy, from download to
a cloud-free composite.

This script is included step by step in the documentation guide
``docs/guides/analysis_ready_data.rst`` and is run end-to-end by the test
suite (``tests/test_analysis_ready_data_example.py``) on small synthetic
``.SAFE`` products, so the documented code is always tested.

Run it on scenes you have already downloaded with, e.g.::

    python analysis_ready_data.py --safe S2*_MSIL2A_*.SAFE --out-dir outputs

or search for and download the scenes first (set the ``RSGIS_COP_USER`` and
``RSGIS_COP_PASS`` environment variables to your Copernicus Data Space
Ecosystem login)::

    python analysis_ready_data.py --bbox -4.2 -3.9 52.3 52.5 \\
        --dates 2024-06-01 2024-08-31 --max-cloud 40 --out-dir outputs
"""

import argparse
import datetime
import glob
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def run_workflow(  # noqa: C901
    out_dir: str,
    scenes: Optional[Mapping[str, Sequence[str]]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    dates: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
    max_cloud: float = 40.0,
    max_scns: int = 25,
    use_cloudsen12: bool = False,
    device: str = "cpu",
    coreg_band: int = 7,
    coreg_win_size: int = 64,
    coreg_pixel_gap: int = 250,
    out_epsg: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``scenes`` maps a Sentinel-2 tile name (e.g.
    ``'T30UVD'``) to its Level-2A ``.SAFE`` directories. If ``bbox``
    (``(x_min, x_max, y_min, y_max)`` in WGS84) and ``dates`` are given, the
    scenes are searched for and downloaded from the Copernicus Data Space
    Ecosystem first and added to ``scenes``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    # pylint: disable=too-many-branches
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    scenes = {tile: list(safes) for tile, safes in (scenes or {}).items()}
    out: Dict[str, Any] = {}

    # [step-1-download]
    if bbox is not None and dates is not None:
        import rsgispy.dataaccess.copernicus_odata as cdse
        import rsgispy.tools.archives
        import rsgispy.tools.credentials

        scns = cdse.query_scn_lst(
            cdse.RSGIS_ODATA_SEN2,
            bbox,  # (x_min, x_max, y_min, y_max) in WGS84 degrees
            start_date=dates[0],
            end_date=dates[1],
            cloud_cover=max_cloud,  # maximum scene cloud cover (%)
            product_type=cdse.RSGIS_ODATA_PROD_TYPE_S2_MSI_2A,
            max_n_rslts=max_scns,
        )
        print(f"Found {len(scns)} scenes:", [scn["Name"] for scn in scns])

        user, passwd = rsgispy.tools.credentials.resolve_env_user_pass(
            None, None, "RSGIS_COP_USER", "RSGIS_COP_PASS", required=True,
            service_name="Copernicus Data Space",
        )
        token = cdse.get_access_token(user, passwd)
        dwnld_dir = os.path.join(out_dir, "downloads")
        cdse.download_scns(token, scns, dwnld_dir, use_pbar=use_pbar)

        for zip_file in sorted(glob.glob(os.path.join(dwnld_dir, "*.SAFE.zip"))):
            safe_dir = zip_file[: -len(".zip")]
            if not os.path.isdir(safe_dir):
                rsgispy.tools.archives.unzip_file(zip_file, dwnld_dir,
                                                  gen_arch_dir=False)
            tile = os.path.basename(safe_dir).split("_")[5]  # e.g. 'T30UVD'
            scenes.setdefault(tile, []).append(safe_dir)
    # [end-step-1-download]
    out["scenes"] = scenes

    # [step-2-stack]
    import rsgispy.imagecalibration.sensorlvl2data as l2

    sref_dir = os.path.join(out_dir, "sref")
    sref_imgs: Dict[str, List[str]] = {}
    for tile, safe_dirs in scenes.items():
        sref_imgs[tile] = []
        for safe_dir in safe_dirs:
            l2.create_stacked_sen2_esa_l2a_img(
                safe_dir,
                sref_dir,
                tmp_dir,
                out_res_10m=True,  # sharpen the 20 m bands to 10 m
                scale_factor=10000,  # reflectance x 10000, no data = 0
                inc_cloudsen12_img=use_cloudsen12,  # 12-band CloudSEN12 input
                use_pbar=use_pbar,
            )
            scn_name = os.path.basename(os.path.normpath(safe_dir))
            sref_imgs[tile].append(
                os.path.join(sref_dir, scn_name.replace(".SAFE", "_10m_sref.tif"))
            )
    print(sref_imgs)
    # [end-step-2-stack]
    out["sref_imgs"] = sref_imgs

    # [step-3-cloud-mask]
    import rsgispy.imageutils.masking
    import rsgispy.imagecalibration.cloudmasking as cldmsk
    from rsgispy.imagecalc.maths import BandDefn, calc_prop_true_exp

    masked_imgs: Dict[str, List[str]] = {}
    clear_props: Dict[str, float] = {}
    for tile, imgs in sref_imgs.items():
        masked_imgs[tile] = []
        for sref_img in imgs:
            if use_cloudsen12:
                cld_img = sref_img.replace("_sref.tif", "_cloudsen12.tif")
                cldmsk.apply_cloudsen12_s2_sref_cloud_mask(
                    sref_img.replace("_sref.tif", "_cloudsen12_sref.tif"),
                    cld_img,
                    scale_factor=10000,
                    add_offset=0,  # BOA offsets were applied when stacking
                    device=device,  # 'cpu', 'cuda', 'mps' or 'auto'
                    use_pbar=use_pbar,
                )
                cld_vals = [2, 3, 4]  # thick cloud, thin cloud, cloud shadow
            else:
                cld_img = sref_img.replace("_sref.tif", "_clouds.tif")
                cld_vals = [1, 2]  # SCL-derived cloud, cloud shadow
            masked_img = sref_img.replace("_sref.tif", "_sref_clrsky.tif")
            rsgispy.imageutils.masking.mask_img(
                sref_img, cld_img, masked_img, out_value=0, mask_value=cld_vals,
                no_data_val=0, use_pbar=use_pbar,
            )
            masked_imgs[tile].append(masked_img)
            # Proportion of the image that is clear-sky (valid) data
            clear_props[masked_img] = calc_prop_true_exp(
                "b1 > 0 ? 1 : 0", [BandDefn("b1", masked_img, 1)],
                use_pbar=use_pbar,
            )
    print({os.path.basename(k): round(v, 3) for k, v in clear_props.items()})
    # [end-step-3-cloud-mask]
    out["masked_imgs"] = masked_imgs
    out["clear_props"] = clear_props

    # [step-4-coregister]
    import rsgispy
    import rsgispy.imageregistration as imreg
    import rsgispy.imageregistration.warping as imreg_warp
    from rsgispy.imagecalc.comparison import calc_img_similarity_metrics

    coreg_imgs: Dict[str, List[str]] = {}
    coreg_info: Dict[str, Dict[str, Any]] = {}
    for tile, imgs in masked_imgs.items():
        ref_img = max(imgs, key=lambda img: clear_props[img])  # clearest scene
        coreg_imgs[tile] = [ref_img]
        for flt_img in imgs:
            if flt_img == ref_img:
                continue
            reg_img = flt_img.replace("_clrsky.tif", "_clrsky_coreg.tif")
            try:
                info = imreg_warp.coregister_images(
                    ref_img,
                    flt_img,
                    reg_img,
                    tie_method=imreg.TIE_METHOD_TEMPLATE_MATCH,
                    outlier_method=imreg.OUTLIER_METHOD_RANSAC,
                    transform_type=imreg.TRANSFORM_SHIFT,  # same orbit/tile
                    ref_band=coreg_band,  # NIR (band 7 of the 10-band stack)
                    flt_band=coreg_band,
                    pixel_gap=coreg_pixel_gap,  # spacing of candidate points
                    win_size=coreg_win_size,  # matching window (pixels)
                    search_area=5,  # largest expected shift (pixels)
                    residual_thres=5.0,  # max tie point residual (metres)
                    interp_method=rsgispy.INTERP_CUBIC,
                    no_data_val=0,
                    rnd_seed=42,
                    out_gcp_file=reg_img.replace(".tif", "_tiepts.gpkg"),
                    use_pbar=use_pbar,
                )
            except rsgispy.RSGISPyException as exc:  # e.g. too cloudy to match
                print(f"Not co-registered ({exc}): {flt_img}")
                coreg_imgs[tile].append(flt_img)
                continue
            for name, img in (("before", flt_img), ("after", reg_img)):
                info[f"corr_{name}"] = calc_img_similarity_metrics(
                    ref_img, img, metrics=[rsgispy.METRIC_CORRELATION],
                    img_a_bands=[coreg_band], img_b_bands=[coreg_band],
                )["correlation"]
            print(os.path.basename(flt_img), info)
            coreg_info[flt_img] = info
            coreg_imgs[tile].append(reg_img)
    # [end-step-4-coregister]
    out["coreg_info"] = coreg_info

    # [step-5-indices]
    from rsgispy.imagecalc.calcindices import calc_indices

    s2_band_names = {1: "blue", 2: "green", 3: "red", 4: "re1", 5: "re2",
                     6: "re3", 7: "nir", 8: "nir8a", 9: "swir1", 10: "swir2"}
    ndvi_imgs: Dict[str, List[str]] = {}
    for tile, imgs in coreg_imgs.items():
        ndvi_imgs[tile] = []
        for img in imgs:
            ndvi_img = img.replace(".tif", "_ndvi.tif")
            calc_indices(
                img,
                ndvi_img,
                s2_band_names,
                ["NDVI"],
                in_gain=10000,  # stored values are reflectance x 10000
                no_data_val=0,
                out_no_data_val=-999,
                use_pbar=use_pbar,
            )
            ndvi_imgs[tile].append(ndvi_img)
    # [end-step-5-indices]
    out["ndvi_imgs"] = ndvi_imgs

    # [step-6-composite]
    import rsgispy.imagecalc.maths
    from rsgispy.imagecalc.percentiles import (
        calc_img_band_pxl_percentiles,
        calc_imgs_pxl_percentiles,
    )
    from rsgispy.imageutils.metadata import get_band_names, set_band_names

    comp_dir = os.path.join(out_dir, "composites")
    os.makedirs(comp_dir, exist_ok=True)
    composites: Dict[str, Dict[str, str]] = {}
    for tile, imgs in coreg_imgs.items():
        med_img = os.path.join(comp_dir, f"{tile}_sref_median.tif")
        # One output band per input band: the per-pixel median of the dates
        calc_img_band_pxl_percentiles(imgs, med_img, [50], no_data_val=0,
                                      use_pbar=use_pbar)
        set_band_names(med_img, get_band_names(imgs[0]))  # 'Blue', 'Green', ...

        ndvi_pcts_img = os.path.join(comp_dir, f"{tile}_ndvi_p10_p50_p90.tif")
        # All the bands of all the inputs are pooled: one band per percentile
        calc_imgs_pxl_percentiles(ndvi_imgs[tile], ndvi_pcts_img, [10, 50, 90],
                                  no_data_val=-999, use_pbar=use_pbar)
        set_band_names(ndvi_pcts_img, ["NDVI_P10", "NDVI_P50", "NDVI_P90"])

        # Number of clear-sky observations contributing to each pixel
        n_obs_img = os.path.join(comp_dir, f"{tile}_n_clear_obs.tif")
        band_defs = [BandDefn(f"d{i}", img, 1) for i, img in enumerate(imgs)]
        exp = " + ".join(f"({bd.band_name} > 0)" for bd in band_defs)
        rsgispy.imagecalc.maths.band_maths(n_obs_img, exp, band_defs,
                                           datatype=rsgispy.TYPE_8UINT,
                                           use_pbar=use_pbar)
        composites[tile] = {"sref": med_img, "ndvi": ndvi_pcts_img,
                            "n_obs": n_obs_img}
    # [end-step-6-composite]
    out["composites"] = composites

    # [step-7-mosaic]
    import rsgispy.imageutils.mosaic
    import rsgispy.imageutils.vrt

    mosaics: Dict[str, str] = {}
    for product, no_data in (("sref", 0), ("ndvi", -999)):
        tile_imgs = [comp[product] for comp in composites.values()]
        vrt_file = os.path.join(out_dir, f"{product}_mosaic.vrt")
        rsgispy.imageutils.vrt.create_mosaic_images_vrt(
            tile_imgs, vrt_file, src_nodata=no_data, vrt_nodata=no_data
        )
        mosaic_img = os.path.join(out_dir, f"{product}_mosaic.tif")
        rsgispy.imageutils.mosaic.create_img_mosaic(
            tile_imgs,
            mosaic_img,
            background_val=no_data,
            skip_val=no_data,  # ignore no data pixels (in band 1)
            overlap_behaviour=0,  # overlaps: later images overwrite earlier
            gdalformat="GTIFF",  # tiled, DEFLATE-compressed, with statistics
            use_pbar=use_pbar,
        )
        mosaics[product] = mosaic_img
    # [end-step-7-mosaic]
    out["mosaics"] = mosaics

    # [step-8-check]
    import rsgispy.tools.checkdatasets as chkdata

    n_bands = {"sref": 10, "ndvi": 3}
    for product, mosaic_img in mosaics.items():
        file_ok, err_str = chkdata.check_gdal_image_file(
            mosaic_img,
            n_bands=n_bands[product],
            chk_proj=True,
            epsg_code=out_epsg or 0,  # expected EPSG code (0 = not checked)
            read_img=True,  # decode some random pixels
        )
        print(product, "OK" if file_ok else f"FAILED: {err_str}")
        out[f"{product}_ok"] = file_ok
    # [end-step-8-check]
    return out


def _parse_date(val: str) -> datetime.datetime:
    return datetime.datetime.strptime(val, "%Y-%m-%d")


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--safe", nargs="*", default=[],
                        help="Sentinel-2 L2A .SAFE directories")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="Search area (WGS84) to download scenes for")
    parser.add_argument("--dates", nargs=2, type=_parse_date, default=None,
                        metavar=("START", "END"), help="Search dates (YYYY-MM-DD)")
    parser.add_argument("--max-cloud", type=float, default=40.0,
                        help="Maximum scene cloud cover (%%) for the search")
    parser.add_argument("--cloudsen12", action="store_true",
                        help="Use the CloudSEN12 cloud masking (needs torch)")
    parser.add_argument("--device", default="cpu", help="CloudSEN12 device")
    parser.add_argument("--epsg", type=int, default=None,
                        help="Expected EPSG code of the outputs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    scenes: Dict[str, List[str]] = {}
    for safe_dir in args.safe:
        tile = os.path.basename(os.path.normpath(safe_dir)).split("_")[5]
        scenes.setdefault(tile, []).append(safe_dir)
    run_workflow(args.out_dir, scenes, bbox=args.bbox, dates=args.dates,
                 max_cloud=args.max_cloud, use_cloudsen12=args.cloudsen12,
                 device=args.device, out_epsg=args.epsg)


if __name__ == "__main__":
    main()
