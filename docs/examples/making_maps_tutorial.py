"""
Run the map-making worked example (``making_maps.py``) on the Aberystwyth
data of the rsgispy tutorials
(https://github.com/remotesensinginfo/rsgispy-tutorials): the 20 m Sentinel-2 image
of 29 June 2018 (``data/sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif``),
the land cover training polygons (``data/cls_data/aber_sen2_cls_training.gpkg``),
the 10 m DEM (``data/dem_10m_osgb.tif``) and the country outlines
(``data/country.geojson``). The figures and printed outputs in
``docs/guides/making_maps.rst`` were made with this script::

    python making_maps_tutorial.py --data-dir rsgispy-tutorials/data \\
        --out-dir maps_tutorial
"""

import argparse
import os
from typing import Any, Dict

import making_maps


def run(data_dir: str, out_dir: str, use_pbar: bool = True,
        dpi: int = 100) -> Dict[str, Any]:
    """Prepare the tutorial inputs and run the map-making worked example."""
    # pylint: disable=too-many-locals,import-outside-toplevel
    # [tutorial-data]
    import geopandas as gpd
    import rsgispy
    import rsgispy.classification.classsklearn as csk
    import rsgispy.elevation.hydrology as hydro
    from rsgispy.classification.classdata import (
        ClassificationConfig, ClassInfoObj, ImageBandInfo,
        extract_class_training_samples, split_class_samples)
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    from rsgispy.imageutils.metadata import get_band_names
    from rsgispy.imageutils.warp import gdal_translate
    from shapely.geometry import box
    from sklearn.ensemble import RandomForestClassifier

    os.makedirs(out_dir, exist_ok=True)
    s2_scene = os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m.tif")
    vmsk_scene = os.path.join(data_dir, "sen2_20180629_t30uvd_orb037_osgb_stdsref_20m_vmsk.tif")

    # The map area: 24 x 18 km around Aberystwyth (British National Grid,
    # EPSG:27700) with the coast, the town, the Rheidol valley and the
    # forests to the east. Window (ulx, uly, lrx, lry).
    x_min, x_max, y_min, y_max = 256000, 280000, 270000, 288000
    s2_img = os.path.join(out_dir, "s2_aber_20m.tif")
    vmsk_img = os.path.join(out_dir, "s2_aber_20m_vmsk.tif")
    for in_img, sub_img in ((s2_scene, s2_img), (vmsk_scene, vmsk_img)):
        gdal_translate(in_img, sub_img, proj_win=(x_min, y_max, x_max, y_min),
                       use_pbar=use_pbar)

    # Land cover: the random forest classification of the tutorial notebook
    # 04_image_classification/03_supervised_pixel_classification (same
    # training polygons, classes, colours and classifier parameters), trained
    # on the whole scene and applied to the map area.
    classes = {  # name: (pixel value, training layer, colour)
        "Artificial surfaces": (1, "Artificial_Surfaces", (180, 180, 180)),
        "Bare rock and sand": (2, "Bare_Rock_Sand", (131, 250, 252)),
        "Conifer forest": (3, "Conifer_Forest", (0, 105, 16)),
        "Deciduous forest": (4, "Deciduous_Forest", (62, 168, 78)),
        "Long grass": (5, "Grass_Long", (247, 176, 32)),
        "Short grass": (6, "Grass_Short", (252, 231, 109)),
        "Non-photosynthetic veg.": (7, "NonPhotosynthetic_Vegetation", (156, 255, 215)),
        "Scrub": (8, "Scrub", (225, 114, 247)),
        "Water": (9, "Water_Training", (0, 0, 255)),
        "Bracken": (10, "Bracken", (140, 255, 0)),
    }
    train_vec = os.path.join(data_dir, "cls_data", "aber_sen2_cls_training.gpkg")
    band_names = get_band_names(s2_scene)
    cls_cfg = ClassificationConfig(
        [ImageBandInfo(s2_scene, name="sen2", bands=list(range(1, 11)),
                       band_names=band_names)],
        {name: ClassInfoObj(out_id=val, vec_file=train_vec, vec_lyr=lyr, colour=clr)
         for name, (val, lyr, clr) in classes.items()},
    )
    samples = extract_class_training_samples(cls_cfg, use_pbar=use_pbar)
    train_df, _, _ = split_class_samples(samples, n_train=350, n_valid=50,
                                         n_test=50, rnd_seed=42)
    rf_model, _ = csk.train_sklearn_classifier(
        cls_cfg, RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42),
        train_df, return_model=True,
    )
    cls_img = os.path.join(out_dir, "land_cover.tif")
    csk.apply_sklearn_classifier(
        cls_cfg, rf_model, vmsk_img, 1, output_img=cls_img,
        img_band_info=[ImageBandInfo(s2_img, name="sen2", bands=list(range(1, 11)),
                                     band_names=band_names)],
        use_pbar=use_pbar,
    )

    # The continuous map: NDVI, with water (from the land cover) as no data.
    ndvi_img = os.path.join(out_dir, "ndvi.tif")
    band_maths(ndvi_img, "cls == 9 ? -999 : (nir - red) / (nir + red)",
               [BandDefn("cls", cls_img, 1), BandDefn("nir", s2_img, 7),
                BandDefn("red", s2_img, 3)],
               datatype=rsgispy.TYPE_32FLOAT, no_data_val=-999, use_pbar=use_pbar)

    # The DEM, on the same area. The sea is 0 m in this DEM: set it to no data.
    dem_sub = os.path.join(out_dir, "dem_aber_10m_sea.tif")
    gdal_translate(os.path.join(data_dir, "dem_10m_osgb.tif"), dem_sub,
                   proj_win=(x_min, y_max, x_max, y_min), use_pbar=use_pbar)
    dem_img = os.path.join(out_dir, "dem_aber_10m.tif")
    band_maths(dem_img, "dem <= 0 ? -32768 : dem", [BandDefn("dem", dem_sub, 1)],
               datatype=rsgispy.TYPE_32FLOAT, no_data_val=-32768, use_pbar=use_pbar)

    # Rivers: channels draining more than 10 km2, from the whole DEM (so the
    # catchments are complete) at 20 m (see the hydrology guide).
    dem_20m = os.path.join(out_dir, "dem_20m.tif")
    gdal_translate(os.path.join(data_dir, "dem_10m_osgb.tif"), dem_20m,
                   target_res=(20, 20), resample_alg=rsgispy.INTERP_AVERAGE,
                   use_pbar=use_pbar)
    dem_land = os.path.join(out_dir, "dem_20m_land.tif")
    band_maths(dem_land, "dem <= 0 ? -32768 : dem", [BandDefn("dem", dem_20m, 1)],
               datatype=rsgispy.TYPE_32FLOAT, no_data_val=-32768, use_pbar=use_pbar)
    hydro_imgs = {n: os.path.join(out_dir, f"{n}_20m.tif")
                  for n in ("cond_dem", "fdir", "facc", "streams")}
    hydro.condition_dem(dem_land, hydro_imgs["cond_dem"], use_pbar=use_pbar)
    hydro.flow_direction(hydro_imgs["cond_dem"], hydro_imgs["fdir"], use_pbar=use_pbar)
    hydro.flow_accumulation(hydro_imgs["fdir"], hydro_imgs["facc"], out_units="area",
                            use_pbar=use_pbar)
    hydro.extract_streams(hydro_imgs["facc"], hydro_imgs["streams"], threshold=10e6,
                          threshold_units="area", use_pbar=use_pbar)
    streams_vec = os.path.join(out_dir, "streams_20m.gpkg")
    hydro.vectorise_streams(hydro_imgs["streams"], hydro_imgs["fdir"], streams_vec,
                            use_pbar=use_pbar)
    map_area = box(x_min, y_min, x_max, y_max)
    rivers_vec = os.path.join(out_dir, "rivers.gpkg")
    gpd.clip(gpd.read_file(streams_vec), map_area).to_file(rivers_vec)

    # Reporting regions: 2 x 2 km squares, named by row (A-I, from the north)
    # and column (1-12, from the west).
    regions_vec = os.path.join(out_dir, "squares_2km.gpkg")
    squares = [(f"{row}{col}", box(x, y - 2000, x + 2000, y))
               for row, y in zip("ABCDEFGHI", range(y_max, y_min, -2000))
               for col, x in enumerate(range(x_min, x_max, 2000), start=1)]
    gpd.GeoDataFrame({"name": [name for name, _ in squares]},
                     geometry=[geom for _, geom in squares],
                     crs="EPSG:27700").to_file(regions_vec)

    out = making_maps.run_workflow(
        s2_img,
        cls_img,
        ndvi_img,
        regions_vec,
        rivers_vec,
        os.path.join(data_dir, "country.geojson"),  # the inset map
        out_dir,
        {val: name for name, (val, _, _) in classes.items()},
        rgb_bands=(3, 2, 1),  # Red, Green, Blue
        user_stretch=(0, 150),  # this image is reflectance x 1000
        score_label="NDVI",
        forest_vals=(3, 4),  # conifer and deciduous forest
        dpi=dpi,
        dem_img=dem_img,
    )
    # [end-tutorial-data]
    # The inputs, for the plot gallery (plot_gallery_tutorial.py).
    out.update(s2_img=s2_img, cls_img=cls_img, ndvi_img=ndvi_img, dem_img=dem_img,
               rivers_vec=rivers_vec,
               classes_info={val: name for name, (val, _, _) in classes.items()})
    return out


def main() -> None:
    """Command line interface."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")  # write files only; no window needed
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", required=True,
                        help="The data folder of the rsgispy-tutorials repository")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--dpi", type=int, default=100, help="Figure resolution")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
