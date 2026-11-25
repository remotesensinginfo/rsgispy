"""
Generate the RSGISLib to rsgispy migration tables for
``docs/guides/migrating_from_rsgislib.rst``.

The RSGISLib names below are curated (from the rsgispy docstrings and the
AI skill references, and from RSGISLib modules whose functions rsgispy keeps
under the same name). The rsgispy side is found and checked from the code:
for functions kept under the same name, the rsgispy module is located by
searching the rsgispy package, and the notes about keyword-only output
options and GeoDataFrame inputs are read from the function signatures. A
name that no longer exists in rsgispy stops the generator with an error.

Update the tables in the guide (between the ``.. begin-generated-tables`` and
``.. end-generated-tables`` comments) with::

    python docs/examples/gen_rsgislib_migration_table.py --write

and check that they are up to date with ``--check`` (also done by
``tests/test_migrating_from_rsgislib_guide.py``).
"""

import argparse
import importlib
import inspect
import os
import pkgutil
import sys
from types import ModuleType
from typing import Dict, List, Sequence, Tuple

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
REPO_DIR = os.path.normpath(os.path.join(DOCS_DIR, os.pardir))
GUIDE = os.path.join(DOCS_DIR, "guides", "migrating_from_rsgislib.rst")
BEGIN = ".. begin-generated-tables"
END = ".. end-generated-tables"

#: (RSGISLib module, rsgispy module(s), note)
MODULES: Sequence[Tuple[str, Sequence[str], str]] = (
    ("rsgislib", ("rsgispy",),
     "``TYPE_*``, ``INTERP_*``, ``SUMTYPE_*`` and other constants."),
    ("rsgislib.imagecalc", ("rsgispy.imagecalc",),
     "Split into submodules (``maths``, ``stats``, ``percentiles``, "
     "``distance``, ...)."),
    ("rsgislib.imagecalc.calcindices", ("rsgispy.imagecalc.calcindices",),
     "One function for all indices."),
    ("rsgislib.imageutils", ("rsgispy.imageutils",),
     "Split into submodules (``metadata``, ``bands``, ``blank``, ``masking``, "
     "``warp``, ...)."),
    ("rsgislib.vectorutils", ("rsgispy.vector", "rsgispy.vectorprocess"),
     "``rsgispy.vector.{io,create,geoms,attrs,stats,selection}``."),
    ("rsgislib.vectorattrs", ("rsgispy.vector.attrs",), ""),
    ("rsgislib.vectorgeoms", ("rsgispy.vector.geoms",), ""),
    ("rsgislib.zonalstats", ("rsgispy.zonalstats",),
     "``polygon``, ``point`` and the new ``labels`` submodule."),
    ("rsgislib.rastergis", ("rsgispy.zonalstats.labels",),
     "Not ported; label raster statistics written to tables instead."),
    ("rsgislib.segmentation", ("rsgispy.segmentation",),
     "One submodule per algorithm."),
    ("rsgislib.classification", ("rsgispy.classification", "rsgispy.ml"),
     "Configuration objects; shared ML tools in ``rsgispy.ml``."),
    ("rsgislib.elevation", ("rsgispy.elevation",),
     "``terrain``, ``illumination``, ``filtering``, ``hydrology``, ..."),
    ("rsgislib.imagefilter", ("rsgispy.imagefilter",),
     "``stats``, ``smooth``, ``edge``, ``sar``, ``texture``, ``chained``."),
    ("rsgislib.imagemorphology", ("rsgispy.imagemorphology",), ""),
    ("rsgislib.imagecalibration", ("rsgispy.imagecalibration",), ""),
    ("rsgislib.imageregistration", ("rsgispy.imageregistration",),
     "``shift``, ``tiepoints``, ``warping``."),
    ("rsgislib.tools", ("rsgispy.tools",),
     "e.g. ``tools.filetools`` is ``rsgispy.tools.files``; ``tools.utils`` is "
     "split (``jsonio``, ``strings``, ``dicts``, ...)."),
)

#: RSGISLib functions and classes kept under the same name in rsgispy:
#: RSGISLib module -> (rsgispy package searched, names).
SAME_NAME: Sequence[Tuple[str, str, Sequence[str]]] = (
    ("rsgislib.imagecalc", "rsgispy.imagecalc",
     ("BandDefn", "image_math", "image_band_math", "calc_prop_true_exp",
      "calc_dist_to_img_vals", "count_pxls_of_val", "get_unique_values")),
    ("rsgislib.imageutils", "rsgispy.imageutils",
     ("create_copy_img", "set_band_names", "get_band_names",
      "get_img_res", "get_img_size", "get_img_band_count",
      "get_img_no_data_value", "set_img_no_data_value", "get_img_bbox",
      "get_epsg_proj_from_img", "get_rsgislib_datatype_from_img",
      "resample_img_to_match", "reproject_image", "mask_img", "gen_valid_mask",
      "subset_bbox", "subset_to_img", "create_img_mosaic", "include_imgs",
      "create_stack_images_vrt", "define_colour_table", "set_img_thematic",
      "set_env_vars_lzw_gtiff_outs", "set_env_vars_deflate_gtiff_outs")),
    ("rsgislib.vectorutils", "rsgispy.vector",
     ("get_vec_feat_count", "get_vec_lyr_cols", "get_vec_lyrs_lst",
      "get_epsg_proj_from_vec", "reproj_vec_lyr", "merge_vec_lyrs",
      "vector_translate", "delete_vec_file", "spatial_select",
      "create_train_test_smpls")),
    ("rsgislib.vectorutils.createvectors", "rsgispy.vector",
     ("polygonise_raster_to_vec_lyr",)),
    ("rsgislib.vectorattrs", "rsgispy.vector",
     ("write_vec_column", "read_vec_column", "read_vec_columns",
      "add_fid_col")),
    ("rsgislib.vectorgeoms", "rsgispy.vector",
     ("buffer_vec_layer_gp", "explode_vec_lyr", "delete_polygon_holes")),
    ("rsgislib.zonalstats", "rsgispy.zonalstats",
     ("calc_zonal_band_stats_file", "calc_zonal_band_stats_test_poly_pts_file",
      "ext_point_band_values_file")),
    ("rsgislib.elevation", "rsgispy.elevation",
     ("slope", "aspect", "hillshade", "shadow_mask", "local_incidence_angle",
      "local_exitance_angle", "fill_dem_soille_gratin_1994",
      "dtm_aspect_median_filter")),
    ("rsgislib.segmentation", "rsgispy.segmentation",
     ("clump", "union_of_clumps")),
    ("rsgislib.imagefilter", "rsgispy.imagefilter",
     ("apply_median_filter", "apply_mean_filter")),
    ("rsgislib.imagemorphology", "rsgispy.imagemorphology",
     ("create_circular_op", "image_dilate", "image_erode", "image_opening",
      "image_closing")),
    ("rsgislib.classification.clustersklearn", "rsgispy.classification",
     ("img_pixel_cluster", "img_pixel_sample_cluster")),
    ("rsgislib.imageregistration", "rsgispy.imageregistration",
     ("find_image_offset",)),
    ("rsgislib.tools.filetools", "rsgispy.tools",
     ("get_file_basename", "find_file", "delete_file_with_basename")),
    ("rsgislib.tools.utils", "rsgispy.tools",
     ("read_json_to_dict", "write_dict_to_json", "uid_generator")),
)

#: RSGISLib names that were renamed or moved, or whose arguments changed:
#: (RSGISLib name, rsgispy name, what changed).
CHANGED: Sequence[Tuple[str, str, str]] = (
    ("rsgislib.imagecalc.band_math", "rsgispy.imagecalc.maths.band_maths",
     "Renamed (``maths``). ``band_defs`` is the third positional argument; "
     "``gdalformat`` and ``datatype`` are keyword-only."),
    ("rsgislib.imagecalc.bandMath (RSGISLib 4)",
     "rsgispy.imagecalc.maths.band_maths",
     "As above; no camelCase names in rsgispy."),
    ("rsgislib.imagecalc.imageMath (RSGISLib 4)",
     "rsgispy.imagecalc.maths.image_math", "As above."),
    ("rsgislib.imagecalc.calcindices.calc_ndvi (and the other per-index "
     "functions)", "rsgispy.imagecalc.calcindices.calc_indices",
     "One function: ``indices=['NDVI', ...]`` and ``band_names`` mapping "
     "band index to band name (e.g. ``{3: 'red', 4: 'nir'}``)."),
    ("rsgislib.segmentation.shepherdseg.run_shepherd_segmentation",
     "rsgispy.segmentation.shepherd.shepherd_segmentation",
     "No temporary directory or KEA intermediate files; check the argument "
     "names against the RSGISLib docs."),
    ("rsgislib.vectorutils.createrasters.rasterise_vec_lyr",
     "rsgispy.imageutils.rasterise.rasterise_vec_lyr",
     "Moved to ``imageutils``; output options keyword-only."),
    ("rsgislib.rastergis.populate_rat_with_stats",
     "rsgispy.zonalstats.labels.calc_label_zonal_stats",
     "Statistics go to a table (one row per label) instead of a raster "
     "attribute table; join to polygons with "
     ":func:`rsgispy.zonalstats.labels.join_label_stats_to_vec`."),
    ("rsgislib.imageutils.pop_img_stats",
     "rsgispy.imageutils.stats.pop_img_stats",
     "No ``use_no_data`` flag: ``no_data_val`` is the second argument "
     "(``None`` uses the image's no data value). Pass it by keyword."),
    ("rsgislib.imageutils.stack_img_bands",
     "rsgispy.imageutils.bands.stack_img_bands",
     "rsgispy order is ``(input_imgs, output_img, band_names, skip_value)``; "
     "compare with your RSGISLib calls (check the RSGISLib docs) and use "
     "keyword arguments."),
    ("rsgislib.imageutils.create_blank_img",
     "rsgispy.imageutils.blank.create_blank_img",
     "Projection given as ``epsg_code`` instead of WKT, after a new fill "
     "value argument ``pxl_val`` (check the RSGISLib docs for the arguments "
     "it replaces); use keyword arguments."),
    ("rsgislib.classification.classsklearn.train_sklearn_classifier",
     "rsgispy.classification.classsklearn.train_sklearn_classifier",
     "Same name; takes a "
     ":class:`rsgispy.classification.classdata.ClassificationConfig` and "
     "sample tables, so the arguments differ."),
    ("rsgislib.classification.classsklearn.apply_sklearn_classifier",
     "rsgispy.classification.classsklearn.apply_sklearn_classifier",
     "As above."),
)

TABLE_HEADER = """.. list-table::
   :header-rows: 1
   :widths: {widths}
"""


def _public_modules(pkg_name: str) -> List[ModuleType]:
    """Import ``pkg_name`` and all its public submodules."""
    pkg = importlib.import_module(pkg_name)
    mods = [pkg]
    if hasattr(pkg, "__path__"):
        for info in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
            parts = info.name.split(".")
            if any(part.startswith("_") for part in parts) or ".cli" in info.name:
                continue
            mods.append(importlib.import_module(info.name))
    return mods


def resolve(dotted: str) -> object:
    """Return the object for an rsgispy dotted name (module or attribute)."""
    try:
        return importlib.import_module(dotted)
    except ImportError:
        mod_name, attr = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(mod_name), attr)


def locate(name: str, pkg_name: str) -> str:
    """
    Find the public rsgispy module under ``pkg_name`` that provides the
    function or class ``name`` and return its full dotted name. Package
    re-exports are dropped in favour of the submodule that defines the name.
    """
    cands = []
    for mod in _public_modules(pkg_name):
        obj = getattr(mod, name, None)
        if obj is not None and (inspect.isfunction(obj) or inspect.isclass(obj)):
            cands.append(mod.__name__)
    # Prefer modules that export the name in __all__ (not just import it).
    exported = [c for c in cands
                if name in getattr(sys.modules[c], "__all__", ())]
    if exported:
        cands = exported
    # Drop packages that re-export a name from one of their submodules.
    cands = [c for c in cands if not any(o.startswith(c + ".") for o in cands)]
    if len(cands) != 1:
        raise LookupError(f"'{name}' in '{pkg_name}': found {cands}")
    return f"{cands[0]}.{name}"


def role(dotted: str) -> str:
    """Return an RST cross-reference for an rsgispy dotted name."""
    obj = resolve(dotted)
    if inspect.ismodule(obj):
        kind = "mod"
    elif inspect.isclass(obj):
        kind = "class"
    else:
        kind = "func"
    return f":{kind}:`{dotted}`"


def signature_notes(dotted: str) -> str:
    """Notes derived from the rsgispy signature."""
    obj = resolve(dotted)
    if inspect.isclass(obj):
        return ""
    params = inspect.signature(obj).parameters
    notes = []
    kw_only = [p for p in ("gdalformat", "datatype", "out_vec_file")
               if p in params and params[p].kind is inspect.Parameter.KEYWORD_ONLY]
    if kw_only:
        notes.append(", ".join(f"``{p}``" for p in kw_only) + " keyword-only")
    if any("GeoDataFrame" in str(p.annotation) for p in params.values()):
        notes.append("accepts a GeoDataFrame")
    elif "GeoDataFrame" in str(inspect.signature(obj).return_annotation):
        notes.append("can return a GeoDataFrame")
    return "; ".join(notes)


def _row(cells: Sequence[str]) -> str:
    lines = []
    for idx, cell in enumerate(cells):
        prefix = "   * - " if idx == 0 else "     - "
        lines.append(prefix + (cell if cell else ""))
    return "\n".join(line.rstrip() for line in lines)


def module_table() -> str:
    """RST table mapping RSGISLib modules to rsgispy modules."""
    rows = [_row(["RSGISLib module", "rsgispy module(s)", "Notes"])]
    for rsgislib_mod, rsgispy_mods, note in MODULES:
        rsgis = ", ".join(role(m) for m in rsgispy_mods)
        rows.append(_row([f"``{rsgislib_mod}``", rsgis, note]))
    return TABLE_HEADER.format(widths="30 35 35") + "\n" + "\n".join(rows)


def same_name_table() -> str:
    """RST table of names kept the same, with their new rsgispy module."""
    rows = [_row(["RSGISLib", "rsgispy", "Notes (from the signature)"])]
    for rsgislib_mod, pkg_name, names in SAME_NAME:
        for name in names:
            dotted = locate(name, pkg_name)
            rows.append(_row([f"``{rsgislib_mod}.{name}``", role(dotted),
                              signature_notes(dotted)]))
    return TABLE_HEADER.format(widths="38 42 20") + "\n" + "\n".join(rows)


def changed_table() -> str:
    """RST table of renamed functions and changed arguments."""
    rows = [_row(["RSGISLib", "rsgispy", "What changed"])]
    for rsgislib_name, dotted, note in CHANGED:
        rows.append(_row([f"``{rsgislib_name}``", role(dotted), note]))
    return TABLE_HEADER.format(widths="32 33 35") + "\n" + "\n".join(rows)


def build_tables() -> Dict[str, str]:
    """Build all generated tables, keyed by section label."""
    return {
        "modules": module_table(),
        "changed": changed_table(),
        "same-name": same_name_table(),
    }


def generated_block() -> str:
    """The full generated RST block placed between the guide markers."""
    tables = build_tables()
    parts = [BEGIN, ""]
    parts += [
        ".. _rsgislib-module-table:", "",
        "Module mapping", "~~~~~~~~~~~~~~", "",
        tables["modules"], "",
        ".. _rsgislib-changed-table:", "",
        "Renamed functions and changed arguments",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
        tables["changed"], "",
        ".. _rsgislib-same-name-table:", "",
        "Functions with the same name in a new module",
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
        tables["same-name"], "",
        END,
    ]
    return "\n".join(parts)


def replace_block(text: str, block: str) -> str:
    """Replace the generated block in ``text``."""
    start = text.index(BEGIN)
    end = text.index(END) + len(END)
    return text[:start] + block + text[end:]


def main() -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true",
                       help="Rewrite the tables in the guide")
    group.add_argument("--check", action="store_true",
                       help="Exit with 1 if the guide's tables are out of date")
    args = parser.parse_args()
    # Document the rsgispy in this checkout rather than an installed copy.
    sys.path.insert(0, REPO_DIR)
    block = generated_block()
    if not (args.write or args.check):
        print(block)
        return 0
    with open(GUIDE, encoding="utf-8") as f_obj:
        text = f_obj.read()
    new_text = replace_block(text, block)
    if args.check:
        if new_text != text:
            print("The migration tables are out of date; run with --write.")
            return 1
        return 0
    with open(GUIDE, "w", encoding="utf-8") as f_obj:
        f_obj.write(new_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
