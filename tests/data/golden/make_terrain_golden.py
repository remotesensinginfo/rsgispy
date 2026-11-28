"""
Generate the golden reference outputs of the terrain and illumination
functions (``rsgispy.elevation.terrain`` / ``rsgispy.elevation.illumination``)
used by ``tests/test_elevation_terrain_nodes.py``.

The outputs were generated with the block-loop implementation that preceded
the tile-graph wrappers, so the tests check that the wrappers reproduce them.
Do not regenerate them unless the intended output of a function changes.

Run from the repository root::

    python tests/data/golden/make_terrain_golden.py

It writes ``tests/data/golden/terrain_golden.npz``. Each output is checked to
be the same for several block sizes before it is stored.
"""

import json
import os
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import Affine

import rsgispy
import rsgispy.elevation.illumination as elev_illum
import rsgispy.elevation.terrain as elev_terr

OUT_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_golden.npz")

#: Block sizes each output is generated with (non-dividing, small, whole).
BLOCK_SIZES = (7, 16, 256)


def _hills(height, width, seed):
    """Smooth synthetic terrain with hills, a flat plateau and some noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dem = (
        120.0
        + 35.0 * np.sin(xx / 6.0) * np.cos(yy / 9.0)
        + 0.8 * xx
        - 0.5 * yy
        + rng.normal(0.0, 1.5, size=(height, width))
    )
    # Flat plateau (aspect NaN / flat handling).
    dem[5:11, 30:40] = 150.0
    return dem


def make_inputs():
    """Return ``{name: (array, meta)}`` of the input rasters."""
    inputs = {}
    height, width = 37, 53

    # A: float32, projected 10 m, nodata -9999 holes (single pixels, block,
    # image edge and corner).
    dem_a = _hills(height, width, 1).astype(np.float32)
    dem_a[0, 3:9] = -9999.0
    dem_a[18, 20] = -9999.0
    dem_a[25:29, 40:44] = -9999.0
    dem_a[height - 1, width - 1] = -9999.0
    dem_a[12, 0] = -9999.0
    inputs["dem_a"] = (
        dem_a,
        {"crs": "EPSG:32630", "transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
         "nodata": -9999.0},
    )

    # B: float64 without a header no-data value, NaN holes, non-square pixels.
    dem_b = _hills(height, width, 2)
    dem_b[7, 7] = np.nan
    dem_b[20:22, 0:3] = np.nan
    dem_b[36, 10] = np.nan
    inputs["dem_b"] = (
        dem_b,
        {"crs": "EPSG:27700", "transform": (30.0, 0.0, 400000.0, 0.0, -20.0, 300000.0),
         "nodata": None},
    )

    # C: int16 DEM, nodata -32768, with zeros (used with no_data_val=0).
    dem_c = np.round(_hills(height, width, 3)).astype(np.int16)
    dem_c[3:6, 3:6] = -32768
    dem_c[30, 44] = 0
    dem_c[31:33, 10:12] = 0
    inputs["dem_c"] = (
        dem_c,
        {"crs": "EPSG:32630", "transform": (5.0, 0.0, 500000.0, 0.0, -5.0, 5200000.0),
         "nodata": -32768.0},
    )

    # D: geographic DEM (EPSG:4326), elevation in metres, pixel size in
    # degrees, nodata -9999.
    dem_d = (_hills(height, width, 4) * 3.0).astype(np.float32)
    dem_d[10, 10] = -9999.0
    dem_d[0:2, 50:53] = -9999.0
    inputs["dem_d"] = (
        dem_d,
        {"crs": "EPSG:4326", "transform": (0.0003, 0.0, -3.5, 0.0, -0.0003, 53.2),
         "nodata": -9999.0},
    )

    # Pixel-resolution image for D: EW and NS resolution in metres per row,
    # with nodata (-1) and zero holes.
    lat = 53.2 - (np.arange(height) + 0.5) * 0.0003
    ew = 0.0003 * 111320.0 * np.cos(np.radians(lat))
    ns = np.full(height, 0.0003 * 110574.0)
    res_d = np.stack(
        [np.repeat(ew[:, None], width, axis=1), np.repeat(ns[:, None], width, axis=1)]
    ).astype(np.float32)
    res_d[0, 15, 15] = -1.0
    res_d[1, 22, 30:33] = -1.0
    res_d[0, 33, 5] = 0.0
    inputs["res_d"] = (
        res_d,
        {"crs": "EPSG:4326", "transform": (0.0003, 0.0, -3.5, 0.0, -0.0003, 53.2),
         "nodata": -1.0},
    )

    # Pixel-resolution image for A: varying resolution, no header nodata,
    # NaN holes.
    res_a = np.stack(
        [
            np.full((height, width), 10.0) + np.linspace(0.0, 2.0, width)[None, :],
            np.full((height, width), 12.0) + np.linspace(0.0, 1.0, height)[:, None],
        ]
    ).astype(np.float32)
    res_a[0, 9, 9] = np.nan
    res_a[1, 30, 50] = np.nan
    inputs["res_a"] = (
        res_a,
        {"crs": "EPSG:32630", "transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
         "nodata": None},
    )

    # Aspect image for categorise_aspect: values over the full range,
    # boundaries, > 360, NaN and nodata.
    rng = np.random.default_rng(5)
    asp = rng.uniform(0.0, 360.0, size=(height, width)).astype(np.float32)
    asp[0, :12] = [0.0, 44.999, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0,
                   359.99, 360.0, 400.0]
    asp[5, 5] = np.nan
    asp[6, 6:9] = -9999.0
    inputs["asp_e"] = (
        asp,
        {"crs": "EPSG:32630", "transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
         "nodata": -9999.0},
    )
    # The same aspect values without a header no-data value, with 45.0 marking
    # no data (used with no_data_val=45).
    asp_f = asp.copy()
    asp_f[asp_f == -9999.0] = 45.0
    inputs["asp_f"] = (
        asp_f,
        {"crs": "EPSG:32630", "transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
         "nodata": None},
    )
    return inputs


def make_cases():
    """Return the list of ``(case_id, func_name, args, kwargs)``."""
    t32 = rsgispy.TYPE_32FLOAT
    cases = []
    # dem_d is in EPSG:4326: the functions without a pixel-resolution image
    # reject it (pixel sizes in degrees), so it is only used with res_d.
    for dem in ("dem_a", "dem_b", "dem_c"):
        cases += [
            (f"slope_deg_{dem}", "slope", [dem], {"unit": "degrees"}),
            (f"slope_rad_{dem}", "slope", [dem], {"unit": "radians"}),
            (f"aspect_{dem}", "aspect", [dem], {}),
            (f"hillshade_{dem}", "hillshade", [dem], {"azimuth": 315.0, "zenith": 45.0}),
            (f"lia_{dem}", "local_incidence_angle", [dem],
             {"azimuth": 135.0, "zenith": 35.0}),
            (f"lea_{dem}", "local_exitance_angle", [dem],
             {"azimuth": 200.0, "zenith": 15.0}),
        ]
    cases += [
        # Data type and no-data variations.
        ("slope_deg_dem_a_int16", "slope", ["dem_a"],
         {"unit": "degrees", "datatype": rsgispy.TYPE_16INT}),
        ("slope_deg_dem_a_uint8", "slope", ["dem_a"],
         {"unit": "degrees", "datatype": rsgispy.TYPE_8UINT}),
        ("aspect_dem_b_uint16", "aspect", ["dem_b"], {"datatype": rsgispy.TYPE_16UINT}),
        ("aspect_dem_a_int16", "aspect", ["dem_a"], {"datatype": rsgispy.TYPE_16INT}),
        ("hillshade_dem_a_f32", "hillshade", ["dem_a"],
         {"azimuth": 90.0, "zenith": 60.0, "datatype": t32}),
        ("hillshade_dem_b_f32", "hillshade", ["dem_b"],
         {"azimuth": 0.0, "zenith": 0.0, "datatype": t32}),
        ("hillshade_dem_c_int16", "hillshade", ["dem_c"],
         {"azimuth": 360.0, "zenith": 90.0, "datatype": rsgispy.TYPE_16INT}),
        ("slope_deg_dem_c_nd0", "slope", ["dem_c"], {"unit": "degrees", "no_data_val": 0}),
        ("hillshade_dem_c_nd0", "hillshade", ["dem_c"],
         {"azimuth": 270.0, "zenith": 30.0, "no_data_val": 0}),
        ("lia_dem_a_f64", "local_incidence_angle", ["dem_a"],
         {"azimuth": 45.0, "zenith": 80.0, "datatype": rsgispy.TYPE_64FLOAT}),
        ("lea_dem_b_nd", "local_exitance_angle", ["dem_b"],
         {"azimuth": 10.0, "zenith": 5.0, "no_data_val": 150.0}),
        # Pixel-resolution image variants.
        ("slope_pxl_deg_dem_d", "slope_pxl_res_img", ["dem_d", "res_d"], {"unit": "degrees"}),
        ("slope_pxl_rad_dem_a", "slope_pxl_res_img", ["dem_a", "res_a"], {"unit": "radians"}),
        ("aspect_pxl_dem_d", "aspect_pxl_res_img", ["dem_d", "res_d"], {}),
        ("aspect_pxl_dem_a", "aspect_pxl_res_img", ["dem_a", "res_a"], {}),
        ("hillshade_pxl_dem_d", "hillshade_pxl_res_img", ["dem_d", "res_d"],
         {"azimuth": 315.0, "zenith": 45.0}),
        ("hillshade_pxl_dem_a_f32", "hillshade_pxl_res_img", ["dem_a", "res_a"],
         {"azimuth": 100.0, "zenith": 50.0, "datatype": t32}),
        ("slope_pxl_dem_d_int16", "slope_pxl_res_img", ["dem_d", "res_d"],
         {"unit": "degrees", "datatype": rsgispy.TYPE_16INT, "no_data_val": 0}),
        # categorise_aspect.
        ("cat_asp_e", "categorise_aspect", ["asp_e"], {}),
        ("cat_asp_e_int16", "categorise_aspect", ["asp_e"], {"datatype": rsgispy.TYPE_16INT}),
        ("cat_asp_f_nd", "categorise_aspect", ["asp_f"], {"no_data_val": 45.0}),
        ("cat_aspect_dem_a", "categorise_aspect", ["@aspect_dem_a"], {}),
        ("cat_aspect_dem_b", "categorise_aspect", ["@aspect_dem_b"], {}),
    ]
    return cases


FUNCS = {
    "slope": elev_terr.slope,
    "slope_pxl_res_img": elev_terr.slope_pxl_res_img,
    "aspect": elev_terr.aspect,
    "aspect_pxl_res_img": elev_terr.aspect_pxl_res_img,
    "categorise_aspect": elev_terr.categorise_aspect,
    "hillshade": elev_illum.hillshade,
    "hillshade_pxl_res_img": elev_illum.hillshade_pxl_res_img,
    "local_incidence_angle": elev_illum.local_incidence_angle,
    "local_exitance_angle": elev_illum.local_exitance_angle,
}


def write_input(path, arr, meta):
    """Write an input array (2D or 3D) as a GeoTIFF."""
    arr3 = arr[np.newaxis] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": arr3.shape[1],
        "width": arr3.shape[2],
        "count": arr3.shape[0],
        "dtype": str(arr3.dtype),
        "crs": meta["crs"],
        "transform": Affine(*meta["transform"]),
        "nodata": meta["nodata"],
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr3)
    return path


def run_case(func_name, in_paths, out_path, kwargs, calc_stats=True):
    """Run one golden case (positional input paths, then the output path)."""
    func = FUNCS[func_name]
    args = list(in_paths) + [out_path]
    if "unit" in kwargs:
        kwargs = dict(kwargs)
        args.append(kwargs.pop("unit"))
    for key in ("azimuth", "zenith"):
        if key in kwargs:
            kwargs = dict(kwargs)
            args.append(kwargs.pop(key))
    func(*args, gdalformat="GTIFF", calc_stats=calc_stats, use_pbar=False, **kwargs)


def read_output(path):
    """Return ``(array, dtype, nodata, band description, stats tags)``."""
    with rasterio.open(path) as ds:
        tags = {k: v for k, v in ds.tags(1).items() if k.startswith("STATISTICS_")}
        return ds.read(1), ds.dtypes[0], ds.nodata, ds.descriptions[0], tags


def stats_match(stats_a, stats_b, rel=1e-9):
    """
    ``True`` if two sets of statistics tags are equal; the floating-point
    statistics (whose last digits depend on the block size used to compute
    them) are compared to a relative tolerance ``rel``.
    """
    if set(stats_a) != set(stats_b):
        return False
    for key, val_a in stats_a.items():
        val_b = stats_b[key]
        if val_a == val_b:
            continue
        try:
            f_a, f_b = float(val_a), float(val_b)
        except ValueError:
            return False
        if abs(f_a - f_b) > rel * max(abs(f_a), abs(f_b), 1e-300):
            return False
    return True


def main():
    """Generate ``terrain_golden.npz``."""
    # pylint: disable=too-many-locals
    inputs = make_inputs()
    cases = make_cases()
    arrays = {}
    manifest = {"inputs": {}, "cases": []}
    with tempfile.TemporaryDirectory() as tmp:
        in_paths = {}
        for name, (arr, meta) in inputs.items():
            in_paths[name] = write_input(os.path.join(tmp, f"{name}.tif"), arr, meta)
            arrays[f"in_{name}"] = arr
            manifest["inputs"][name] = meta
        for case_id, func_name, ins, kwargs in cases:
            results = []
            for blk in BLOCK_SIZES:
                os.environ["RSGISPY_BLOCK_SIZE"] = str(blk)
                paths = []
                for name in ins:
                    if name.startswith("@"):
                        paths.append(os.path.join(tmp, f"{name[1:]}_{blk}.tif"))
                    else:
                        paths.append(in_paths[name])
                out_path = os.path.join(tmp, f"{case_id}_{blk}.tif")
                run_case(func_name, paths, out_path, kwargs)
                results.append(read_output(out_path))
            ref = results[0]
            for other in results[1:]:
                np.testing.assert_array_equal(other[0], ref[0])
                assert other[1:4] == ref[1:4], (case_id, other[1:4], ref[1:4])
                assert stats_match(other[4], ref[4]), (case_id, other[4], ref[4])
            arrays[f"out_{case_id}"] = ref[0]
            manifest["cases"].append(
                {
                    "id": case_id,
                    "func": func_name,
                    "inputs": ins,
                    "kwargs": kwargs,
                    "dtype": ref[1],
                    "nodata": None if ref[2] is None else repr(float(ref[2])),
                    "description": ref[3],
                    "stats": ref[4],
                }
            )
        os.environ.pop("RSGISPY_BLOCK_SIZE", None)
    arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(OUT_NPZ, **arrays)
    print(f"Wrote {len(manifest['cases'])} cases to {OUT_NPZ} "
          f"({os.path.getsize(OUT_NPZ) / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
