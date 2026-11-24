# Raster Processing, Band Maths, Spectral Indices & Image Utilities

Covers top-level `BBox` & `imageprocess` (including `TileGraph`), `imagecalc.*`, and all `imageutils.*` submodules in `rsgispy` v0.3.1.

---

## 1. Typed Bounding Box (`rsgispy.BBox`) & `rsgispy.imageprocess`

### `rsgispy.BBox`
Immutable `tuple` subclass storing `(west, east, south, north)` (RSGIS order). Unpacks to `(xmin, xmax, ymin, ymax)` and carries an optional `.epsg` integer (`None` if unset).
- Properties: `.west`, `.east`, `.south`, `.north`, `.xmin`, `.xmax`, `.ymin`, `.ymax`, `.epsg`, `.width`, `.height`, `.area`, `.centre` -> `(x, y)`
- Class constructors:
  - `BBox.from_rsgis(bbox, epsg=None)` — from `(xmin, xmax, ymin, ymax)`
  - `BBox.from_geo(bounds, epsg=None)` — from `(minx, miny, maxx, maxy)` (Shapely/Rasterio order)
  - `BBox.from_dict(d)`
- Methods: `.as_rsgis()`, `.as_geo()`, `.to_polygon()` (Shapely Polygon), `.with_epsg(epsg)`, `.buffer(dist)`, `.reproject(out_epsg)`, `.intersects(other)`, `.intersection(other)`, `.union(other)`, `.contains_point(x, y)`, `.equals(other)`, `.to_dict()`.

### `rsgispy.imageprocess` (Shared Low-Level Raster Core)
- **Block size**: `DEFAULT_BLOCK_SIZE = 512`, `ENV_VAR_BLOCK_SIZE = 'RSGISPY_BLOCK_SIZE'`, `ENV_VAR_RSGISLIB_BLOCK_SIZE = 'RSGISLIB_BLOCK_SIZE'`, `get_default_block_size()`, `set_default_block_size(block_size)`, `reset_default_block_size()`.
- **Validation & masks**: `ensure_file_exists(file_path, label='Input image')`, `validate_band_index(img_band, n_bands, img_path)`, `validate_odd_window_size(size, min_size=3, name='filter_size')`, `build_valid_mask(arr, no_data_val=None, msk_arr=None, valid_msk_val=1)`, `build_valid_cube_mask(arr_nd, no_data_val=None, msk_arr=None, valid_msk_val=1, axis=0)`.
- **Dtype & nodata**: `resolve_out_dtype(datatype, default_dtype=9, ref_np_dtype=None, require_int=False)` -> `(rsgis_dtype, rio_dtype_str, np_dtype)`, `fits_dtype_range(value, out_np_dtype, allow_round_int=False)`, `resolve_out_nodata(out_np_dtype, out_no_data_val=None, fallback_nodata=None, *, default_float=-9999.0, default_int=0.0, clamp_int=False, unsigned_clamp='max', strict_explicit=False, allow_round_int=False, allow_nonfinite_float=False)`, `resolve_nodata_and_fill(eff_nd, out_np_dtype)`.
- **Streaming & stats**: `normalise_creation_options(options)`, `get_gdal_img_creation_opts(gdalformat)`, `parse_epsg_crs(epsg_code)`, `maybe_pop_stats(output_img, calc_stats, no_data_val=None, thematic=False, use_pbar=True, ignore_cog=False, gdalformat=None)`, `OnlineStatsAccumulator(n_bands=1)` (`.update_band(band_idx, vals)`, `.update_matrix(flat_valid)`, `.stds(ddof=0)`), `update_chan_moments(...)`, `compute_dataset_band_stats(ds, bidx, nodata=None)`.
- **Dataset grids & windows**: `IntersectionGrid`, `bbox_from_dataset(src)`, `compute_spatial_intersection(datasets)`, `validate_crs_and_res(datasets, first_ds=None)`, `open_input_datasets(stack, norm_defs)`, `open_multi_image_datasets(stack, input_imgs)`, `open_output_dataset(...)`, `create_output_from_dataset(...)`, `iter_dataset_windows(height, width, block_size=None, use_pbar=True, block_width=None)`, `iter_image_blocks(...)`, `iter_single_image_band_blocks(...)`, `process_blocks(...)`, `run_window_halo_blocks(...)`, `read_window_with_halo(...)`, `read_raster_band(input_img, img_band=1, no_data_val=None)` -> `(arr, crs, transform, nodata)`.

### Declarative Tile DAG (`rsgispy.imageprocess.TileGraph`)
Compose multi-step out-of-core raster pipelines with automatic halo propagation and zero intermediate disk writes:
- Sources: `RasterSource(path, bands=None, dtype=np.float64, border=HaloBorder(mode='edge', fill_value=0.0), nodata=None, name=None)`, `ArraySource(array, transform=None, crs=None, border=None, nodata=None, dtype=None, name=None)`.
- Nodes: `PointwiseOp(inputs, func, pass_ctx=False, name=None)`, `WindowOp(source, radius, func, border=None, pass_ctx=False, name=None)`, `BandMathOp(inputs, expr, nodata=None, name=None)`.
- Sinks & execution: `DiskSink(source, output_img, gdalformat='GTIFF', datatype=9, nodata=None, clamp_int=True, creation_opts=None, band_names=None, calc_stats=False, name=None)`, `MemorySink(source, dtype=None, clamp_int=True, name=None)`, `TileGraph(sinks, block_size=256).run(use_pbar=True)` or `run_tile_graph(sinks, block_size=256, use_pbar=True)` -> `Dict[str, np.ndarray]`.

```python
import rsgispy
from rsgispy.imageprocess import (
    RasterSource, HaloBorder, BandMathOp, WindowOp, DiskSink, run_tile_graph,
)
import numpy as np

src = RasterSource("s2.tif", bands=[3, 4], border=HaloBorder("edge"))
ndvi = BandMathOp({"red": src, "nir": src}, expr="(nir[1] - red[0]) / (nir[1] + red[0] + 1e-6)")
smooth = WindowOp(ndvi, radius=1, func=lambda arr: np.nanmean(arr, axis=(-2, -1)))  # custom halo kernel
run_tile_graph(DiskSink(smooth, "ndvi_smooth.tif", datatype=rsgispy.TYPE_32FLOAT, calc_stats=True))
```

---

## 2. `rsgispy.imagecalc.maths` (Band Maths via `pymu-eval`)

### Classes & Functions
- `BandDefn(band_name=None, input_img=None, img_band=None)` (`.to_dict()`; note `img_band` is **1-based**)
- `band_maths(output_img, exp, band_defs, exp_band_name=False, output_exists=False, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=True, creation_opts=None, band_names=None)`
- `image_band_math(input_img, output_img, exp, exp_band_name=False, output_exists=False, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=True, creation_opts=None, band_names=None)` — binds bands as `b1, b2, ...` automatically from `input_img`.
- `image_math(input_img, output_img, exp, exp_band_name=False, output_exists=False, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, use_pbar=True, creation_opts=None)` — single expression applied per band (use `b` or band variable).
- `calc_prop_true_exp(exp, band_defs, in_vld_img=None, use_pbar=True) -> float` — returns fraction (`0.0–1.0`) of valid pixels where `exp` evaluates truthy.

### Multi-Output Expression Rules (`pymu-eval`)
- Separate output expressions with **top-level commas** (commas inside `min(...)`, `max(...)`, `avg(...)` stay function arguments).
- Assignments (`tmp = a + b`, or updating an existing variable) **emit an output band** AND remain in scope for subsequent expressions in the same call.
- Pass `band_names=[...]` matching the number of emitted expressions/outputs to name each band cleanly.
- Ternary conditionals use C/muParser syntax (`cond ? true_val : false_val`), logical operators `&&`, `||`, `!`, and standard math (`sqrt`, `abs`, `log`, `log10`, `exp`, `sin`, `cos`, `tan`, `pow`, `min`, `max`, `avg`, `sum`, `isnan`).

---

## 3. `rsgispy.imagecalc.calcindices` (97 Optical & 63 SAR Spectral Indices)

### Functions
- `calc_indices(input_img, output_img, band_names, indices, in_gain=1.0, in_offset=0.0, out_gain=1.0, out_offset=0.0, index_params=None, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, no_data_val=None, out_no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
  - Physical conversion: `physical = (dn - in_offset) / in_gain` (e.g., Sentinel-2 L2A `0–10000` -> reflectance `0–1` via `in_gain=10000.0`). `in_gain`/`in_offset` accept scalar, per-band sequence, or `{band_idx_or_name: val}` dict.
  - Output conversion: `out = (index * out_gain) + out_offset` (rounded/clamped when `datatype` is integer).
- `get_available_indices() -> List[str]`
- `get_index_req_bands(index_name) -> Tuple[str, ...]`
- `get_index_equation(index_name) -> str`
- `calc_sar_indices(input_img, output_img, band_names, indices, in_units, in_gain=1.0, in_offset=0.0, out_gain=1.0, out_offset=0.0, index_params=None, *, gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT, no_data_val=None, out_no_data_val=None, calc_stats=True, use_pbar=True, creation_opts=None)`
  - `in_units`: required `"dB"` or `"power"`. Automatically converts `power = 10 ** (dB / 10)` or `dB = 10 * log10(power)` per index.
- `get_available_sar_indices() -> List[str]`
- `get_sar_index_req_bands(index_name) -> Tuple[str, ...]`
- `get_sar_index_equation(index_name) -> str`
- `get_sar_index_units(index_name) -> str` (`'power'` or `'dB'`)

### Accepted Band Aliases
- **Optical** (`band_names={1: "blue", ...}`): `'blue'`, `'green'`, `'red'`, `'nir'`, `'nir_narrow'` (`nir-narrow`, `narrow_nir`, `nir8a`, `re4`, `red_edge_4`), `'swir1'` (`swir`, `swir_1`, `swir-1`), `'swir2'` (`swir_2`, `swir-2`), `'red_edge'` (`re`, `rededge`, `red-edge`), `'re1'` (`re_1`, `red_edge_1`), `'re2'` (`re_2`, `red_edge_2`), `'re3'` (`re_3`, `red_edge_3`).
- **SAR** (`band_names={1: "vv", ...}`): `'vv'` (`sigma0_vv`, `gamma0_vv`, `beta0_vv`), `'vh'` (`sigma0_vh`, `gamma0_vh`, `beta0_vh`), `'hh'` (`sigma0_hh`, `gamma0_hh`, `beta0_hh`), `'hv'` (`sigma0_hv`, `gamma0_hv`, `beta0_hv`).

### Optional `index_params` Overrides
- `ARVI`: `gamma=1.0`
- `Brightness_Scaled`: `stretch_std_devs=2.0` (auto-computes `mean ± k*std` from input if `in_min`/`in_max` not overridden)
- `EVI`: `g=2.5, c1=6.0, c2=7.5, l=1.0`
- `EVI2`: `g=2.5, c=2.4, l=1.0`
- `GCMI`: `a=3.4, b=2.9, c=4.3, d=1.1`
- `MFI`: `lambda_red=665.0, lambda_re1=705.0, lambda_re2=740.0, lambda_re3=783.0, lambda_nir_narrow=865.0, lambda_swir2=2190.0`
- `OSAVI`: `y=0.16` (falls back to `l` if passed)
- `PVI`: `a=1.0, b=0.0`
- `SAVI` / `SATVI`: `l=0.5`
- `WDRVI`: `a=0.2`
- `WDVI`: `s=1.0`
- `DPSVI` (SAR): `vv_max` (auto-computed over valid pixels in linear power if omitted)

### All 97 Optical Indices

| Index | Required Bands | Equation |
| --- | --- | --- |
| `AMMI` | `red, nir, swir1` | `((nir - red) / (red + swir1)) * ((nir - swir1) / (swir1 - 0.65 * red))` |
| `ARVI` | `blue, red, nir` | `(nir - (red - gamma * (blue - red))) / (nir + (red - gamma * (blue - red)))` |
| `AWEI_nsh` | `green, nir, swir1, swir2` | `4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)` |
| `AWEI_sh` | `blue, green, nir, swir1, swir2` | `blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2` |
| `BAI` | `red, nir` | `1 / ((0.1 - red)^2 + (0.06 - nir)^2)` |
| `BSI` | `blue, red, nir, swir1` | `((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))` |
| `Brightness` | `blue, green, red` | `(blue + green + red) / 3` |
| `Brightness_Scaled` | `blue, green, red` | `clip((Brightness - (mean - k * std)) / (2 * k * std), 0, 1)` |
| `CI_red_edge` | `red_edge, nir` | `(nir / red_edge) - 1` |
| `CIgreen` | `green, nir` | `(nir / green) - 1` |
| `CIre` | `red_edge, nir` | `(nir / red_edge) - 1` |
| `CMR` | `swir1, swir2` | `swir1 / swir2` |
| `CMRI` | `green, red, nir` | `((nir - red) / (nir + red)) - ((green - nir) / (green + nir))` |
| `CTVI` | `red, nir` | `(NDVI + 0.5) / sqrt(abs(NDVI + 0.5))` |
| `DNVI` | `swir1, swir2` | `(swir1 - swir2)^2 / (swir1 + swir2)` |
| `DVI` | `red, nir` | `nir - red` |
| `EMI` | `green, nir, swir1` | `(nir - swir1) / (green + nir)` |
| `EMVI` | `green, swir1, swir2` | `(green - swir2) / (swir1 - green)` |
| `EVI` | `blue, red, nir` | `G * ((nir - red) / (nir + C1 * red - C2 * blue + L))` |
| `EVI2` | `red, nir` | `G * ((nir - red) / (nir + C * red + L))` |
| `ExG` | `blue, green, red` | `2 * green - red - blue` |
| `FDI` | `green, red, nir` | `nir - (red + green)` |
| `FMR` | `nir, swir1` | `swir1 / nir` |
| `GCMI` | `blue, green, red, nir, swir1` | `(A * EVI + B * GNDVI) / (C * LSWI + D * RNDWI)` |
| `GCVI` | `green, nir` | `(nir / green) - 1` |
| `GEMI` | `red, nir` | `eta * (1 - 0.25 * eta) - ((red - 0.125) / (1 - red)), eta = (2 * (nir^2 - red^2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)` |
| `GLI` | `blue, green, red` | `(2 * green - red - blue) / (2 * green + red + blue)` |
| `GMNDWI` | `green, swir1` | `(green - swir1) / (green + swir1)` |
| `GNDVI` | `green, nir` | `(nir - green) / (nir + green)` |
| `GNDWI` | `green, nir` | `(green - nir) / (green + nir)` |
| `IMFI` | `blue, green, nir` | `(blue + green - 2 * nir) / (blue + green + 2 * nir)` |
| `IMI` | `green, red, nir, swir1` | `(nir - red) / (swir1 + green)` |
| `IOR` | `blue, red` | `red / blue` |
| `IRECI` | `red, re1, re2, re3` | `(re3 - red) / (re1 / re2)` |
| `LSWI` | `nir, swir1` | `(nir - swir1) / (nir + swir1)` |
| `MCARI` | `green, red, re1` | `((re1 - red) - 0.2 * (re1 - green)) * (re1 / red)` |
| `MDI1` | `nir, swir1` | `(nir - swir1) / swir1` |
| `MDI2` | `nir, swir2` | `(nir - swir2) / swir2` |
| `MFI` | `red, re1, re2, re3, nir_narrow, swir2` | `0.25 * sum_i(rho_i - (swir2 + (red - swir2) * (2190 - lambda_i) / 1525))` |
| `MI` | `nir, swir1` | `((nir - swir1) / (nir * swir1)) * 10000` |
| `MMRI` | `green, red, nir, swir1` | `(|MNDWI| - |NDVI|) / (|MNDWI| + |NDVI|)` |
| `MNDWI` | `green, swir1` | `(green - swir1) / (green + swir1)` |
| `MSAVI` | `red, nir` | `(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2` |
| `MSAVI2` | `red, nir` | `(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2` |
| `MTCI` | `red, re1, re2` | `(re2 - re1) / (re1 - red)` |
| `MVI` | `green, nir, swir1` | `(nir - green) / (swir1 - green)` |
| `MVI_Abs` | `green, nir, swir1` | `|nir - green| / |swir1 - green|` |
| `NBR` | `nir, swir2` | `(nir - swir2) / (nir + swir2)` |
| `NBR2` | `swir1, swir2` | `(swir1 - swir2) / (swir1 + swir2)` |
| `NBRI` | `nir, swir2` | `(nir - swir2) / (nir + swir2)` |
| `NDBI` | `nir, swir1` | `(swir1 - nir) / (swir1 + nir)` |
| `NDBSI` | `blue, green, red, nir, swir1` | `Liu et al. (2022): v = red - green, r = 1 - (swir1 - nir) / (3 * |nir - red|), k = v * r; -|swir1 - blue| / (swir1 + blue) if k < 0 else (swir1 - blue) / (swir1 + blue)` |
| `NDGI` | `green, red` | `(green - red) / (green + red)` |
| `NDII` | `nir, swir1` | `(nir - swir1) / (nir + swir1)` |
| `NDMI` | `nir, swir1` | `(nir - swir1) / (nir + swir1)` |
| `NDMaI` | `green, swir2` | `(swir2 - green) / (swir2 + green)` |
| `NDRE` | `red_edge, nir` | `(nir - red_edge) / (nir + red_edge)` |
| `NDRE_RE` | `re1, re2` | `(re2 - re1) / (re2 + re1)` |
| `NDSI` | `green, swir1` | `(green - swir1) / (green + swir1)` |
| `NDVI` | `red, nir` | `(nir - red) / (nir + red)` |
| `NDWI` | `green, nir` | `(green - nir) / (green + nir)` |
| `NDWI_Gao` | `nir, swir1` | `(nir - swir1) / (nir + swir1)` |
| `NDWI_McFeeters` | `green, nir` | `(green - nir) / (green + nir)` |
| `NDWVI` | `green, swir2` | `(swir2 - green) / (swir2 + green)` |
| `NGRDI` | `green, red` | `(green - red) / (green + red)` |
| `NIMI` | `red, re2, re3, nir` | `(3 * red - (re2 + re3 + nir)) / (3 * red + re2 + re3 + nir)` |
| `NIRv` | `red, nir` | `((nir - red) / (nir + red)) * nir` |
| `NPCRI` | `red, blue` | `(red - blue) / (red + blue)` |
| `NRVI` | `red, nir` | `((red / nir) - 1) / ((red / nir) + 1)` |
| `OSAVI` | `red, nir` | `(1 + Y) * ((nir - red) / (nir + red + Y))` |
| `PSRI` | `blue, red, re2` | `(red - blue) / re2` |
| `PSRI_Green` | `green, red, red_edge` | `(red - green) / red_edge` |
| `PVI` | `red, nir` | `(nir - a * red - b) / sqrt(1 + a^2)` |
| `REMCH` | `re1, nir, swir1, swir2` | `(nir / swir1) - ((nir / swir2) / (nir / re1)) + (nir / re1)` |
| `REMI` | `green, red, red_edge, swir1` | `(red_edge - red) / (swir1 - green)` |
| `RNDWI` | `red, swir1` | `(red - swir1) / (red + swir1)` |
| `RVI` | `red, nir` | `nir / red` |
| `RVI_NIR_Red` | `red, nir` | `nir / red` |
| `RVI_Red_NIR` | `red, nir` | `red / nir` |
| `S2REP` | `red, re1, re2, re3` | `705 + 35 * (((re3 + red) / 2) - re1) / (re2 - re1)` |
| `SATVI` | `red, swir1, swir2` | `((swir1 - red) / (swir1 + red + L)) * (1 + L) - (swir2 / 2)` |
| `SAVI` | `red, nir` | `((nir - red) / (nir + red + L)) * (1 + L)` |
| `SI` | `blue, green, red` | `((1 - blue) * (1 - green) * (1 - red))^(1/3)` |
| `SLAVI` | `red, nir, swir2` | `nir / (red + swir2)` |
| `SR` | `red, nir` | `nir / red` |
| `TTVI` | `red, nir` | `sqrt(abs(NDVI + 0.5))` |
| `TVI` | `red, nir` | `sqrt(NDVI + 0.5)` |
| `VARI` | `blue, green, red` | `(green - red) / (green + red - blue)` |
| `WBI` | `blue, nir` | `blue / nir` |
| `WBI_BlueNIR` | `blue, nir` | `blue / nir` |
| `WDRVI` | `red, nir` | `(a * nir - red) / (a * nir + red)` |
| `WDVI` | `red, nir` | `nir - s * red` |
| `WFI` | `red, nir, swir2` | `(nir - red) / swir2` |
| `WI2015` | `green, red, nir, swir1, swir2` | `1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir1 - 71 * swir2` |
| `WRI` | `green, red, nir, swir1` | `(green + red) / (nir + swir1)` |
| `Whiteness` | `blue, green, red` | `(|blue - mean| + |green - mean| + |red - mean|) / mean, mean = (blue + green + red) / 3` |
| `kNDVI` | `red, nir` | `tanh(((nir - red) / (nir + red))^2)` |

### All 63 SAR Indices

| Index | Domain | Required Bands | Equation |
| --- | --- | --- | --- |
| `BMI` | `power` | `hh, vv` | `(hh + vv) / 2` |
| `CR_HH_HV` | `power` | `hh, hv` | `hh / hv` |
| `CR_HH_HV_dB` | `dB` | `hh, hv` | `hh_dB - hv_dB` |
| `CR_HV_HH` | `power` | `hh, hv` | `hv / hh` |
| `CR_HV_HH_dB` | `dB` | `hh, hv` | `hv_dB - hh_dB` |
| `CR_VH_VV` | `power` | `vv, vh` | `vh / vv` |
| `CR_VH_VV_dB` | `dB` | `vv, vh` | `vh_dB - vv_dB` |
| `CR_VV_VH` | `power` | `vv, vh` | `vv / vh` |
| `CR_VV_VH_dB` | `dB` | `vv, vh` | `vv_dB - vh_dB` |
| `CSI` | `power` | `hh, vv` | `vv / (vv + hh)` |
| `DPSVI` | `power` | `vv, vh` | `(((vv_max - vv) + vh) / sqrt(2)) * ((vv + vh) / vv) * vh` |
| `DPSVIm` | `power` | `vv, vh` | `(vv^2 + vv * vh) / sqrt(2)` |
| `DpRVI_HH` | `power` | `hh, hv` | `(4 * hv) / (hh + hv)` |
| `DpRVI_VV` | `power` | `vv, vh` | `(4 * vh) / (vv + vh)` |
| `DpRVIc` | `power` | `vv, vh` | `(q * (q + 3)) / (q + 1)^2, q = vh / vv` |
| `DpRVIc_HH_HV` | `power` | `hh, hv` | `(q * (q + 3)) / (q + 1)^2, q = hv / hh` |
| `DpRVIc_VV_VH` | `power` | `vv, vh` | `(q * (q + 3)) / (q + 1)^2, q = vh / vv` |
| `HDDPI` | `power` | `hh, hv` | `(hh + hv) / hh` |
| `Mean_HH_HV_dB` | `dB` | `hh, hv` | `(hh_dB + hv_dB) / 2` |
| `Mean_VV_VH_dB` | `dB` | `vv, vh` | `(vv_dB + vh_dB) / 2` |
| `NDI_HH_HV` | `power` | `hh, hv` | `(hh - hv) / (hh + hv)` |
| `NDI_HV_HH` | `power` | `hh, hv` | `(hv - hh) / (hv + hh)` |
| `NDI_VH_VV` | `power` | `vv, vh` | `(vh - vv) / (vh + vv)` |
| `NDI_VV_VH` | `power` | `vv, vh` | `(vv - vh) / (vv + vh)` |
| `NDPDI` | `power` | `vv, vh` | `(vv - vh) / (vv + vh)` |
| `NDPDI_HH_HV` | `power` | `hh, hv` | `(hh - hv) / (hh + hv)` |
| `NDPDI_VV_VH` | `power` | `vv, vh` | `(vv - vh) / (vv + vh)` |
| `NDPI_HH_HV` | `power` | `hh, hv` | `(hh - hv) / (hh + hv)` |
| `NDPI_VV_VH` | `power` | `vv, vh` | `(vv - vh) / (vv + vh)` |
| `NRPB` | `power` | `vv, vh` | `(vh - vv) / (vh + vv)` |
| `PR_HH_HV` | `power` | `hh, hv` | `hh / hv` |
| `PR_HH_HV_dB` | `dB` | `hh, hv` | `hh_dB - hv_dB` |
| `PR_HH_VV` | `power` | `hh, vv` | `hh / vv` |
| `PR_HH_VV_dB` | `dB` | `hh, vv` | `hh_dB - vv_dB` |
| `PR_HV_HH` | `power` | `hh, hv` | `hv / hh` |
| `PR_HV_HH_dB` | `dB` | `hh, hv` | `hv_dB - hh_dB` |
| `PR_VH_VV` | `power` | `vv, vh` | `vh / vv` |
| `PR_VH_VV_dB` | `dB` | `vv, vh` | `vh_dB - vv_dB` |
| `PR_VV_HH` | `power` | `hh, vv` | `vv / hh` |
| `PR_VV_HH_dB` | `dB` | `hh, vv` | `vv_dB - hh_dB` |
| `PR_VV_VH` | `power` | `vv, vh` | `vv / vh` |
| `PR_VV_VH_dB` | `dB` | `vv, vh` | `vv_dB - vh_dB` |
| `Pol_Diff_HH_HV` | `power` | `hh, hv` | `hh - hv` |
| `Pol_Diff_VV_VH` | `power` | `vv, vh` | `vv - vh` |
| `RFDI` | `power` | `hh, hv` | `(hh - hv) / (hh + hv)` |
| `RFDI_HH_HV` | `power` | `hh, hv` | `(hh - hv) / (hh + hv)` |
| `RFDI_VV_VH` | `power` | `vv, vh` | `(vv - vh) / (vv + vh)` |
| `RVI` | `power` | `hh, vv, hv` | `(8 * hv) / (hh + vv + 2 * hv)` |
| `RVI4S1` | `power` | `vv, vh` | `(4 * vh) / (vv + vh)` |
| `RVI_HH_HV` | `power` | `hh, hv` | `(4 * hv) / (hh + hv)` |
| `RVI_Quad` | `power` | `hh, vv, hv` | `(8 * hv) / (hh + vv + 2 * hv)` |
| `RVI_VV_VH` | `power` | `vv, vh` | `(4 * vh) / (vv + vh)` |
| `SDWI` | `power` | `vv, vh` | `ln(10 * vv * vh) - 8` |
| `SDWI_HH_HV` | `power` | `hh, hv` | `ln(10 * hh * hv) - 8` |
| `SDWI_VV_VH` | `power` | `vv, vh` | `ln(10 * vv * vh) - 8` |
| `Span_HH_HV` | `power` | `hh, hv` | `hh + hv` |
| `Span_HH_HV_dB` | `power` | `hh, hv` | `10 * log10(hh + hv)` |
| `Span_Quad` | `power` | `hh, vv, hv` | `hh + vv + 2 * hv` |
| `Span_Quad_dB` | `power` | `hh, vv, hv` | `10 * log10(hh + vv + 2 * hv)` |
| `Span_VV_VH` | `power` | `vv, vh` | `vv + vh` |
| `Span_VV_VH_dB` | `power` | `vv, vh` | `10 * log10(vv + vh)` |
| `VDDPI` | `power` | `vv, vh` | `(vv + vh) / vv` |
| `VSI` | `power` | `hh, vv, hv` | `hv / (hv + (hh + vv) / 2)` |

---

## 4. Remaining `rsgispy.imagecalc.*` Submodules

- **`stats`** (+ `imagecalc.get_img_histogram`):
  - `StatsSummary(min=0.0, max=0.0, sum=0.0, median=0.0, stdev=0.0, mean=0.0, mode=0.0, calc_min=False, calc_max=False, calc_sum=False, calc_mean=False, calc_stdev=False, calc_median=False, calc_mode=False)`
  - `calc_basic_img_band_stats(input_img, no_data_val=None, use_no_data=True, use_pbar=True)` -> `(mins, maxs, means, stdevs)` arrays per band
  - `count_pxls_of_val(input_img, vals, img_band=None, use_pbar=True)` -> `np.ndarray`
  - `get_unique_values(input_img, img_band=1)` -> `np.ndarray`
  - `image_pixel_column_summary(input_img, output_img, sum_stats: StatsSummary, use_no_data=True, *, gdalformat='GTIFF', datatype=9, no_data_val=0.0, use_pbar=True, creation_opts=None)`
  - `calc_multi_img_band_stats(input_imgs, output_img, summary_stat, use_no_data=False, *, ...)` (`summary_stat`: `rsgispy.SUMTYPE_*`)
  - `get_img_sum_stats_in_pxl(in_ref_img, in_stats_img, output_img, sum_stats=(2,), stats_img_band=1, use_no_data=True, io_grid_x=16, io_grid_y=16, *, ...)`
  - `calc_mask_img_pxl_val_prob(input_img, in_msk_img, output_img, img_bands, msk_val, bin_widths=None, use_no_data=True, rescale_probs=True, *, gdalformat='GTIFF', use_pbar=True, creation_opts=None)`
  - `get_img_histogram(input_img, img_band=1, n_bins=None, bin_size=None, integer_bins=False, val_range=None, no_data_val=None, use_no_data=True, in_msk_img=None, img_msk_val=1, use_pbar=True)` -> `Dict[str, Any]`
- **`percentiles`**:
  - `calc_band_percentile(input_img, percentile, no_data_val=None)` -> `List[float]`
  - `calc_band_percentile_msk(input_img, in_msk_img, msk_val, percentiles, no_data_val=None, use_pbar=True)` -> `np.ndarray`
  - `calc_imgs_pxl_percentiles(input_imgs, output_img, percentiles, *, gdalformat='GTIFF', no_data_val=0.0, use_pbar=True, creation_opts=None)`
  - `calc_img_band_pxl_percentiles(input_imgs, output_img, percentiles, *, ...)`
- **`rescale`**:
  - `ALL_NORM_METHODS` (23 methods): `('none', 'linear', 'stdev', 'cumulative', 'user', 'zscore', 'robust_zscore', 'decimal_scaling', 'log', 'log1p', 'sigmoid', 'tanh', 'softmax', 'l1', 'l2', 'max_norm', 'standard_scaler', 'minmax_scaler', 'maxabs_scaler', 'robust_scaler', 'power_transformer', 'quantile_transformer', 'normalizer')`
  - `ImageBandRescale(band=0, in_min=0.0, in_max=0.0, no_data_val=0.0, out_min=0.0, out_max=0.0, out_no_data=0.0)`
  - `calc_img_rescale(input_imgs, output_img, c_no_data_val=0.0, c_offset=0.0, c_gain=1.0, n_no_data_val=0.0, n_offset=0.0, n_gain=1.0, *, ...)`
  - `rescale_img_pxl_vals(input_img, output_img, band_rescale_objs=(), trim_to_limits=True, *, ...)`
  - `calc_img_norm_params(input_img, method='linear', no_data_val=None, out_stats_file=None, out_min=None, out_max=None, bands=None, lower=2.0, upper=98.0, n_stdevs=2.0, in_min_max=None, method_kwargs=None, max_samples=None, rnd_seed=None, use_pbar=True)` -> `Dict[str, Any]`
  - `apply_img_norm_params(input_img, output_img, norm_params=None, stats_file=None, out_min=None, out_max=None, trim_to_limits=None, *, gdalformat='GTIFF', datatype=9, out_no_data_val=None, use_pbar=True, creation_opts=None)`
  - `normalise_img_pxl_vals(input_img, output_img, method='linear', out_min=None, out_max=None, out_stats_file=None, bands=None, lower=2.0, upper=98.0, n_stdevs=2.0, in_min_max=None, method_kwargs=None, trim_to_limits=None, max_samples=None, rnd_seed=None, *, gdalformat='GTIFF', datatype=9, no_data_val=None, out_no_data_val=None, use_pbar=True, creation_opts=None)` -> `Dict[str, Any]`
- **`comparison`**:
  - `calc_img_difference(in_a_img, in_b_img, output_img, *, gdalformat='GTIFF', datatype=9, use_pbar=True, creation_opts=None)`
  - `are_imgs_equal(in_ref_img, in_cmp_img, prop_eql=1.0, flt_dif=0.0001, use_pbar=True)` -> `(bool, float)`
  - `are_img_bands_equal(in_ref_img, img_ref_band, in_cmp_img, img_cmp_band, prop_eql=1.0, flt_dif=0.0001, use_pbar=True)` -> `(bool, float)`
  - `calc_img_correlation(in_a_img, in_b_img, img_a_band, img_b_band, img_a_no_data=None, img_b_no_data=None, corr_stat_method=rsgispy.STATS_CORR_PEARSONS)` -> `(stat, pval)`
  - `calc_img_earth_move_dist(...)`, `calc_img_mutual_info(...)`, `calc_img_similarity_metrics(in_a_img, in_b_img, metrics=None, img_a_bands=None, img_b_bands=None, img_a_no_data=None, img_b_no_data=None, hist_bins=64, sub_pxl_res=4)` (uses `rsgispy.METRIC_*` or string names).
- **`distance`**: `calc_dist_to_img_vals(input_img, output_img, pxl_vals, img_band=1, max_dist=None, unit_geo=True, *, gdalformat='GTIFF', no_data_val=None, out_no_data_val=None, use_pbar=True, creation_opts=None)`, `calc_dist_to_img_vals_tiled(..., max_dist=1000, unit_geo=True, tmp_dir='tmp', tile_size=2000, n_cores=-1, ...)`, `buffer_img_pxl_vals(input_img, output_img, pxl_vals, buf_thres, img_band=1, unit_geo=True, tmp_dir=None, *, ...)`.
- **`encoding`**: `recode_int_raster(input_img, output_img, recode_dict, keep_vals_not_in_dict=True, *, datatype=3, ...)`, `one_hot_category_encoding(input_img, output_img, recode_dict=None, *, ...)`.
- **`fill`**: `calc_fill_regions_knn(in_ref_img, in_fill_regions_img, output_img, ref_no_data, fill_region_val, k=5, summary=1, *, datatype=3, ...)`, `region_growing_fill(in_regions_img, in_fill_regions_img, output_img, ref_no_data=None, fill_region_val=1, connectivity=8, max_iters=None, *, ...)`, `region_growing_fill_tiled(in_regions_imgs, in_fill_regions_imgs, output_imgs, ...)`.
- **`leastcostpath`**: `perform_least_cost_path_calc(cost_surface_img, output_img, start_coord, stop_coord, cost_img_band=1, *, gdalformat='GTIFF')`.
- **`calc_pt_win_smpls`**: `RSGISCalcSumVals`, `RSGISDebugExportImg(out_path, img_name)`, `calc_pt_smpl_img_vals(input_img, vec_file, vec_lyr, calc_objs, out_vec_file=None, out_vec_lyr=None, out_format='GPKG', interp_method=2, angle_col=None, x_box_col='xbox', y_box_col='ybox', *, no_data_val=None, use_pbar=True, out_compress=None)` -> `GeoDataFrame | None` (raises `RSGISPyException` if input image lacks CRS).

---

## 5. `rsgispy.imageutils.*` Submodules

- **`arrays`**: `create_img_from_array`, `create_img_from_array_ref_img`, `get_img_coords_pxl_values`, `get_img_pxl_values`, `set_img_pxl_values`, `get_img_pxl_column`, `get_img_column`, `get_img_band_pxl_data`, `get_img_data_as_arr`.
- **`bands`**: `select_img_bands(input_img, output_img, bands, *)`, `stack_img_bands(input_imgs, output_img, band_names=None, skip_value=None, *)`, `SharpBandInfo(band=None, status=None, name=None)`, `sharpen_low_res_bands(input_img, output_img, band_info, win_size=7, *)`.
- **`blank`**: `create_blank_img`, `create_copy_img`, `create_blank_buf_img_from_ref_img`, `create_blank_img_from_bbox`, `create_blank_ref_imgs` (also re-exported at `rsgispy.imageutils.create_blank_ref_imgs`).
- **`checks`**: `do_img_res_match`, `check_img_lst`, `check_img_file_comparison`, `do_images_overlap`, `do_gdal_layers_have_same_proj`, `test_img_lst_intersects`.
- **`colourtable`**: `set_img_thematic(input_img, ignore_cog=False)`, `set_img_not_thematic(input_img, ignore_cog=False)`, `is_img_thematic(input_img, img_band=1)`, `get_img_band_colour_interp`, `set_img_band_colour_interp`, `get_colour_tab_info`, `get_colour_tab_info_for_def`, `define_colour_table(input_img, clr_lut, img_band=1)`.
- **`envopts`**: `get_gdal_img_creation_opts`, `set_img_creation_opts`, `set_env_vars_{lzw,deflate,zstd}_{gtiff,cog}_outs(bigtiff=False)`.
- **`imagelut`**: `create_img_extent_lut(input_imgs, vec_file=None, vec_lyr=None, out_format='GPKG', *, ignore_none_imgs=False, out_proj_wgs84=False, overwrite_lut_file=False, use_pbar=True, out_compress=None)`, `query_img_lut(scn_bbox, lut_db_file, lyr_name=None)`, `get_all_lut_imgs`, `get_raster_lyr`, `query_file_lut`.
- **`masking`**: `ImageBandInfo(file_name=None, name=None, bands=None, band_names=None)` (`.from_dict()`, `.to_dict()`), `mask_img(input_img, in_msk_img, output_img, out_value, mask_value, *)`, `gen_finite_mask`, `gen_valid_mask`, `gen_img_edge_mask`, `mask_all_band_zero_vals`, `create_valid_mask(img_band_info, output_img, tmp_dir='', default_no_data_val=0.0, *)` (also re-exported at `rsgispy.imageutils.create_valid_mask`).
- **`metadata`**: `get_img_no_data_value`, `set_img_no_data_value`, `get_img_res`, `get_img_size`, `get_img_band_count`, `get_img_bbox` -> `BBox`, `get_img_bbox_in_proj` -> `BBox`, `transform_bounds_to_wgs84` -> `BBox | None`, `get_img_bbox_wgs84` -> `BBox | None`, `get_img_subset_pxl_bbox`, `get_img_pxl_spatial_coords`, `get_img_pxl_coords`, `get_img_files`, `has_gcps`, `copy_gcps`, `get_gdal_format_name`, `assign_proj`, `assign_spatial_info`, `copy_proj_from_img`, `copy_spatial_and_proj_from_img`, `get_wkt_proj_from_img`, `get_epsg_proj_from_img`, `get_utm_zone`, `set_band_names(input_img, band_names, feedback=False)`, `get_band_names(input_img)`, `get_gdal_datatype_from_img`, `get_rsgislib_datatype_from_img`, `get_gdal_datatype_name_from_img`, `get_gdal_format_from_ext`, `get_file_img_extension`, `rename_gdal_layer`, `delete_gdal_layer`, `set_img_band_metadata`, `get_img_band_metadata`, `get_img_band_metadata_fields`, `get_img_band_metadata_fields_dict`, `set_img_metadata`, `get_img_metadata`, `get_img_metadata_fields`, `get_img_metadata_fields_dict`.
- **`mosaic`**: `create_img_mosaic(input_imgs, output_img, background_val=0.0, skip_val=0.0, skip_band=1, overlap_behaviour=0, *, ...)` (`overlap_behaviour`: `0` overwrite, `1` min, `2` max), `include_imgs(input_img, include_imgs, *, input_bands=None, skip_val=None, use_pbar=True)`, `include_imgs_with_overlap(input_img, include_imgs, overlap, *, use_pbar=True)`, `combine_imgs_to_band`, `combine_binary_masks`.
- **`rasterise`**: `rasterise_vec_lyr(vec_file, vec_lyr, input_img, output_img, burn_val=1, att_column=None, use_vec_extent=False, thematic=True, *, no_data_val=0.0, ...)`, `create_vec_range_lut_score_img`, `create_vec_lst_lut_score_img`, `create_dist_zones_to_vec_lyr`.
- **`sampling`**: `perform_random_pxl_sample_in_mask`, `extract_img_pxl_sample`, `extract_img_pxl_vals_in_msk`, `assign_random_pxls`, `generate_random_pxl_vals_img`.
- **`spatial`**: `calc_wgs84_pixel_area`, `calc_wgs84_pixel_size`, `calc_pixel_locations`, `calc_wgs84_pixel_locations`, `grid_scattered_pts`.
- **`stats`**: `build_img_pyramids(input_img, resampling_method=Resampling.average, ignore_cog=False)`, `build_img_pyramids_multi_core(...)`, `pop_thmt_img_stats(input_img, add_clr_tab=True, calc_pyramids=True, ignore_zero=True, use_pbar=True, use_multi_core_pyramids=False, n_cores=1, tile_size=1024, ignore_cog=False)`, `pop_img_stats(input_img, no_data_val=None, calc_pyramids=True, ignore_cog=False, use_approx=False, use_pbar=True, use_multi_core_pyramids=False, n_cores=1, tile_size=1024)`.
- **`subsetting`**: `subset_bbox(input_img, output_img, min_x, max_x, min_y, max_y, *)`, `subset_to_img(input_img, in_roi_img, output_img, *)`, `subset(input_img, vec_file, vec_lyr, output_img, *)`.
- **`tiling`**: `create_tiles(input_img, out_img_base, tile_width, tile_height, tile_overlap=0, offset_tiles=False, out_img_ext='tif', *)`, `create_tiles_multi_core`, `create_raster_tiles_bbox`.
- **`vrt`**: `create_stack_images_vrt(...)`, `create_mosaic_images_vrt(...)`, `create_vrt_band_subset(...)`.
- **`warp`**: `resample_img_to_match(in_ref_img, in_process_img, output_img, interp_method=0, multicore=False, *)`, `reproject_image(input_img, output_img, out_epsg, interp_method=0, in_epsg=None, out_pxl_res='image', snap_to_grid=True, multicore=False, *)`, `gdal_warp(...)`, `gdal_translate(...)`.
