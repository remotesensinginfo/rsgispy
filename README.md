[![Python package](https://github.com/remotesensinginfo/rsgispy/actions/workflows/python-package.yml/badge.svg)](https://github.com/remotesensinginfo/rsgispy/actions/workflows/python-package.yml)[![Pylint](https://github.com/remotesensinginfo/rsgispy/actions/workflows/pylint.yml/badge.svg)](https://github.com/remotesensinginfo/rsgispy/actions/workflows/pylint.yml)[![CodeQL](https://github.com/remotesensinginfo/rsgispy/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/remotesensinginfo/rsgispy/actions/workflows/github-code-scanning/codeql)[![Upload Python Package](https://github.com/remotesensinginfo/rsgispy/actions/workflows/python-publish.yml/badge.svg)](https://github.com/remotesensinginfo/rsgispy/actions/workflows/python-publish.yml)![PyPI Version](https://img.shields.io/pypi/v/rsgispy)




# rsgispy

**rsgispy** is a pure Python reimplementation of the [Remote Sensing and GIS Software Library (RSGISLib)](https://www.rsgislib.org) designed to be easy to install, cross-platform, and straightforward to maintain while preserving a familiar API.

Instead of wrapping compiled C++ extensions and GDAL SWIG (`osgeo`) bindings, `rsgispy` builds directly on standard modern Python geospatial libraries—including `rasterio`, `geopandas`, `shapely`, `pyproj`, `numpy`, `pandas`, `matplotlib`, and `pymu-eval`—and is distributed under the Apache-2.0 licence.

## What is it for?

`rsgispy` provides high-level building blocks for end-to-end Earth observation, remote sensing, and GIS workflows:

- **Raster mathematics & spectral indices**: Block-wise muParser band/image expressions (`pymu-eval`), unified multi-band spectral index stacks with gain/offset scaling and integerisation, summary statistics, percentiles, image comparison, rescaling, KNN gap filling, and proximity/distance transforms.
- **Image & elevation processing**: Raster creation, mosaicking, tiling, VRTs, reprojection/warping, masking, rasterisation, morphological filtering, and DEM terrain derivatives (slope, aspect, illumination, roughness, and DEM filtering/fusion).
- **Vector & zonal workflows**: Vector metadata, reprojection, selection, attribute table operations, geometry creation (including grids and polygonisation), overlays, spatial metrics, GeoParquet I/O, and polygon/point zonal statistics.
- **Data access, drone & cartographic utilities**: Search and download clients for Copernicus OData, NASA CMR, OpenStreetMap, Planet, and USGS M2M; drone imagery organisation and flightline vector generation; and publication-ready mapping, WMTS, and plotting utilities.

## Who should use it?

- **Existing RSGISLib users** who want a `pip` installable, pure-Python alternative without C++/GDAL compilation constraints, while retaining familiar subpackage names, data type constants (`TYPE_*`), and workflows. Note, while the functionality is similar, `rsgispy` is not a drop-in replacement for RSGISLib and some workflows may require minor adjustments as the module structure has been reorganised and some functions have been renamed or restructured. Also, it is important to note that `rsgispy` is still under active development and some features from RSGISLib may not yet be implemented but RSGISLib will not be maintained in the long term and will shortly be deprecated in favour of `rsgispy`.
- **Remote sensing researchers & Earth observation practitioners** building reproducible satellite, airborne, or UAV processing pipelines in Python or Jupyter notebooks.
- **GIS analysts & data engineers** working with Cloud-Optimised GeoTIFFs, GeoParquet, and multi-layer raster/vector stacks on Linux, macOS, or Windows (Python 3.9–3.14).

## Background & RSGISLib History

`rsgispy` draws directly from the design and heritage of **RSGISLib** (*Remote Sensing and GIS Software Library*):

- **Origins (2008–2009)**: RSGISLib was started in April 2008 by Pete Bunting in the Department of Geography and Earth Sciences at Aberystwyth University to consolidate research code for environmental remote sensing and vegetation science. Dan Clewley joined in 2008, and the library was released publicly as open-source software (GPLv3) in November 2009, with Nathan Thomas later joining the core team around 2015.
- **Technical evolution**: Originally implemented in C++ atop GDAL/OGR, Boost, GSL, HDF5/KEA, and muParser, RSGISLib initially exposed over 300 commands through an XML scripting interface (`rsgisexe`) before introducing Python bindings around 2013–2014 and integrating closely with GDAL Raster Attribute Tables, the KEA image format, and RIOS. In 2021–2022, RSGISLib 5.0 overhauled the module layout, streamlined dependencies, and formalised automated testing across more than 800 tools.
- **Motivation for `rsgispy`**: While RSGISLib's C++ core offered strong performance, maintaining and distributing compiled binaries across platforms and Python versions remained complex outside `conda-forge`. `rsgispy` modernises and restructures those workflows in pure Python with zero `osgeo`/`rsgislib` dependencies, making installation via `pip` lightweight and transparent.

### RSGISLib Publication & Citation

If you use `rsgispy` or build on concepts from RSGISLib in academic work, please cite the foundational RSGISLib paper:

> Bunting, P., Clewley, D., Lucas, R. M., & Gillingham, S. (2014). The Remote Sensing and GIS Software Library (RSGISLib). *Computers & Geosciences*, 62, 216–226. https://doi.org/10.1016/j.cageo.2013.08.007

```bibtex
@article{bunting2014rsgislib,
  title   = {The Remote Sensing and GIS Software Library ({RSGISLib})},
  author  = {Bunting, Peter and Clewley, Daniel and Lucas, Richard M. and Gillingham, Sam},
  journal = {Computers \& Geosciences},
  volume  = {62},
  pages   = {216--226},
  year    = {2014},
  doi     = {10.1016/j.cageo.2013.08.007}
}
```

## Package Structure

`rsgispy` is organised into focused subpackages separating core processing engines from domain utilities:

| Module / Subpackage | Purpose |
| --- | --- |
| `rsgispy` | Core `TYPE_*` / `INTERP_*` constants, `RSGISPyException`, datatype helpers, and progress bars |
| `rsgispy.changedetect` | Image differencing, ratioing, CVA, ISFA, IR-MAD, and per-class outlier change detection (`pxloutlierchng`) |
| `rsgispy.imageprocess` | Windowed raster block-processing engine, spatial intersection, and dataset validation |
| `rsgispy.imageutils` | Raster creation, rasterisation, metadata, masking, mosaicking, tiling, VRTs, warping, and spatial helpers |
| `rsgispy.imagecalc` | Band/image maths (`pymu-eval`), spectral index stacks (`calcindices`), least-cost paths (`leastcostpath`), statistics, percentiles, comparison, encoding, distance, KNN fill, rescaling, and windowed sampling |
| `rsgispy.imagespectral` | Spectral classification (SAM, SCM), linear spectral unmixing, endmembers & PPI, spectral curve summaries (`indices`), spectral pre-processing (smoothing, outlier masking, polyfill), and transforms (whitening, PCA, MNF, Tasseled Cap, histogram matching) |
| `rsgispy.imagefilter` | Spatial moving-window filters (mean, median, min/max/mode, stddev, variance, difference, Kuwahara, Lee SAR, Sobel, Prewitt, Gaussian, Laplacian, and custom convolution kernels) |
| `rsgispy.imagemorphology` | Morphological raster filtering (dilation, erosion, opening, closing) |
| `rsgispy.elevation` | Terrain derivatives, solar illumination, DEM filtering, fusion, and surface roughness |
| `rsgispy.vectorprocess` | Shared low-level vector I/O (including compressed GeoParquet), CRS, bounding-box, and attribute helpers |
| `rsgispy.vectorutils` | Vector metadata, format translation, reprojection, merging, splitting, and selection |
| `rsgispy.vectorattrs` | Attribute table I/O (CSV/Excel/Parquet), column operations, and spatial joins |
| `rsgispy.vectorgeoms` | Geometry creation, grids (rectangular/H3), polygonisation, line/polygon manipulation, overlays, and spatial predicates |
| `rsgispy.vectorstats` | Spatial metrics, area/length/distance calculations, attribute summaries, and semi-variograms |
| `rsgispy.zonalstats` | Polygon and point zonal statistics extraction |
| `rsgispy.dataaccess` | API clients for Copernicus OData, NASA CMR, OpenStreetMap, Planet, and USGS M2M |
| `rsgispy.droneutils` | Drone photo organisation, EXIF metadata handling, and flightline vector generation |
| `rsgispy.tools` | General geospatial, geometry, projection, plotting, mapping, WMTS, archiving, file/system, accuracy/feature selection, and HTTP/FTP utilities |

## Installation

### Minimal installation

Install `rsgispy` with only its core dependencies (`geopandas`, `matplotlib`, `numpy`, `pandas`, `pymu-eval`, `pyproj`, `rasterio`, `shapely`, and `tqdm`):

```bash
pip install .
```

### Optional dependencies

Optional feature groups can be installed individually or all together via `pip` extras:

- `stats`: `scikit-gstat`, `scikit-learn`, `scipy`
- `vector`: `alphashape`, `h3`, `openpyxl`, `pyarrow`, `rtree`, `xlsxwriter`
- `mapping`: `contextily`, `mpl-scatter-density`, `owslib`
- `dataaccess`: `beautifulsoup4`, `osmnx`, `planet`, `pycurl`, `pysondb`, `requests`
- `imaging`: `h5py`, `imageio`, `pillow`, `pyexiftool`, `pymupdf`, `qrcode`, `scikit-image`
- `unmixing`: `cvxopt`, `pyMCR`, `pysptools`, `scikit-learn`, `scipy`
- `jupyter`: `ipywidgets`
- `all`: all optional runtime dependencies above
- `test`: `flake8`, `pylint`, `pytest`, `pytest-cov`, and `stats`
- `docs`: `sphinx`, `sphinx-rtd-theme`
- `dev`: `all`, `test`, and `docs`

```bash
pip install ".[vector]"
pip install ".[all]"
pip install -e ".[test]"
pip install -e ".[all,test]"
pip install -e ".[dev]"
```

> **Note:** On Linux, installing `dataaccess` or `all` may require `libcurl4-openssl-dev` and `libssl-dev` if `pycurl` is built from source. Functions using `pyexiftool` (`imaging` / `all`) also require the `exiftool` command-line tool on `PATH`.

## Quickstart

Calculate a stack of spectral indices scaled to `int16` in a single pass:

```python
import rsgispy
import rsgispy.imagecalc.calcindices

band_names = {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"}
rsgispy.imagecalc.calcindices.calc_indices(
    input_img="s2_refl.tif",
    band_names=band_names,
    indices=["NDVI", "NDWI_McFeeters", "MNDWI", "EVI"],
    output_img="indices_int16.tif",
    gdalformat="GTIFF",
    datatype=rsgispy.TYPE_16INT,
    in_gain=10000.0,
    in_offset=0.0,
    out_gain=1000.0,
    out_offset=0.0,
)
```

## Documentation & Licence

Full Sphinx API documentation and index equation references live under [`docs/`](docs/index.rst) and can be built locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

`rsgispy` is licensed under the [Apache-2.0 Licence](LICENSE).
