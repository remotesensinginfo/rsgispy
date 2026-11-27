Tutorials
=========

The `rsgispy tutorials <https://github.com/remotesensinginfo/rsgispy-tutorials>`_
are Jupyter notebooks that teach remote sensing and GIS analysis with
rsgispy on real data: Sentinel-2, Landsat and Sentinel-1 images, a 10 m DEM,
HyMap hyperspectral data, airborne lidar metrics, ICESat-2 depths and species
sightings. They start with reading an image and calculating an index, and
build up to classification, regression, change detection, spectral unmixing,
hydrology, chained tile-graph workflows and two case studies. Each notebook
says what you will learn, explains why each step is done and points out where
rsgispy differs from RSGISLib.

The tutorials and the :doc:`guides <index>` complement each other: the
tutorials go step by step through one dataset at a time, while each guide
shows a complete workflow as a single tested script. Several guides use the
tutorial data for their figures and outputs, and each guide links to the
notebooks that cover the same methods.

Getting the tutorials
---------------------

The data (about 1 GB) is stored with `Git LFS <https://git-lfs.com/>`_, so
install Git LFS before cloning (or run ``git lfs pull`` after cloning). The
conda environment installs rsgispy and everything the notebooks use:

.. code-block:: bash

   git lfs install
   git clone https://github.com/remotesensinginfo/rsgispy-tutorials.git
   cd rsgispy-tutorials
   conda env create -f environment.yml
   conda activate rsgispy-tutorials
   jupyter lab

Each notebook reads its inputs from ``data/`` and writes its outputs to the
``outputs/`` folder of its own section. Run the notebooks of a section in
order, as some read the outputs of earlier ones. The notebooks in section 12
download data and need an account with each data provider.

Running the guides on the tutorial data
---------------------------------------

The guides that use the tutorial data have a script
``docs/examples/<guide>_tutorial.py`` next to the guide's example. It
prepares the tutorial inputs and runs the example. These scripts made the
figures and printed outputs shown in those guides:

.. code-block:: bash

   python docs/examples/hydrology_workflow_tutorial.py \
       --data-dir rsgispy-tutorials/data --out-dir hydro_out

To run all of them as tests, point ``RSGISPY_TUTORIAL_DATA`` at the data
folder:

.. code-block:: bash

   RSGISPY_TUTORIAL_DATA=rsgispy-tutorials/data pytest tests/test_tutorial_data_examples.py

The notebooks
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Section
     - Notebook
     - What it covers
   * - 1. Getting started
     - :tutorial:`01_image_basics_and_ndvi <01_getting_started/01_image_basics_and_ndvi.ipynb>`
     - Image information, band maths, NDVI, statistics and a first map
   * - 2. Making maps (guides: :doc:`guides/making_maps`, :doc:`guides/plot_gallery`)
     - :tutorial:`01_intro_matplotlib <02_making_maps/01_intro_matplotlib.ipynb>`
     - How Matplotlib figures and axes work, and the rsgispy plot style
   * -
     - :tutorial:`02_maps_from_eo_data <02_making_maps/02_maps_from_eo_data.ipynb>`
     - Maps of multi-band satellite images, with scale bars, north arrows and maps side by side
   * -
     - :tutorial:`03_maps_from_vector_data <02_making_maps/03_maps_from_vector_data.ipynb>`
     - Maps of vector layers, with basemaps
   * -
     - :tutorial:`04_combining_data_sources <02_making_maps/04_combining_data_sources.ipynb>`
     - Matching projections and mapping several layers together
   * - 3. Vector data and zonal statistics (guides: :doc:`guides/zonal_stats_and_vectors`)
     - :tutorial:`01_zonal_statistics <03_vector_data_and_zonal_stats/01_zonal_statistics.ipynb>`
     - Extracting image values to points and summarising them in polygons
   * -
     - :tutorial:`02_semivariograms <03_vector_data_and_zonal_stats/02_semivariograms.ipynb>`
     - Random sample points and empirical semi-variograms
   * - 4. Image classification (guides: :doc:`guides/pixel_classification_workflow`, :doc:`guides/segment_ml_workflow`)
     - :tutorial:`01_rule_based_classification <04_image_classification/01_rule_based_classification.ipynb>`
     - Spectral indices and rule-based classification
   * -
     - :tutorial:`02_unsupervised_classification <04_image_classification/02_unsupervised_classification.ipynb>`
     - Clustering with K-means, mean shift and other algorithms, and applying a fitted clusterer to another image
   * -
     - :tutorial:`03_supervised_pixel_classification <04_image_classification/03_supervised_pixel_classification.ipynb>`
     - Training samples, a random forest and applying it to an image
   * -
     - :tutorial:`04_object_based_classification <04_image_classification/04_object_based_classification.ipynb>`
     - Segmentation and classifying segments
   * - 5. Regression (guides: :doc:`guides/pixel_regression_workflow`)
     - :tutorial:`01_intro_and_explore_data <05_regression/01_intro_and_explore_data.ipynb>`
     - A forest inventory regression problem, explored with lidar metrics
   * -
     - :tutorial:`02_feature_selection <05_regression/02_feature_selection.ipynb>`
     - Selecting features with VIF, correlation, CQV and LassoLars
   * -
     - :tutorial:`03_simple_linear_regression <05_regression/03_simple_linear_regression.ipynb>`
     - A simple linear regression, applied to an image
   * -
     - :tutorial:`04_multiple_linear_regression <05_regression/04_multiple_linear_regression.ipynb>`
     - Multiple linear regression with the selected features, and jackknife and k-fold assessment
   * -
     - :tutorial:`05_multivariate_regression <05_regression/05_multivariate_regression.ipynb>`
     - Tuning and applying an Extra Trees regressor for several variables
   * -
     - :tutorial:`06_compare_regressors <05_regression/06_compare_regressors.ipynb>`
     - Comparing six regressors with repeated k-fold cross-validation
   * - 6. Change detection (guides: :doc:`guides/change_detection`)
     - :tutorial:`01_create_baseline_class <06_change_detection/01_create_baseline_class.ipynb>`
     - A baseline mangrove classification with XGBoost
   * -
     - :tutorial:`02_map_to_map_change <06_change_detection/02_map_to_map_change.ipynb>`
     - Comparing two classifications
   * -
     - :tutorial:`03_image_to_image_change <06_change_detection/03_image_to_image_change.ipynb>`
     - Differences and ratios of spectral indices
   * -
     - :tutorial:`04_map_to_image_change <06_change_detection/04_map_to_image_change.ipynb>`
     - Finding changed pixels of a class in a new image
   * -
     - :tutorial:`05_multivariate_change_detection <06_change_detection/05_multivariate_change_detection.ipynb>`
     - Change vector analysis, slow feature analysis and IR-MAD compared
   * - 7. Spectral analysis (guides: :doc:`guides/spectral_unmixing`)
     - :tutorial:`01_spectral_analysis_intro <07_spectral_analysis/01_spectral_analysis_intro.ipynb>`
     - Spectra, spectral resampling and continuum removal
   * -
     - :tutorial:`02_hyperspectral_image_analysis <07_spectral_analysis/02_hyperspectral_image_analysis.ipynb>`
     - Hyperspectral metadata, smoothing, PCA, MNF and continuum removal of an image
   * -
     - :tutorial:`03_spectral_unmixing <07_spectral_analysis/03_spectral_unmixing.ipynb>`
     - Endmembers (including PPI) and linear spectral unmixing
   * - 8. Hydrology (guides: :doc:`guides/hydrology_workflow`)
     - :tutorial:`01_hydrological_analysis <08_hydrology/01_hydrological_analysis.ipynb>`
     - Conditioning a DEM, flow routing, streams, watersheds, HAND flooding and terrain indices
   * - 9. Tile-graph workflows (guides: :doc:`guides/tile_graph`, :doc:`guides/large_rasters`)
     - :tutorial:`01_indices_classification_filtering <09_tile_graph_workflows/01_indices_classification_filtering.ipynb>`
     - Indices, a random forest classification and a majority filter chained in one in-memory tile graph
   * - 10. Case study: ICESat-2 bathymetry (guides: :doc:`guides/analysis_ready_data`, :doc:`guides/pixel_regression_workflow`)
     - :tutorial:`01_visualise_icesat2 <10_case_study_icesat2_bathymetry/01_visualise_icesat2.ipynb>`
     - The ICESat-2 depth points
   * -
     - :tutorial:`02_landsat_composites <10_case_study_icesat2_bathymetry/02_landsat_composites.ipynb>`
     - Composites of a stack of Landsat images: max-NDVI/NDWI, geomedian, median and quality mosaics
   * -
     - :tutorial:`03_landsat_composite_bathymetry <10_case_study_icesat2_bathymetry/03_landsat_composite_bathymetry.ipynb>`
     - A bathymetry map from a composite and ICESat-2 depths
   * -
     - :tutorial:`04_landsat_scenes_bathymetry <10_case_study_icesat2_bathymetry/04_landsat_scenes_bathymetry.ipynb>`
     - Bathymetry from individual scenes, combined
   * -
     - :tutorial:`05_cshelph_bathymetry <10_case_study_icesat2_bathymetry/05_cshelph_bathymetry.ipynb>`
     - Optional: classifying ICESat-2 photons with C-SHELPh
   * - 11. Case study: species distribution modelling (guides: :doc:`guides/species_distribution_modelling`)
     - :tutorial:`01_prepare_datasets <11_case_study_species_distribution/01_prepare_datasets.ipynb>`
     - Preparing sightings and environmental layers
   * -
     - :tutorial:`02_species_distribution_modelling <11_case_study_species_distribution/02_species_distribution_modelling.ipynb>`
     - Pseudo-absences, model fitting, evaluation and habitat suitability
   * - 12. Downloading data (guides: :doc:`guides/analysis_ready_data`)
     - :tutorial:`01_dwnld_usgs_landsat_scenes <12_downloading_data/01_dwnld_usgs_landsat_scenes.ipynb>`
     - Landsat scenes from the USGS (M2M API)
   * -
     - :tutorial:`02_nasa_cmr_download_icesat2 <12_downloading_data/02_nasa_cmr_download_icesat2.ipynb>`
     - ICESat-2 data from NASA's CMR
   * -
     - :tutorial:`03_nasa_cmr_download_gedi <12_downloading_data/03_nasa_cmr_download_gedi.ipynb>`
     - GEDI data from NASA's CMR
   * -
     - :tutorial:`04_nasa_cmr_download_modis <12_downloading_data/04_nasa_cmr_download_modis.ipynb>`
     - MODIS data from NASA's CMR
   * -
     - :tutorial:`05_copernicus_odata_sen1 <12_downloading_data/05_copernicus_odata_sen1.ipynb>`
     - Sentinel-1 from the Copernicus Data Space
   * -
     - :tutorial:`06_copernicus_odata_sen2 <12_downloading_data/06_copernicus_odata_sen2.ipynb>`
     - Sentinel-2 from the Copernicus Data Space
   * -
     - :tutorial:`07_planet_psscene <12_downloading_data/07_planet_psscene.ipynb>`
     - PlanetScope scenes from the Planet API
   * -
     - :tutorial:`08_icepyx_download_icesat2 <12_downloading_data/08_icepyx_download_icesat2.ipynb>`
     - ICESat-2 ATL03 data with icepyx (used by 10.05)
