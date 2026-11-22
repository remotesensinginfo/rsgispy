rsgispy.imagecalc.calcindices
=============================

.. automodule:: rsgispy.imagecalc.calcindices
   :members:
   :undoc-members:
   :show-inheritance:

Implemented Spectral Indices & Equations
----------------------------------------

All optical spectral index equations below operate on physical input band values
computed via ``physical = (dn - in_offset) / in_gain``. Canonical band names
accepted in ``band_names`` are:

* **Visible & Broad NIR/SWIR:** ``blue``, ``green``, ``red``, ``nir``,
  ``swir1`` (alias ``swir``, ~1.6 um), and ``swir2`` (~2.2 um).
* **Red-Edge & Narrow NIR:** ``red_edge`` (alias ``re``) for single red-edge
  indices; ``re1`` (~705 nm), ``re2`` (~740 nm), ``re3`` (~783 nm), and
  ``nir_narrow`` (aliases ``re4``, ``nir8a``, ~865 nm) for Sentinel-2
  multi-red-edge indices.

.. note::

   For sensor-specific orthogonal Tasseled Cap Brightness, Greenness, and
   Wetness (``TCB`` / ``GVI`` / ``WI``) transformations, use
   :func:`~rsgispy.imagespectral.transforms.calc_tasseled_cap` in
   :mod:`rsgispy.imagespectral.transforms`.

Naming Disambiguation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Water & moisture indices**: ``NDWI_McFeeters`` (also ``NDWI`` and
  ``GNDWI``) implements McFeeters (1996) ``(green - nir) / (green + nir)``;
  ``NDWI_Gao`` (also ``NDMI``, ``NDII``, and ``LSWI``) implements Gao (1996) /
  Hardisky et al. (1983) / Xiao et al. (2002)
  ``(nir - swir1) / (nir + swir1)`` (note that ``rsgislib.imagecalc.calc_ndwi``
  historically used this NIR/SWIR1 formula); ``MNDWI`` (also ``GMNDWI`` and
  ``NDSI``) implements Xu (2006) ``(green - swir1) / (green + swir1)``.
* **Mangrove NDMI vs. Moisture NDMI**: Shi et al. (2016)'s Normalised
  Difference Mangrove Index ``(swir2 - green) / (swir2 + green)`` is registered
  as ``NDMaI`` (with alias ``NDWVI``) to avoid colliding with canopy-moisture
  ``NDMI`` ``(nir - swir1) / (nir + swir1)``.
* **Mangrove Vegetation Index variants**: ``MVI`` computes
  ``(nir - green) / (swir1 - green)`` and ``MVI_Abs`` computes
  ``|nir - green| / |swir1 - green|`` (Baloloy et al., 2020).
* **Red-Edge vs. Integrated Mangrove Index**: ``REMI`` implements Chen et al.
  (2023) ``(red_edge - red) / (swir1 - green)``; ``IMI`` implements Umar &
  Abbas (2025) ``(nir - red) / (swir1 + green)``.
* **Water Band Ratio**: ``WBI_BlueNIR`` (alias ``WBI``) implements the
  multispectral RSGISLib ``blue / nir`` ratio, distinct from the narrowband
  hyperspectral Water Band Index (``R900 / R970``).
* **Ratio Vegetation Index**: ``SR``, ``RVI``, and ``RVI_NIR_Red`` compute
  ``nir / red``, whereas ``RVI_Red_NIR`` computes ``red / nir``.

General Vegetation & Canopy Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 22 14 31 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Defaults / Reference
   * - ``NDVI``
     - Normalised Difference Vegetation Index
     - ``red``, ``nir``
     - ``(nir - red) / (nir + red)``
     - Rouse et al. (1973, 1974)
   * - ``EVI``
     - Enhanced Vegetation Index
     - ``blue``, ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C1 * red - C2 * blue + L))``
     - ``G=2.5, C1=6.0, C2=7.5, L=1.0`` (Huete et al., 2002)
   * - ``EVI2``
     - Two-Band Enhanced Vegetation Index
     - ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C * red + L))``
     - ``G=2.5, C=2.4, L=1.0`` (Jiang et al., 2008)
   * - ``GNDVI``
     - Green Normalised Difference Vegetation Index
     - ``green``, ``nir``
     - ``(nir - green) / (nir + green)``
     - Gitelson et al. (1996)
   * - ``SAVI``
     - Soil Adjusted Vegetation Index
     - ``red``, ``nir``
     - ``((nir - red) / (nir + red + L)) * (1 + L)``
     - ``L=0.5`` (Huete, 1988)
   * - ``OSAVI``
     - Optimised Soil Adjusted Vegetation Index
     - ``red``, ``nir``
     - ``(1 + Y) * ((nir - red) / (nir + red + Y))``
     - ``Y=0.16`` (Rondeaux et al., 1996)
   * - ``MSAVI2`` (``MSAVI``)
     - Modified Soil Adjusted Vegetation Index 2
     - ``red``, ``nir``
     - ``(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2``
     - Qi et al. (1994)
   * - ``SATVI``
     - Soil Adjusted Total Vegetation Index
     - ``red``, ``swir1``, ``swir2``
     - ``((swir1 - red) / (swir1 + red + L)) * (1 + L) - (swir2 / 2)``
     - ``L=0.5`` (Marsett et al., 2006)
   * - ``SR`` (``RVI``, ``RVI_NIR_Red``)
     - Simple Ratio / Ratio Vegetation Index (NIR/Red)
     - ``red``, ``nir``
     - ``nir / red``
     - Birth & McVey (1968); Jordan (1969)
   * - ``RVI_Red_NIR``
     - Ratio Vegetation Index (Red/NIR)
     - ``red``, ``nir``
     - ``red / nir``
     - Inverse ratio convention
   * - ``DVI``
     - Difference Vegetation Index
     - ``red``, ``nir``
     - ``nir - red``
     - Tucker (1979)
   * - ``WDVI``
     - Weighted Difference Vegetation Index
     - ``red``, ``nir``
     - ``nir - s * red``
     - ``s=1.0`` (Clevers, 1989)
   * - ``PVI``
     - Perpendicular Vegetation Index
     - ``red``, ``nir``
     - ``(nir - a * red - b) / sqrt(1 + a^2)``
     - ``a=1.0, b=0.0`` (Richardson & Wiegand, 1977)
   * - ``NRVI``
     - Normalised Ratio Vegetation Index
     - ``red``, ``nir``
     - ``((red / nir) - 1) / ((red / nir) + 1)``
     - Baret & Guyot (1991)
   * - ``CTVI``
     - Corrected Transformed Vegetation Index
     - ``red``, ``nir``
     - ``(NDVI + 0.5) / sqrt(abs(NDVI + 0.5))``
     - Perry & Lautenschlager (1984)
   * - ``TVI``
     - Transformed Vegetation Index
     - ``red``, ``nir``
     - ``sqrt(NDVI + 0.5)``
     - Deering et al. (1975)
   * - ``TTVI``
     - Thiam's Transformed Vegetation Index
     - ``red``, ``nir``
     - ``sqrt(abs(NDVI + 0.5))``
     - Thiam (1997)
   * - ``GEMI``
     - Global Environmental Monitoring Index
     - ``red``, ``nir``
     - ``eta * (1 - 0.25 * eta) - ((red - 0.125) / (1 - red))``, where ``eta = (2 * (nir^2 - red^2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)``
     - Pinty & Verstraete (1992)
   * - ``ARVI``
     - Atmospherically Resistant Vegetation Index
     - ``blue``, ``red``, ``nir``
     - ``(nir - (red - gamma * (blue - red))) / (nir + (red - gamma * (blue - red)))``
     - ``gamma=1.0`` (Kaufman & Tanre, 1992)
   * - ``SLAVI``
     - Specific Leaf Area Vegetation Index
     - ``red``, ``nir``, ``swir2``
     - ``nir / (red + swir2)``
     - Lymburner et al. (2000)
   * - ``kNDVI``
     - Kernel Normalised Difference Vegetation Index
     - ``red``, ``nir``
     - ``tanh(((nir - red) / (nir + red))^2)``
     - Camps-Valls et al. (2021)
   * - ``NIRv``
     - Near-Infrared Reflectance of Vegetation
     - ``red``, ``nir``
     - ``((nir - red) / (nir + red)) * nir``
     - Badgley et al. (2017)
   * - ``WDRVI``
     - Wide Dynamic Range Vegetation Index
     - ``red``, ``nir``
     - ``(a * nir - red) / (a * nir + red)``
     - ``a=0.2`` (Gitelson, 2004)
   * - ``CIgreen`` (``GCVI``)
     - Chlorophyll Index Green / Green Chlorophyll Vegetation Index
     - ``green``, ``nir``
     - ``(nir / green) - 1``
     - Gitelson et al. (2003)
   * - ``NPCRI``
     - Normalised Pigment Chlorophyll Ratio Index
     - ``blue``, ``red``
     - ``(red - blue) / (red + blue)``
     - Penuelas et al. (1994)

Red-Edge Vegetation & Stress Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 22 16 29 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NDRE``
     - Normalised Difference Red-Edge Index
     - ``red_edge``, ``nir``
     - ``(nir - red_edge) / (nir + red_edge)``
     - Gitelson & Merzlyak (1994); Barnes et al. (2000)
   * - ``NDRE_RE``
     - Red-Edge-Only Normalised Difference Index
     - ``re1``, ``re2``
     - ``(re2 - re1) / (re2 + re1)``
     - Red-edge pair variant of ``NDRE``
   * - ``CIre`` (``CI_red_edge``)
     - Red-Edge Chlorophyll Index
     - ``red_edge``, ``nir``
     - ``(nir / red_edge) - 1``
     - Gitelson et al. (2003, 2005)
   * - ``MTCI``
     - MERIS Terrestrial Chlorophyll Index
     - ``red``, ``re1``, ``re2``
     - ``(re2 - re1) / (re1 - red)``
     - Dash & Curran (2004)
   * - ``MCARI``
     - Modified Chlorophyll Absorption Ratio Index
     - ``green``, ``red``, ``re1``
     - ``((re1 - red) - 0.2 * (re1 - green)) * (re1 / red)``
     - Daughtry et al. (2000)
   * - ``IRECI``
     - Inverted Red-Edge Chlorophyll Index
     - ``red``, ``re1``, ``re2``, ``re3``
     - ``(re3 - red) / (re1 / re2)``
     - Frampton et al. (2013)
   * - ``S2REP``
     - Sentinel-2 Red-Edge Position
     - ``red``, ``re1``, ``re2``, ``re3``
     - ``705 + 35 * (((re3 + red) / 2) - re1) / (re2 - re1)``
     - Frampton et al. (2013)
   * - ``PSRI``
     - Plant Senescence Reflectance Index
     - ``blue``, ``red``, ``re2``
     - ``(red - blue) / re2``
     - Merzlyak et al. (1999)
   * - ``PSRI_Green``
     - Plant Senescence Reflectance Index (Green/RE Variant)
     - ``green``, ``red``, ``red_edge``
     - ``(red - green) / red_edge``
     - Green-band variant of ``PSRI``

Visible-Band / UAV Vegetation Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 23 15 29 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``ExG``
     - Excess Green Index
     - ``blue``, ``green``, ``red``
     - ``2 * green - red - blue``
     - Woebbecke et al. (1995)
   * - ``NGRDI`` (``NDGI``)
     - Normalised Green-Red Difference Index
     - ``green``, ``red``
     - ``(green - red) / (green + red)``
     - Tucker (1979); Keshri et al. (2009)
   * - ``GLI``
     - Green Leaf Index
     - ``blue``, ``green``, ``red``
     - ``(2 * green - red - blue) / (2 * green + red + blue)``
     - Louhaichi et al. (2001)
   * - ``VARI``
     - Visible Atmospherically Resistant Index
     - ``blue``, ``green``, ``red``
     - ``(green - red) / (green + red - blue)``
     - Gitelson et al. (2002)

Mangrove & Wetland Forest Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 21 16 31 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``CMRI``
     - Combined Mangrove Recognition Index
     - ``green``, ``red``, ``nir``
     - ``NDVI - NDWI_McFeeters``
     - Gupta et al. (2018)
   * - ``EMI``
     - Enhanced Mangrove Index
     - ``green``, ``nir``, ``swir1``
     - ``(nir - swir1) / (green + nir)``
     - Prayudha et al. (2024)
   * - ``NDMaI`` (``NDWVI``)
     - Normalised Difference Mangrove / Wetland Vegetation Index
     - ``green``, ``swir2``
     - ``(swir2 - green) / (swir2 + green)``
     - Shi et al. (2016); Kumar et al. (2017)
   * - ``MVI``
     - Mangrove Vegetation Index
     - ``green``, ``nir``, ``swir1``
     - ``(nir - green) / (swir1 - green)``
     - Baloloy et al. (2020)
   * - ``MVI_Abs``
     - Mangrove Vegetation Index (Absolute-Value Form)
     - ``green``, ``nir``, ``swir1``
     - ``|nir - green| / |swir1 - green|``
     - Baloloy et al. (2020)
   * - ``MI``
     - Mangrove Index
     - ``nir``, ``swir1``
     - ``((nir - swir1) / (nir * swir1)) * 10000``
     - Winarso & Purwanto (2014)
   * - ``AMMI``
     - Automatic Mangrove Map and Index
     - ``red``, ``nir``, ``swir1``
     - ``((nir - red) / (red + swir1)) * ((nir - swir1) / (swir1 - 0.65 * red))``
     - Suyarso (2022); Suyarso & Avianto (2022)
   * - ``IMFI``
     - Inundated Mangrove Forest Index
     - ``blue``, ``green``, ``nir``
     - ``(blue + green - 2 * nir) / (blue + green + 2 * nir)``
     - Jia (2014); Jia et al. (2018)
   * - ``MDI1``
     - Mangrove Discrimination Index 1
     - ``nir``, ``swir1``
     - ``(nir - swir1) / swir1``
     - Wang et al. (2018)
   * - ``MDI2``
     - Mangrove Discrimination Index 2
     - ``nir``, ``swir2``
     - ``(nir - swir2) / swir2``
     - Wang et al. (2018)
   * - ``WFI``
     - Wetland Forest Index
     - ``red``, ``nir``, ``swir2``
     - ``(nir - red) / swir2``
     - Wang et al. (2018)
   * - ``FDI``
     - Forest Discrimination Index
     - ``green``, ``red``, ``nir``
     - ``nir - (red + green)``
     - Bunting & Lucas (2006); Kamal et al. (2015)
   * - ``DNVI``
     - Discriminant Normalised Vegetation Index
     - ``swir1``, ``swir2``
     - ``(swir1 - swir2)^2 / (swir1 + swir2)``
     - Manna & Raychaudhuri (2018)
   * - ``MMRI``
     - Modular Mangrove Recognition Index
     - ``green``, ``red``, ``nir``, ``swir1``
     - ``(|MNDWI| - |NDVI|) / (|MNDWI| + |NDVI|)``
     - Diniz et al. (2019)
   * - ``MFI``
     - Mangrove Forest Index
     - ``red``, ``re1``, ``re2``, ``re3``, ``nir_narrow``, ``swir2``
     - ``0.25 * sum_i(rho_i - (swir2 + (red - swir2) * (2190 - lambda_i) / 1525))``
     - Jia et al. (2019); default Sentinel-2 wavelengths ``665, 705, 740, 783, 865, 2190`` nm
   * - ``NIMI``
     - Normalised Intertidal Mangrove Index
     - ``red``, ``re2``, ``re3``, ``nir``
     - ``(3 * red - (re2 + re3 + nir)) / (3 * red + re2 + re3 + nir)``
     - Xu et al. (2020)
   * - ``EMVI``
     - Enhanced Mangrove Vegetation Index
     - ``green``, ``swir1``, ``swir2``
     - ``(green - swir2) / (swir1 - green)``
     - Yang et al. (2022)
   * - ``GCMI``
     - Generalised Composite Mangrove Index
     - ``blue``, ``green``, ``red``, ``nir``, ``swir1``
     - ``(A * EVI + B * GNDVI) / (C * LSWI + D * RNDWI)``
     - ``a=3.4, b=2.9, c=4.3, d=1.1`` (Xue & Qian, 2022)
   * - ``REMI``
     - Red-Edge Mangrove Index
     - ``green``, ``red``, ``red_edge``, ``swir1``
     - ``(red_edge - red) / (swir1 - green)``
     - Chen et al. (2023)
   * - ``IMI``
     - Integrated Mangrove Index
     - ``green``, ``red``, ``nir``, ``swir1``
     - ``(nir - red) / (swir1 + green)``
     - Umar & Abbas (2025)
   * - ``REMCH``
     - Red-Edge Mangrove Canopy Height Index
     - ``re1``, ``nir``, ``swir1``, ``swir2``
     - ``(nir / swir1) - ((nir / swir2) / (nir / re1)) + (nir / re1)``
     - Jamaluddin et al. (2025)

Water, Intertidal, Snow & Canopy Moisture Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 22 16 28 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NDWI_McFeeters`` (``NDWI``, ``GNDWI``)
     - Normalised Difference Water Index (McFeeters)
     - ``green``, ``nir``
     - ``(green - nir) / (green + nir)``
     - McFeeters (1996)
   * - ``NDWI_Gao`` (``NDMI``, ``NDII``, ``LSWI``)
     - Normalised Difference Water / Moisture / Infrared / Land Surface Water Index
     - ``nir``, ``swir1``
     - ``(nir - swir1) / (nir + swir1)``
     - Hardisky et al. (1983); Gao (1996); Xiao et al. (2002); Wilson & Sader (2002)
   * - ``MNDWI`` (``GMNDWI``, ``NDSI``)
     - Modified Normalised Difference Water / Snow Index
     - ``green``, ``swir1``
     - ``(green - swir1) / (green + swir1)``
     - Salomonson & Appel (2004); Xu (2006)
   * - ``NDGI`` (``NGRDI``)
     - Normalised Difference Glacier Index
     - ``green``, ``red``
     - ``(green - red) / (green + red)``
     - Keshri et al. (2009)
   * - ``AWEI_nsh``
     - Automated Water Extraction Index (Non-Shadow)
     - ``green``, ``nir``, ``swir1``, ``swir2``
     - ``4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)``
     - Feyisa et al. (2014), non-shadow
   * - ``AWEI_sh``
     - Automated Water Extraction Index (Shadow)
     - ``blue``, ``green``, ``nir``, ``swir1``, ``swir2``
     - ``blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2``
     - Feyisa et al. (2014), shadow
   * - ``WI2015``
     - Water Index 2015
     - ``green``, ``red``, ``nir``, ``swir1``, ``swir2``
     - ``1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir1 - 71 * swir2``
     - Fisher et al. (2016)
   * - ``WBI_BlueNIR`` (``WBI``)
     - Water Band Index (Blue/NIR Ratio)
     - ``blue``, ``nir``
     - ``blue / nir``
     - RSGISLib Blue/NIR ratio
   * - ``WRI``
     - Water Ratio Index
     - ``green``, ``red``, ``nir``, ``swir1``
     - ``(green + red) / (nir + swir1)``
     - Shen & Li (2010)
   * - ``RNDWI``
     - Revised Normalised Difference Water Index
     - ``red``, ``swir1``
     - ``(red - swir1) / (red + swir1)``
     - Cao et al. (2008)

Burn, Soil, Built-up, Cloud & Mineral Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 23 16 28 18

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NBR`` (``NBRI``)
     - Normalised Burn Ratio / Normalised Burn Ratio Index
     - ``nir``, ``swir2``
     - ``(nir - swir2) / (nir + swir2)``
     - Key & Benson (2006)
   * - ``NBR2``
     - Normalised Burn Ratio 2
     - ``swir1``, ``swir2``
     - ``(swir1 - swir2) / (swir1 + swir2)``
     - USGS NBR2
   * - ``BAI``
     - Burn Area Index
     - ``red``, ``nir``
     - ``1 / ((0.1 - red)^2 + (0.06 - nir)^2)``
     - Chuvieco et al. (2002)
   * - ``BSI``
     - Bare Soil Index
     - ``blue``, ``red``, ``nir``, ``swir1``
     - ``((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))``
     - Rikimaru et al. (2002)
   * - ``NDBSI``
     - Normalised Difference Bare Soil Index
     - ``blue``, ``green``, ``red``, ``nir``, ``swir1``
     - ``v = red - green``; ``r = 1 - (swir1 - nir) / (3 * |nir - red|)``; ``k = v * r``; ``-|swir1 - blue| / (swir1 + blue)`` if ``k < 0`` else ``(swir1 - blue) / (swir1 + blue)``
     - Liu et al. (2022)
   * - ``NDBI``
     - Normalised Difference Built-up Index
     - ``nir``, ``swir1``
     - ``(swir1 - nir) / (swir1 + nir)``
     - Zha et al. (2003)
   * - ``Brightness``
     - Visible Mean Brightness
     - ``blue``, ``green``, ``red``
     - ``(blue + green + red) / 3``
     - Unscaled mean visible reflectance
   * - ``Brightness_Scaled``
     - Standard-Deviation Stretched Visible Brightness
     - ``blue``, ``green``, ``red``
     - ``clip((Brightness - (mean - k * std)) / (2 * k * std), 0, 1)``
     - ``stretch_std_devs=2.0``
   * - ``Whiteness``
     - Cloud Whiteness Index
     - ``blue``, ``green``, ``red``
     - ``(|blue - mean| + |green - mean| + |red - mean|) / mean``, ``mean = (blue + green + red) / 3``
     - Gomez-Chova et al. (2007)
   * - ``SI``
     - Shadow Index
     - ``blue``, ``green``, ``red``
     - ``((1 - blue) * (1 - green) * (1 - red))^(1/3)``
     - Rikimaru et al. (2002)
   * - ``CMR``
     - Clay Minerals Ratio
     - ``swir1``, ``swir2``
     - ``swir1 / swir2``
     - Dogan (2009)
   * - ``FMR``
     - Ferrous Minerals Ratio
     - ``nir``, ``swir1``
     - ``swir1 / nir``
     - Segal (1982)
   * - ``IOR``
     - Iron Oxide Ratio
     - ``blue``, ``red``
     - ``red / blue``
     - Segal (1982)

Implemented SAR Indices & Equations
-----------------------------------

:func:`~rsgispy.imagecalc.calcindices.calc_sar_indices` calculates SAR
polarimetric backscatter indices from multi-band SAR imagery whose physical
values (after ``physical = (dn - in_offset) / in_gain``) are supplied in either
decibels (``in_units="dB"``) or linear power (``in_units="power"``).
Canonical polarisation band names accepted in ``band_names`` are ``vv``,
``vh``, ``hh``, and ``hv`` (case-insensitive; prefixes ``sigma0_*``,
``gamma0_*``, and ``beta0_*`` are also recognised).

Automatic Unit Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

* Indices defined in **linear power** (``power``) automatically convert
  decibel inputs via ``power = 10 ** (dB / 10)``.
* Indices defined in **decibels** (``dB``) automatically convert linear-power
  inputs via ``dB = 10 * log10(power)`` (pixels with ``power <= 0`` are masked
  as ``out_no_data_val``).
* Total-power ``Span_*_dB`` indices sum polarisations in linear power first
  and then convert the sum to dB via ``10 * log10(span)``.

SAR Vegetation, Canopy Volume & Forest Degradation Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 21 11 9 25 16

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Domain
     - Equation
     - Reference / Notes
   * - ``RVI`` (``RVI_Quad``)
     - Quad-Pol Radar Vegetation Index
     - ``hh``, ``vv``, ``hv``
     - ``power``
     - ``(8 * hv) / (hh + vv + 2 * hv)``
     - Kim & van Zyl (2000, 2009)
   * - ``RVI_VV_VH`` (``DpRVI_VV``, ``RVI4S1``)
     - Dual-Pol Radar Vegetation Index (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``(4 * vh) / (vv + vh)``
     - Charbonneau et al. (2005); Nasirzadehdizaji et al. (2019)
   * - ``RVI_HH_HV`` (``DpRVI_HH``)
     - Dual-Pol Radar Vegetation Index (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``(4 * hv) / (hh + hv)``
     - Charbonneau et al. (2005); Kim et al. (2012)
   * - ``DpRVIc_VV_VH`` (``DpRVIc``)
     - GRD Dual-Pol Radar Vegetation Index (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``(q * (q + 3)) / (q + 1)^2``, ``q = vh / vv``
     - Mandal et al. (2020); Bhogapurapu et al. (2022)
   * - ``DpRVIc_HH_HV``
     - GRD Dual-Pol Radar Vegetation Index (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``(q * (q + 3)) / (q + 1)^2``, ``q = hv / hh``
     - HH/HV analogue of ``DpRVIc``
   * - ``DPSVI``
     - Dual-Polarisation SAR Vegetation Index
     - ``vv``, ``vh``
     - ``power``
     - ``(((vv_max - vv) + vh) / sqrt(2)) * ((vv + vh) / vv) * vh``
     - Periasamy (2018); ``vv_max`` auto-computed or via ``index_params``
   * - ``DPSVIm``
     - Modified Dual-Pol SAR Vegetation Index
     - ``vv``, ``vh``
     - ``power``
     - ``(vv^2 + vv * vh) / sqrt(2)``
     - Dos Santos et al. (2021)
   * - ``VDDPI``
     - Vertical Dual Depolarisation Index
     - ``vv``, ``vh``
     - ``power``
     - ``(vv + vh) / vv``
     - Periasamy (2018)
   * - ``HDDPI``
     - Horizontal Dual Depolarisation Index
     - ``hh``, ``hv``
     - ``power``
     - ``(hh + hv) / hh``
     - HH/HV analogue of ``VDDPI``
   * - ``RFDI`` (``RFDI_HH_HV``, ``NDPI_HH_HV``, ``NDPDI_HH_HV``, ``NDI_HH_HV``)
     - Radar Forest Degradation / Normalised Difference Polarisation Index (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``(hh - hv) / (hh + hv)``
     - Saatchi et al. (2011); Mitchard et al. (2012); Saatchi (2019)
   * - ``RFDI_VV_VH`` (``NDPI_VV_VH``, ``NDPDI``, ``NDPDI_VV_VH``, ``NDI_VV_VH``)
     - Radar Forest Degradation / Normalised Difference Polarisation Index (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``(vv - vh) / (vv + vh)``
     - Mitchard et al. (2012); Sentinel-1 RFDI / NDPDI
   * - ``NDI_VH_VV`` (``NRPB``)
     - Normalised Difference Index / NRPB (VH/VV)
     - ``vv``, ``vh``
     - ``power``
     - ``(vh - vv) / (vh + vv)``
     - Filgueiras et al. (2019)
   * - ``NDI_HV_HH``
     - Normalised Difference Index (HV/HH)
     - ``hh``, ``hv``
     - ``power``
     - ``(hv - hh) / (hv + hh)``
     - Cross-minus-co HH/HV normalised difference
   * - ``BMI``
     - Biomass Index
     - ``hh``, ``vv``
     - ``power``
     - ``(hh + vv) / 2``
     - Pope et al. (1994)
   * - ``CSI``
     - Canopy Structure Index
     - ``hh``, ``vv``
     - ``power``
     - ``vv / (vv + hh)``
     - Pope et al. (1994)
   * - ``VSI``
     - Volume Scattering Index
     - ``hh``, ``vv``, ``hv``
     - ``power``
     - ``hv / (hv + (hh + vv) / 2)``
     - Pope et al. (1994)

SAR Water Detection Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 22 12 10 24 16

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Domain
     - Equation
     - Reference / Notes
   * - ``SDWI`` (``SDWI_VV_VH``)
     - Sentinel-1 Dual-Polarised Water Index
     - ``vv``, ``vh``
     - ``power``
     - ``ln(10 * vv * vh) - 8``
     - Jia et al. (2019)
   * - ``SDWI_HH_HV``
     - SAR Dual-Polarised Water Index (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``ln(10 * hh * hv) - 8``
     - HH/HV analogue of ``SDWI``

SAR Polarisation Ratios, Differences & Summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 22 12 10 24 16

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Domain
     - Equation
     - Reference / Notes
   * - ``CR_VH_VV`` (``PR_VH_VV``)
     - Cross-Polarisation Ratio (VH/VV)
     - ``vv``, ``vh``
     - ``power``
     - ``vh / vv``
     - Rosenqvist et al. (2007); Veloso et al. (2017); Bunting et al. (2018)
   * - ``CR_VV_VH`` (``PR_VV_VH``)
     - Co-to-Cross Polarisation Ratio (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``vv / vh``
     - Linear co/cross ratio
   * - ``CR_HV_HH`` (``PR_HV_HH``)
     - Cross-Polarisation Ratio (HV/HH)
     - ``hh``, ``hv``
     - ``power``
     - ``hv / hh``
     - Rosenqvist et al. (2007); Bunting et al. (2018)
   * - ``CR_HH_HV`` (``PR_HH_HV``)
     - Co-to-Cross Polarisation Ratio (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``hh / hv``
     - Linear co/cross ratio
   * - ``PR_HH_VV``
     - Co-Polarisation Ratio (HH/VV)
     - ``hh``, ``vv``
     - ``power``
     - ``hh / vv``
     - Linear co-pol ratio
   * - ``PR_VV_HH``
     - Co-Polarisation Ratio (VV/HH)
     - ``hh``, ``vv``
     - ``power``
     - ``vv / hh``
     - Linear co-pol ratio
   * - ``Pol_Diff_VV_VH``
     - Linear Polarisation Difference (VV - VH)
     - ``vv``, ``vh``
     - ``power``
     - ``vv - vh``
     - Linear backscatter difference
   * - ``Pol_Diff_HH_HV``
     - Linear Polarisation Difference (HH - HV)
     - ``hh``, ``hv``
     - ``power``
     - ``hh - hv``
     - Linear backscatter difference
   * - ``CR_VH_VV_dB`` (``PR_VH_VV_dB``)
     - Cross-Polarisation Ratio in dB (VH - VV)
     - ``vv``, ``vh``
     - ``dB``
     - ``vh_dB - vv_dB``
     - ``10 * log10(vh / vv)`` (Veloso et al., 2017)
   * - ``CR_VV_VH_dB`` (``PR_VV_VH_dB``)
     - Co-to-Cross Polarisation Ratio in dB (VV - VH)
     - ``vv``, ``vh``
     - ``dB``
     - ``vv_dB - vh_dB``
     - ``10 * log10(vv / vh)``
   * - ``CR_HV_HH_dB`` (``PR_HV_HH_dB``)
     - Cross-Polarisation Ratio in dB (HV - HH)
     - ``hh``, ``hv``
     - ``dB``
     - ``hv_dB - hh_dB``
     - ``10 * log10(hv / hh)``
   * - ``CR_HH_HV_dB`` (``PR_HH_HV_dB``)
     - Co-to-Cross Polarisation Ratio in dB (HH - HV)
     - ``hh``, ``hv``
     - ``dB``
     - ``hh_dB - hv_dB``
     - ``10 * log10(hh / hv)``
   * - ``PR_HH_VV_dB``
     - Co-Polarisation Ratio in dB (HH - VV)
     - ``hh``, ``vv``
     - ``dB``
     - ``hh_dB - vv_dB``
     - ``10 * log10(hh / vv)``
   * - ``PR_VV_HH_dB``
     - Co-Polarisation Ratio in dB (VV - HH)
     - ``hh``, ``vv``
     - ``dB``
     - ``vv_dB - hh_dB``
     - ``10 * log10(vv / hh)``
   * - ``Mean_VV_VH_dB``
     - Mean Dual-Pol Backscatter in dB (VV/VH)
     - ``vv``, ``vh``
     - ``dB``
     - ``(vv_dB + vh_dB) / 2``
     - Mean dB backscatter
   * - ``Mean_HH_HV_dB``
     - Mean Dual-Pol Backscatter in dB (HH/HV)
     - ``hh``, ``hv``
     - ``dB``
     - ``(hh_dB + hv_dB) / 2``
     - Mean dB backscatter

SAR Total Backscatter Power (Span) Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 22 12 10 24 16

   * - Index (Aliases)
     - Full Name
     - Required Bands
     - Domain
     - Equation
     - Reference / Notes
   * - ``Span_VV_VH``
     - Dual-Pol Total Power Span (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``vv + vh``
     - Total linear power
   * - ``Span_VV_VH_dB``
     - Dual-Pol Total Power Span in dB (VV/VH)
     - ``vv``, ``vh``
     - ``power``
     - ``10 * log10(vv + vh)``
     - Summed in power, reported in dB
   * - ``Span_HH_HV``
     - Dual-Pol Total Power Span (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``hh + hv``
     - Total linear power
   * - ``Span_HH_HV_dB``
     - Dual-Pol Total Power Span in dB (HH/HV)
     - ``hh``, ``hv``
     - ``power``
     - ``10 * log10(hh + hv)``
     - Summed in power, reported in dB
   * - ``Span_Quad``
     - Quad-Pol Total Power Span
     - ``hh``, ``vv``, ``hv``
     - ``power``
     - ``hh + vv + 2 * hv``
     - Total quad-pol linear power
   * - ``Span_Quad_dB``
     - Quad-Pol Total Power Span in dB
     - ``hh``, ``vv``, ``hv``
     - ``power``
     - ``10 * log10(hh + vv + 2 * hv)``
     - Summed in power, reported in dB
