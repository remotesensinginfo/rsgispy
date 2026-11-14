rsgispy.imagecalc.calcindices
=============================

.. automodule:: rsgispy.imagecalc.calcindices
   :members:
   :undoc-members:
   :show-inheritance:

Implemented Spectral Indices & Equations
----------------------------------------

All equations below operate on physical input band values computed via
``physical = (dn - in_offset) / in_gain``. Canonical band names accepted in
``band_names`` are ``blue``, ``green``, ``red``, ``red_edge`` (alias ``re``),
``nir``, ``swir1`` (alias ``swir``), and ``swir2``.

Naming Disambiguation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Water indices**: ``NDWI_McFeeters`` (also ``NDWI`` and ``GNDWI``) implements
  McFeeters (1996) ``(green - nir) / (green + nir)``; ``NDWI_Gao`` (also
  ``NDMI``) implements Gao (1996) ``(nir - swir1) / (nir + swir1)`` (note that
  ``rsgislib.imagecalc.calc_ndwi`` historically used Gao's NIR/SWIR1 formula);
  ``MNDWI`` (also ``GMNDWI`` and ``NDSI``) implements Xu (2006)
  ``(green - swir1) / (green + swir1)``.
* **Water Band Ratio**: ``WBI_BlueNIR`` (alias ``WBI``) implements the
  multispectral RSGISLib ``blue / nir`` ratio, distinct from the narrowband
  hyperspectral Water Band Index (``R900 / R970``).
* **Ratio Vegetation Index**: ``SR``, ``RVI``, and ``RVI_NIR_Red`` compute
  ``nir / red``, whereas ``RVI_Red_NIR`` computes ``red / nir``.

Vegetation & Canopy Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 32 18

   * - Index
     - Full Name
     - Required Bands
     - Equation
     - Defaults / Notes
   * - ``NDVI``
     - Normalised Difference Vegetation Index
     - ``red``, ``nir``
     - ``(nir - red) / (nir + red)``
     - Rouse et al. (1974)
   * - ``EVI``
     - Enhanced Vegetation Index
     - ``blue``, ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C1 * red - C2 * blue + L))``
     - ``G=2.5, C1=6.0, C2=7.5, L=1.0``
   * - ``EVI2``
     - Two-Band Enhanced Vegetation Index
     - ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C * red + L))``
     - ``G=2.5, C=2.4, L=1.0``
   * - ``GNDVI``
     - Green Normalised Difference Vegetation Index
     - ``green``, ``nir``
     - ``(nir - green) / (nir + green)``
     - Gitelson et al. (1996)
   * - ``SAVI``
     - Soil Adjusted Vegetation Index
     - ``red``, ``nir``
     - ``((nir - red) / (nir + red + L)) * (1 + L)``
     - ``L=0.5`` (Huete 1988)
   * - ``OSAVI``
     - Optimised Soil Adjusted Vegetation Index
     - ``red``, ``nir``
     - ``(1 + Y) * ((nir - red) / (nir + red + Y))``
     - ``Y=0.16`` (Rondeaux et al. 1996)
   * - ``MSAVI2`` (``MSAVI``)
     - Modified Soil Adjusted Vegetation Index 2 / Modified Soil Adjusted Vegetation Index
     - ``red``, ``nir``
     - ``(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2``
     - Qi et al. (1994)
   * - ``SATVI``
     - Soil Adjusted Total Vegetation Index
     - ``red``, ``swir1``, ``swir2``
     - ``((swir1 - red) / (swir1 + red + L)) * (1 + L) - (swir2 / 2)``
     - ``L=0.5`` (Marsett et al. 2006)
   * - ``SR`` (``RVI``, ``RVI_NIR_Red``)
     - Simple Ratio / Ratio Vegetation Index (NIR/Red)
     - ``red``, ``nir``
     - ``nir / red``
     - Birth & McVey (1968)
   * - ``RVI_Red_NIR``
     - Ratio Vegetation Index (Red/NIR)
     - ``red``, ``nir``
     - ``red / nir``
     - Inverse ratio convention
   * - ``DVI``
     - Difference Vegetation Index
     - ``red``, ``nir``
     - ``nir - red``
     - Richardson & Wiegand (1977)
   * - ``WDVI``
     - Weighted Difference Vegetation Index
     - ``red``, ``nir``
     - ``nir - s * red``
     - ``s=1.0`` (Richardson & Wiegand 1977)
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
     - ``gamma=1.0`` (Kaufman & Tanre 1992)
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
   * - ``NDRE``
     - Normalised Difference Red-Edge Index
     - ``red_edge``, ``nir``
     - ``(nir - red_edge) / (nir + red_edge)``
     - Barnes et al. (2000)
   * - ``CIre``
     - Chlorophyll Index Red-Edge
     - ``red_edge``, ``nir``
     - ``(nir / red_edge) - 1``
     - Gitelson et al. (2003)
   * - ``CIgreen``
     - Chlorophyll Index Green
     - ``green``, ``nir``
     - ``(nir / green) - 1``
     - Gitelson et al. (2003)
   * - ``NPCRI``
     - Normalised Pigment Chlorophyll Ratio Index
     - ``blue``, ``red``
     - ``(red - blue) / (red + blue)``
     - Penuelas et al. (1994)
   * - ``REMI``
     - Red-Edge Mangrove Index
     - ``green``, ``red``, ``red_edge``, ``swir1``
     - ``(red_edge - red) / (swir1 + green)``
     - Chen et al. (2023)
   * - ``MVI``
     - Mangrove Vegetation Index
     - ``green``, ``nir``, ``swir1``
     - ``(nir - green) / (swir1 - green)``
     - Baloloy et al. (2020)
   * - ``VARI``
     - Visible Atmospherically Resistant Index
     - ``blue``, ``green``, ``red``
     - ``(green - red) / (green + red - blue)``
     - Gitelson et al. (2002)
   * - ``GLI``
     - Green Leaf Index
     - ``blue``, ``green``, ``red``
     - ``(2 * green - red - blue) / (2 * green + red + blue)``
     - Louhaichi et al. (2001)
   * - ``ExG``
     - Excess Green Index
     - ``blue``, ``green``, ``red``
     - ``2 * green - red - blue``
     - Woebbecke et al. (1995)

Water, Snow & Moisture Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 23 16 28 18

   * - Index
     - Full Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NDWI_McFeeters`` (``NDWI``, ``GNDWI``)
     - Normalised Difference Water Index (McFeeters) / Green Normalised Difference Water Index
     - ``green``, ``nir``
     - ``(green - nir) / (green + nir)``
     - McFeeters (1996)
   * - ``NDWI_Gao`` (``NDMI``)
     - Normalised Difference Water Index (Gao) / Normalised Difference Moisture Index
     - ``nir``, ``swir1``
     - ``(nir - swir1) / (nir + swir1)``
     - Gao (1996)
   * - ``MNDWI`` (``GMNDWI``)
     - Modified Normalised Difference Water Index / Green Modified Normalised Difference Water Index
     - ``green``, ``swir1``
     - ``(green - swir1) / (green + swir1)``
     - Xu (2006)
   * - ``NDSI``
     - Normalised Difference Snow Index
     - ``green``, ``swir1``
     - ``(green - swir1) / (green + swir1)``
     - Salomonson & Appel (2004)
   * - ``NDGI``
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

Burn, Soil, Built-up, Cloud & Mineral Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 23 16 28 18

   * - Index
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
