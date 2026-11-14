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
   :widths: 16 20 44 20

   * - Index Name
     - Required Bands
     - Equation
     - Defaults / Notes
   * - ``NDVI``
     - ``red``, ``nir``
     - ``(nir - red) / (nir + red)``
     - Rouse et al. (1974)
   * - ``EVI``
     - ``blue``, ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C1 * red - C2 * blue + L))``
     - ``G=2.5, C1=6.0, C2=7.5, L=1.0``
   * - ``EVI2``
     - ``red``, ``nir``
     - ``G * ((nir - red) / (nir + C * red + L))``
     - ``G=2.5, C=2.4, L=1.0``
   * - ``GNDVI``
     - ``green``, ``nir``
     - ``(nir - green) / (nir + green)``
     - Gitelson et al. (1996)
   * - ``SAVI``
     - ``red``, ``nir``
     - ``((nir - red) / (nir + red + L)) * (1 + L)``
     - ``L=0.5`` (Huete 1988)
   * - ``OSAVI``
     - ``red``, ``nir``
     - ``(1 + Y) * ((nir - red) / (nir + red + Y))``
     - ``Y=0.16`` (Rondeaux et al. 1996)
   * - ``MSAVI2`` (``MSAVI``)
     - ``red``, ``nir``
     - ``(2 * nir + 1 - sqrt((2 * nir + 1)^2 - 8 * (nir - red))) / 2``
     - Qi et al. (1994)
   * - ``SATVI``
     - ``red``, ``swir1``, ``swir2``
     - ``((swir1 - red) / (swir1 + red + L)) * (1 + L) - (swir2 / 2)``
     - ``L=0.5`` (Marsett et al. 2006)
   * - ``SR`` (``RVI``, ``RVI_NIR_Red``)
     - ``red``, ``nir``
     - ``nir / red``
     - Birth & McVey (1968)
   * - ``RVI_Red_NIR``
     - ``red``, ``nir``
     - ``red / nir``
     - Inverse ratio convention
   * - ``DVI``
     - ``red``, ``nir``
     - ``nir - red``
     - Difference Vegetation Index
   * - ``WDVI``
     - ``red``, ``nir``
     - ``nir - s * red``
     - ``s=1.0`` (Richardson & Wiegand 1977)
   * - ``NRVI``
     - ``red``, ``nir``
     - ``((red / nir) - 1) / ((red / nir) + 1)``
     - Baret & Guyot (1991)
   * - ``CTVI``
     - ``red``, ``nir``
     - ``(NDVI + 0.5) / sqrt(abs(NDVI + 0.5))``
     - Perry & Lautenschlager (1984)
   * - ``TVI``
     - ``red``, ``nir``
     - ``sqrt(NDVI + 0.5)``
     - Deering et al. (1975)
   * - ``TTVI``
     - ``red``, ``nir``
     - ``sqrt(abs(NDVI + 0.5))``
     - Thiam (1997)
   * - ``GEMI``
     - ``red``, ``nir``
     - ``eta * (1 - 0.25 * eta) - ((red - 0.125) / (1 - red))``, where ``eta = (2 * (nir^2 - red^2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)``
     - Pinty & Verstraete (1992)
   * - ``ARVI``
     - ``blue``, ``red``, ``nir``
     - ``(nir - (red - gamma * (blue - red))) / (nir + (red - gamma * (blue - red)))``
     - ``gamma=1.0`` (Kaufman & Tanre 1992)
   * - ``SLAVI``
     - ``red``, ``nir``, ``swir2``
     - ``nir / (red + swir2)``
     - Lymburner et al. (2000)
   * - ``kNDVI``
     - ``red``, ``nir``
     - ``tanh(((nir - red) / (nir + red))^2)``
     - Camps-Valls et al. (2021)
   * - ``NIRv``
     - ``red``, ``nir``
     - ``((nir - red) / (nir + red)) * nir``
     - Badgley et al. (2017)
   * - ``NDRE``
     - ``red_edge``, ``nir``
     - ``(nir - red_edge) / (nir + red_edge)``
     - Barnes et al. (2000)
   * - ``CIre``
     - ``red_edge``, ``nir``
     - ``(nir / red_edge) - 1``
     - Gitelson et al. (2003)
   * - ``CIgreen``
     - ``green``, ``nir``
     - ``(nir / green) - 1``
     - Gitelson et al. (2003)
   * - ``NPCRI``
     - ``blue``, ``red``
     - ``(red - blue) / (red + blue)``
     - Pigment Chlorophyll Ratio
   * - ``REMI``
     - ``green``, ``red``, ``red_edge``, ``swir1``
     - ``(red_edge - red) / (swir1 + green)``
     - Chen et al. (2023)
   * - ``MVI``
     - ``green``, ``nir``, ``swir1``
     - ``(nir - green) / (swir1 - green)``
     - Baloloy et al. (2020)
   * - ``VARI``
     - ``blue``, ``green``, ``red``
     - ``(green - red) / (green + red - blue)``
     - Gitelson et al. (2002)
   * - ``GLI``
     - ``blue``, ``green``, ``red``
     - ``(2 * green - red - blue) / (2 * green + red + blue)``
     - Louhaichi et al. (2001)
   * - ``ExG``
     - ``blue``, ``green``, ``red``
     - ``2 * green - red - blue``
     - Woebbecke et al. (1995)

Water, Snow & Moisture Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 40 20

   * - Index Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NDWI_McFeeters`` (``NDWI``, ``GNDWI``)
     - ``green``, ``nir``
     - ``(green - nir) / (green + nir)``
     - McFeeters (1996)
   * - ``NDWI_Gao`` (``NDMI``)
     - ``nir``, ``swir1``
     - ``(nir - swir1) / (nir + swir1)``
     - Gao (1996)
   * - ``MNDWI`` (``GMNDWI``)
     - ``green``, ``swir1``
     - ``(green - swir1) / (green + swir1)``
     - Xu (2006)
   * - ``NDSI``
     - ``green``, ``swir1``
     - ``(green - swir1) / (green + swir1)``
     - Salomonson & Appel (2004)
   * - ``NDGI``
     - ``green``, ``red``
     - ``(green - red) / (green + red)``
     - Normalised Difference Glacier Index
   * - ``AWEI_nsh``
     - ``green``, ``nir``, ``swir1``, ``swir2``
     - ``4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)``
     - Feyisa et al. (2014), non-shadow
   * - ``AWEI_sh``
     - ``blue``, ``green``, ``nir``, ``swir1``, ``swir2``
     - ``blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2``
     - Feyisa et al. (2014), shadow
   * - ``WI2015``
     - ``green``, ``red``, ``nir``, ``swir1``, ``swir2``
     - ``1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir1 - 71 * swir2``
     - Fisher et al. (2016)
   * - ``WBI_BlueNIR`` (``WBI``)
     - ``blue``, ``nir``
     - ``blue / nir``
     - RSGISLib Blue/NIR ratio

Burn, Soil, Built-up, Cloud & Mineral Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 40 20

   * - Index Name
     - Required Bands
     - Equation
     - Reference / Notes
   * - ``NBR`` (``NBRI``)
     - ``nir``, ``swir2``
     - ``(nir - swir2) / (nir + swir2)``
     - Key & Benson (2006)
   * - ``NBR2``
     - ``swir1``, ``swir2``
     - ``(swir1 - swir2) / (swir1 + swir2)``
     - USGS NBR2
   * - ``BAI``
     - ``red``, ``nir``
     - ``1 / ((0.1 - red)^2 + (0.06 - nir)^2)``
     - Chuvieco et al. (2002)
   * - ``BSI``
     - ``blue``, ``red``, ``nir``, ``swir1``
     - ``((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))``
     - Bare Soil Index
   * - ``NDBSI``
     - ``blue``, ``green``, ``red``, ``nir``, ``swir1``
     - ``v = red - green``; ``r = 1 - (swir1 - nir) / (3 * |nir - red|)``; ``k = v * r``; ``-|swir1 - blue| / (swir1 + blue)`` if ``k < 0`` else ``(swir1 - blue) / (swir1 + blue)``
     - Liu et al. (2022)
   * - ``NDBI``
     - ``nir``, ``swir1``
     - ``(swir1 - nir) / (swir1 + nir)``
     - Zha et al. (2003)
   * - ``Brightness``
     - ``blue``, ``green``, ``red``
     - ``(blue + green + red) / 3``
     - Visible mean brightness
   * - ``Brightness_Scaled``
     - ``blue``, ``green``, ``red``
     - ``clip((Brightness - (mean - k * std)) / (2 * k * std), 0, 1)``
     - ``stretch_std_devs=2.0``
   * - ``Whiteness``
     - ``blue``, ``green``, ``red``
     - ``(|blue - mean| + |green - mean| + |red - mean|) / mean``, ``mean = (blue + green + red) / 3``
     - Gomez-Chova et al. (2007)
   * - ``SI``
     - ``blue``, ``green``, ``red``
     - ``((1 - blue) * (1 - green) * (1 - red))^(1/3)``
     - Shadow Index
   * - ``CMR``
     - ``swir1``, ``swir2``
     - ``swir1 / swir2``
     - Dogan (2009)
   * - ``FMR``
     - ``nir``, ``swir1``
     - ``swir1 / nir``
     - Segal (1982)
   * - ``IOR``
     - ``blue``, ``red``
     - ``red / blue``
     - Segal (1982)
