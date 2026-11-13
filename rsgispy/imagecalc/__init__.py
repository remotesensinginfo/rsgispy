"""
The imagecalc subpackage provides modules for raster mathematical expressions,
summary statistics, image comparison, categorical encoding, rescaling, distance
transforms, region filling, percentiles, and windowed point sampling using
``rasterio``:

* :mod:`rsgispy.imagecalc.calc_pt_win_smpls` — windowed point sampling calculators
* :mod:`rsgispy.imagecalc.comparison` — difference, equality, correlation, MI, and EMD
* :mod:`rsgispy.imagecalc.distance` — proximity distance transforms and buffering
* :mod:`rsgispy.imagecalc.encoding` — integer recoding and one-hot category encoding
* :mod:`rsgispy.imagecalc.fill` — KNN spatial region filling
* :mod:`rsgispy.imagecalc.maths` — muParser band/image math and area under curve
* :mod:`rsgispy.imagecalc.percentiles` — band and per-pixel percentile computations
* :mod:`rsgispy.imagecalc.rescale` — gain/offset and min-max pixel rescaling
* :mod:`rsgispy.imagecalc.stats` — summary statistics, column stats, and value probabilities
"""
