rsgispy.droneutils
==================

The :mod:`rsgispy.droneutils` module provides utilities for organising drone
photography by time, creating photo point and flightline vector layers, and
writing external GPS/orientation metadata into image EXIF/XMP headers.

Example
-------

Grouping drone images into flight folders by acquisition timestamp:

.. code-block:: python

   import rsgispy.droneutils
   import rsgispy.tools.files

   imgs = rsgispy.tools.files.find_files_ext("./drone_photos", ".JPG")
   rsgispy.droneutils.split_photos_by_time(
       input_imgs=imgs,
       output_base_dir="./flights",
       time_split_secs=120,
   )

API Reference
-------------

.. automodule:: rsgispy.droneutils
   :members:
   :undoc-members:
   :show-inheritance:
