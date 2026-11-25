Gemini Skill for AI-Assisted Coding
====================================

``rsgispy`` ships an official `Agent Skill <https://geminicli.com/docs/cli/skills/>`_
under ``skills/rsgispy/`` in the repository so Gemini can generate idiomatic,
accurate ``rsgispy`` Python code, calculate optical and SAR spectral indices,
write multi-output ``pymu-eval`` expressions, and translate legacy RSGISLib
scripts without inventing invalid function names.

What the Skill Provides
-----------------------

- **API routing & conventions** (``SKILL.md``): pure-Python imports (no
  ``osgeo`` or ``rsgislib``), 1-based band indexing, ``rsgispy.TYPE_*`` and
  ``INTERP_*`` constants, keyword-only output options, and RSGISLib migration
  guidance.
- **Workflow guides** (``SKILL.md``): a table mapping common analyses to the
  guides in this documentation (for example
  :doc:`guides/pixel_classification_workflow` and
  :doc:`guides/hydrology_workflow`), so generated code follows the same
  tested step order, plus workflow pitfalls found while writing the guides.
- **Progressive-disclosure references** (``skills/rsgispy/references/``):

  - ``raster_and_indices.md`` — ``rsgispy.BBox``, ``rsgispy.imageprocess``
    (including ``TileGraph``), ``band_maths`` / ``image_band_math`` (single
    and multi-output ``pymu-eval`` expressions), ``calc_indices`` (97 optical
    indices), ``calc_sar_indices`` (63 SAR indices with automatic dB/power
    conversion), ``imagecalc.*``, and all 18 ``imageutils.*`` submodules.
  - ``calibration_registration_and_spectral.md`` — ``imagecalibration``
    (radiometry, solar angles, Landsat C2 L1/L2, Sentinel-2 L2A, CloudSEN12
    cloud masking), ``imageregistration`` (``shift``, ``tiepoints``,
    ``warping``), ``changedetect``, and ``imagespectral``.
  - ``elevation_segmentation_and_filters.md`` — ``elevation`` (terrain,
    illumination, filtering, fusion, roughness, hydrology & flood indices),
    ``segmentation`` (Shepherd, region-merge, SLIC, graph, watershed, clumping),
    ``imagefilter`` (stats, smooth, edge, SAR speckle, GLCM, ``FilterStep``
    chains), and ``imagemorphology``.
  - ``vector_and_zonalstats.md`` — ``vectorprocess``, ``rsgispy.vector``
    (``io``, ``create``, ``geoms``, ``attrs`` incl. ``vector_maths``,
    ``stats``, ``selection``), and ``rsgispy.zonalstats`` (polygon and point).
  - ``ml_classification_regression_and_sdm.md`` — ``rsgispy.ml``
    (``ImageFeatureStack``, ``FeatureScaler``, spatial CV, Optuna tuning,
    tiled inference), ``rsgispy.classification`` (sklearn/GBDT/clustering,
    accuracy metrics, post-processing), ``rsgispy.regression``, and
    ``rsgispy.modelling.species_distribution``.
  - ``tools_mapping_dataaccess_and_cli.md`` — ``rsgispy.mapping``,
    ``rsgispy.dataaccess``, ``rsgispy.droneutils``, all 29
    ``rsgispy.tools.*`` modules, and all 11 ``rsgispy.cli`` tools.

Installing in Gemini CLI
------------------------

Install Directly from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the skill globally (available in all your projects under
``~/.gemini/skills/rsgispy``):

.. code-block:: bash

   gemini skills install https://github.com/remotesensinginfo/rsgispy.git --path skills/rsgispy

To install the skill only for your current project/workspace
(``.gemini/skills/rsgispy``):

.. code-block:: bash

   gemini skills install https://github.com/remotesensinginfo/rsgispy.git --path skills/rsgispy --scope workspace

.. tip::

   Pass ``--consent`` to skip the interactive confirmation prompt in scripts or
   automated setups.

Link from a Local ``rsgispy`` Clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a local clone of ``rsgispy`` (for example, an editable
development checkout) and want edits to stay synced automatically:

.. code-block:: bash

   # Global symlink (~/.gemini/skills/rsgispy)
   gemini skills link ./skills/rsgispy

   # Or project-local symlink (.gemini/skills/rsgispy)
   gemini skills link ./skills/rsgispy --scope workspace

Uploading to Gemini Online (``gemini.google.com``)
--------------------------------------------------

You can also upload the ``rsgispy`` skill directly into the Gemini web app at
`https://gemini.google.com/customize/skills <https://gemini.google.com/customize/skills>`_
(or via the sidebar under **Customize** → **Skills**).

Prepare the Skill Package
~~~~~~~~~~~~~~~~~~~~~~~~~

Gemini online accepts either:

- the uncompressed ``skills/rsgispy`` folder (containing ``SKILL.md`` at its
  root alongside ``references/``),
- a clean ``.zip`` archive with ``SKILL.md`` at the root of the archive, or
- the standalone ``skills/rsgispy/SKILL.md`` file (with reference files added
  afterward).

If creating a ``.zip`` archive from a local clone of ``rsgispy``, package the
contents of ``skills/rsgispy`` so ``SKILL.md`` sits at the root of the archive
and exclude hidden OS metadata files (such as ``.DS_Store`` or ``__MACOSX``):

.. code-block:: bash

   cd skills/rsgispy
   zip -r ../../rsgispy-skill.zip SKILL.md references/ -x "*.DS_Store" "*__MACOSX*"

.. note::

   If you download the whole ``rsgispy`` repository as a ``.zip`` from GitHub,
   extract it first and upload only the ``skills/rsgispy`` folder (or re-zip
   its contents as shown above). Uploading a repository archive where
   ``SKILL.md`` is nested inside subdirectories will fail validation.

Upload Steps
~~~~~~~~~~~~

1. Open `https://gemini.google.com/customize/skills <https://gemini.google.com/customize/skills>`_
   in your browser (signed in to an eligible Gemini account).
2. Click **Create** → **Upload** (or **Upload**).
3. Select the ``rsgispy-skill.zip`` archive, the uncompressed
   ``skills/rsgispy`` folder, or ``skills/rsgispy/SKILL.md``.
4. Review the parsed skill name (``rsgispy``), description, and reference
   files, then click **Create**.

Using & Updating the Uploaded Skill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Invoke in chat:** Once enabled, Gemini automatically activates ``rsgispy``
  when relevant to your prompt, or you can invoke it explicitly in any chat by
  typing ``@rsgispy`` or ``/`` and selecting ``rsgispy``.
- **Update later:** Open the **More** (``...``) menu next to the skill on
  `https://gemini.google.com/customize/skills <https://gemini.google.com/customize/skills>`_
  and choose **Replace skill** to upload a newer version.
- **Troubleshooting uploads:** Ensure hidden files (``.DS_Store``,
  ``__MACOSX``) are excluded, keep ``SKILL.md`` at the root of the folder or
  ``.zip``, or upload the uncompressed ``skills/rsgispy`` folder directly if a
  ``.zip`` upload stalls on *"Saving metadata..."*.

Manual Installation (Gemini CLI / Antigravity / Agents Standard)
----------------------------------------------------------------

If your Gemini client does not expose the ``gemini skills`` subcommand (or you
prefer managing files directly), copy or symlink ``skills/rsgispy`` into the
appropriate skills directory:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Scope
     - Standard Agents Path (Precedence)
     - Gemini Path
   * - User (all projects)
     - ``~/.agents/skills/rsgispy/``
     - ``~/.gemini/skills/rsgispy/``
   * - Workspace (current repo)
     - ``.agents/skills/rsgispy/``
     - ``.gemini/skills/rsgispy/``

For example:

.. code-block:: bash

   mkdir -p ~/.gemini/skills
   cp -r skills/rsgispy ~/.gemini/skills/rsgispy

Verifying & Managing the Skill
------------------------------

From the terminal:

.. code-block:: bash

   # List discovered skills and status
   gemini skills list

   # Enable or disable the skill
   gemini skills enable rsgispy
   gemini skills disable rsgispy

   # Remove the installed skill
   gemini skills uninstall rsgispy

Inside an active Gemini CLI session, use slash commands to inspect or reload
skills without restarting:

.. code-block:: text

   /skills list
   /skills reload

Example Prompts
---------------

Once ``rsgispy`` is active in Gemini, you can ask questions such as:

- *"Write a Python script using rsgispy to calculate NDVI, NDWI_McFeeters, MVI, and NDMaI from a scaled Sentinel-2 GeoTIFF and save as int16."*
- *"Use rsgispy.imagecalc.calcindices.calc_sar_indices to compute RVI_VV_VH, DpRVIc_VV_VH, and CR_VH_VV_dB from a Sentinel-1 dB image."*
- *"Use band_maths to scale four Sentinel-2 bands and compute NDVI and EVI in a single multi-band pass."*
- *"Convert this RSGISLib classification script to rsgispy."*
