# rsgispy tests

Run the full suite from the repository root:

```bash
MPLBACKEND=Agg python -m pytest -n 8 -q -p no:cacheprovider
```

## Golden reference files

`tests/data/golden/*.npz` holds reference outputs for the file-based raster
functions that were rebuilt on the tile graph. The test checks the current
code against them, usually at several block sizes, including sizes that do not
divide the test image. Integers must match exactly. Floats must match within
1 ulp, except `imagefilter_*`, which falls back to a small tolerance on a
platform other than the one recorded in the file.

| Golden files | Generator | Checked by |
| --- | --- | --- |
| `pixelwise_<case>.npz` (one per case) | `tests/pixelwise_golden.py` | `tests/test_pixelwise_golden.py` |
| `morph_golden.npz`, `postproc_golden.npz`, `predict_golden.npz` | `tests/golden_morph_class_ml.py` | `tests/test_golden_morph_class_ml.py` |
| `imagefilter_{edge,sar,smooth,stats,texture}.npz` | `tests/test_imagefilter_golden.py --regen` | `tests/test_imagefilter_golden.py` |
| `terrain_golden.npz` | `tests/data/golden/make_terrain_golden.py` | `tests/test_elevation_terrain_nodes.py` |

### Read this before regenerating

* **The references were made with the implementation from before the tile
  graph** (the block-loop code). They are an independent check that the tile
  graph wrappers reproduce the old outputs. Regenerating them freezes whatever
  the current code produces and removes that check for the cases you
  regenerate. Do it only when you **intend** an output to change (a bug fix or
  a deliberate change of behaviour), never to make a failing golden test pass
  without understanding why it fails.
* **Regenerate only the affected cases or groups**, and leave the other files
  untouched (see the per-generator notes below).
* **Use the working tree, not an installed rsgispy.** Each generator is run
  as a script, so Python puts the script's directory, not the repository
  root, on `sys.path`, and `import rsgispy` may pick up an older rsgispy from
  `site-packages`. Always run the generators from the repository root with
  `PYTHONPATH=<repo root>`, e.g. `PYTHONPATH=$PWD`, and check that the
  working tree is the copy being imported:

  ```bash
  PYTHONPATH=$PWD python -c "import rsgispy; print(rsgispy.__file__)"
  ```

* **Diff the old and new arrays before committing** (see
  [Checking a regenerated file](#checking-a-regenerated-file)): list the keys
  that changed and check that the changes are only in the outputs and pixels
  you expect.

### `tests/pixelwise_golden.py`: pixel-wise functions

Indices, value transforms, masks, change detection, multi-date percentiles and
spectral transforms. There is one file per case,
`tests/data/golden/pixelwise_<case>.npz`. Each case is run with block sizes
256, 16 and 23. An output that is the same for all of them is stored once
(`<label>__all`). An output that varies is stored per block size
(`<label>__bs<N>`), and the script reports it. Metadata (dtype, no data, band
names, statistics tags, overviews, etc.) is in `__meta__`.

```bash
# Only the affected case(s): the names are the case_* functions without the
# prefix, e.g. calc_indices_float, mask_img, spec_unmix_fcls.
PYTHONPATH=$PWD python tests/pixelwise_golden.py calc_indices_float mask_img

# List the case names.
PYTHONPATH=$PWD python -c "import sys; sys.path.insert(0, 'tests'); \
import pixelwise_golden; print(list(pixelwise_golden.CASES))"
```

With no arguments, the script regenerates **every** case. Avoid that unless
every output has intentionally changed. The KEA output of
`rescale_img_pxl_vals` (`kea_nd`) needs a GDAL build with the KEA driver.

### `tests/golden_morph_class_ml.py`: morphology, post-processing, prediction

There are three group files: `morph_golden.npz` (`rsgispy.imagemorphology`),
`postproc_golden.npz` (`rsgispy.classification.classpostprocess`) and
`predict_golden.npz` (`rsgispy.ml.predict`). The case name prefix (`morph_`,
`postproc_`, `predict_`) sets the group. Each case is run with block sizes 16,
23 and 256. The keys are `<case>__<all|blocksize>__<output index>__{data,meta,ctab}`.

`generate(out_dir)` always runs **all** cases and writes all three group
files. Running the script (`python tests/golden_morph_class_ml.py`) writes all
three into `tests/data/golden/`. To regenerate only the affected group, write
to a scratch directory and copy back only that group's file:

```bash
PYTHONPATH=$PWD python -c "
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tests')
import golden_morph_class_ml as g
out = Path(tempfile.mkdtemp())
g.generate(out)
shutil.copy(out / 'postproc_golden.npz', g.GOLDEN_DIR / 'postproc_golden.npz')
print('scratch copies of all groups in', out)
"
```

Even within the group you copy back, only the cases you meant to change should
differ. Check this with the diff below. If an unrelated case in the same group
changed, find out why before you commit.

### `tests/test_imagefilter_golden.py --regen`: image filters

There is one file per filter module: `imagefilter_{stats,edge,smooth,sar,texture}.npz`.
Every case is run on four input variants (`f32nd`, `u16nd`, `f32nan`, `f64x`)
with block sizes 256, 16 and 7. The keys are `<case>|<variant>|<block size>`.
A smaller block size is stored only when its output differs from the 256
reference. `__meta__` records the platform (`<sys.platform>-<machine>-numba<0|1>`),
since the last bit of a float output depends on the platform.

```bash
PYTHONPATH=$PWD python tests/test_imagefilter_golden.py --regen
```

`--regen` rewrites **all five** module files with the current platform tag.
If only one module's output was meant to change, restore the others
afterwards:

```bash
git checkout -- tests/data/golden/imagefilter_{edge,sar,smooth,texture}.npz  # keep stats
```

Regenerating on a different platform than the one recorded changes the
platform tag. The test then requires 1 ulp agreement on the new platform and
uses the loose tolerance on the old one. Note this in the commit message.

### `tests/data/golden/make_terrain_golden.py`: terrain and illumination

There is a single file, `terrain_golden.npz`, which holds the inputs
(`in_<name>`), the outputs (`out_<case>`) and a JSON `manifest` (case
function, inputs, keyword arguments, dtype, no data, band name, statistics).
Each case is run with block sizes 7, 16 and 256, and the script stops with an
assertion error if the outputs differ between block sizes.

```bash
PYTHONPATH=$PWD python tests/data/golden/make_terrain_golden.py
```

The script has no options and always rewrites every case, so the diff check
is essential. Only the `out_<case>` keys (and their `manifest` entries) of
the functions you changed should differ. The `in_*` inputs should never
change.

### Checking a regenerated file

Compare the committed version (`HEAD`) with the regenerated file before
committing. The snippet below lists added, removed and changed keys. For each
changed array, it gives the number of changed pixels, their bounding box and
the largest change. It also asserts that the changes are confined to the
pixels you expect. Edit `EXPECTED` for your change: map each key you expect to
change to a boolean mask of the pixels allowed to change, or to `None` if any
pixel may change.

```bash
FILE=tests/data/golden/postproc_golden.npz
git show HEAD:$FILE > /tmp/old_golden.npz
PYTHONPATH=$PWD python - "$FILE" <<'EOF'
import sys
import numpy as np

old = np.load("/tmp/old_golden.npz", allow_pickle=False)
new = np.load(sys.argv[1], allow_pickle=False)
print("removed:", sorted(set(old.files) - set(new.files)))
print("added:  ", sorted(set(new.files) - set(old.files)))

# Keys you expect to change -> boolean mask of the pixels allowed to change
# (same shape as the array), or None if any change in that key is expected
# (e.g. its metadata).
EXPECTED = {
    # "postproc_majority_cls3__all__0__data": allowed_mask,
}
changed = []
for key in sorted(set(old.files) & set(new.files)):
    a, b = old[key], new[key]
    if a.dtype.kind in "US":  # JSON metadata / manifest
        if str(a) == str(b):
            continue
        print(f"{key}: metadata changed")
        diff = None
    elif a.shape != b.shape or a.dtype != b.dtype:
        print(f"{key}: {a.dtype}{a.shape} -> {b.dtype}{b.shape}")
        diff = None
    else:
        same = a == b
        if a.dtype.kind == "f":
            same |= np.isnan(a) & np.isnan(b)
        diff = ~same
        if not diff.any():
            continue
        idx = np.argwhere(diff)
        delta = np.abs(b[diff].astype(float) - a[diff].astype(float))
        print(f"{key}: {int(diff.sum())} pixels changed, index range "
              f"{idx.min(axis=0).tolist()}..{idx.max(axis=0).tolist()}, "
              f"max |change| {np.nanmax(delta)}")
    changed.append(key)
    assert key in EXPECTED, f"unexpected change in {key}"
    if EXPECTED[key] is not None:
        assert diff is not None, f"{key}: shape, dtype or metadata changed"
        assert not (diff & ~EXPECTED[key]).any(), (
            f"{key}: pixels outside the expected area changed")
print(f"{len(changed)} changed keys")
EOF
```

Then run the matching golden test and the full suite, and say in the commit
message which cases were regenerated and why their outputs changed.
