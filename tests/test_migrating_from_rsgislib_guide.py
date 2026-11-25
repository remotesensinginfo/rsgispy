"""
Checks for the RSGISLib migration guide
(``docs/guides/migrating_from_rsgislib.rst``):

* every rsgispy name referenced in the guide (including the generated
  tables) exists;
* the generated tables are up to date with
  ``docs/examples/gen_rsgislib_migration_table.py``;
* the short rsgispy code examples in the guide run.
"""

import importlib.util
import os
import re
import textwrap

import pytest

from zonal_scene import make_zonal_scene

DOCS = os.path.join(os.path.dirname(__file__), os.pardir, "docs")
GUIDE = os.path.join(DOCS, "guides", "migrating_from_rsgislib.rst")
GENERATOR = os.path.join(DOCS, "examples", "gen_rsgislib_migration_table.py")

ROLE_RE = re.compile(r":(?:func|class|mod|data|meth|attr):`~?(rsgispy[\w.]*)`")


def _read_guide() -> str:
    with open(GUIDE, encoding="utf-8") as f_obj:
        return f_obj.read()


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_rsgislib_migration_table",
                                                  GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _python_blocks(text: str):
    """Yield the contents of the ``.. code-block:: python`` blocks."""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() == ".. code-block:: python":
            idx += 1
            block = []
            while idx < len(lines) and (
                not lines[idx].strip() or lines[idx].startswith((" ", "\t"))
            ):
                block.append(lines[idx])
                idx += 1
            yield textwrap.dedent("\n".join(block)).strip()
        else:
            idx += 1


def test_rsgispy_names_exist():
    """Every rsgispy cross-reference in the guide resolves."""
    gen = _load_generator()
    names = sorted(set(ROLE_RE.findall(_read_guide())))
    assert len(names) >= 80
    missing = []
    for name in names:
        try:
            gen.resolve(name)
        except (ImportError, AttributeError):
            missing.append(name)
    assert not missing, f"Names in the guide not found in rsgispy: {missing}"


def test_tables_up_to_date():
    """The generated tables in the guide match the generator output."""
    gen = _load_generator()
    text = _read_guide()
    assert gen.BEGIN in text and gen.END in text
    assert gen.replace_block(text, gen.generated_block()) == text, (
        "Run: python docs/examples/gen_rsgislib_migration_table.py --write"
    )


def test_rsgispy_snippets_run(tmp_path, monkeypatch):
    """The rsgispy code blocks in the guide run in order."""
    scene = make_zonal_scene(str(tmp_path))
    assert os.path.basename(scene["s2"]) == "s2_sref.tif"
    assert os.path.basename(scene["fields"]) == "fields.gpkg"
    # Two endmembers x 4 bands, band-outer / endmember-inner order.
    with open(tmp_path / "endmembers.mtxt", "w", encoding="utf-8") as f_obj:
        f_obj.write("m=2\nn=4\n400,1100,700,1200,500,1300,3000,1600\n")
    monkeypatch.chdir(tmp_path)

    blocks = [b for b in _python_blocks(_read_guide())
              if "import rsgispy" in b and "import rsgislib" not in b]
    assert len(blocks) >= 4
    for block in blocks:
        exec(compile(block, GUIDE, "exec"), {})  # pylint: disable=exec-used
    for out_file in ("ndvi.tif", "s2_refl.tif", "endmembers.csv", "s2_unmix.tif"):
        assert os.path.exists(tmp_path / out_file)


@pytest.mark.parametrize("name", ["band_math", "bandMath"])
def test_rsgislib_names_not_in_rsgispy(name):
    """Renamed functions do not exist under their RSGISLib names."""
    import rsgispy.imagecalc.maths  # pylint: disable=import-outside-toplevel

    assert not hasattr(rsgispy.imagecalc.maths, name)
