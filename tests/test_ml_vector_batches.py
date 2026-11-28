"""
Direct tests of :func:`rsgispy.ml.vector.iter_vec_batches` (chunked reading
of GeoParquet, attribute-only Parquet and GDAL vector layers as
:class:`pyarrow.Table` objects).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point

import rsgispy
from rsgispy.ml.vector import iter_vec_batches, read_vec_layer_info

pa = pytest.importorskip("pyarrow")

N_FEATS = 23


@pytest.fixture(name="vecs", scope="module")
def fixture_vecs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("vec_batches")
    gdf = gpd.GeoDataFrame(
        {
            "seg_id": np.arange(1, N_FEATS + 1, dtype=np.int64),
            "val": np.linspace(0.0, 1.0, N_FEATS),
            "name": [f"n{i}" for i in range(N_FEATS)],
        },
        geometry=[Point(500000.0 + 10 * i, 5200000.0 - i) for i in range(N_FEATS)],
        crs="EPSG:32630",
    )
    paths = {
        "parquet": str(tmp_path / "segs.parquet"),
        "gpkg": str(tmp_path / "segs.gpkg"),
        "table": str(tmp_path / "table.parquet"),
    }
    gdf.to_parquet(paths["parquet"])
    gdf.to_file(paths["gpkg"], layer="segs")
    pd.DataFrame(gdf.drop(columns="geometry")).to_parquet(paths["table"], index=False)
    return gdf, paths


def _concat(tables):
    return pa.concat_tables(tables).to_pandas()


@pytest.mark.parametrize("fmt", ["parquet", "gpkg", "table"])
def test_batches_columns_and_order(vecs, fmt):
    gdf, paths = vecs
    tables = list(iter_vec_batches(paths[fmt], chunk_size=5))
    assert all(isinstance(t, pa.Table) for t in tables)
    assert sum(t.num_rows for t in tables) == N_FEATS
    assert max(t.num_rows for t in tables) <= 5
    if fmt != "gpkg":  # Parquet row groups are split exactly
        assert [t.num_rows for t in tables] == [5, 5, 5, 5, 3]
    # All attribute columns, in order, no geometry by default.
    assert tables[0].column_names == ["seg_id", "val", "name"]
    df = _concat(tables)
    pd.testing.assert_frame_equal(
        df, pd.DataFrame(gdf.drop(columns="geometry")), check_dtype=False
    )

    sub = _concat(iter_vec_batches(paths[fmt], columns=["val", "seg_id"], chunk_size=100))
    assert list(sub.columns) == ["val", "seg_id"]  # requested order
    np.testing.assert_array_equal(sub["seg_id"], gdf["seg_id"])


@pytest.mark.parametrize("fmt", ["parquet", "gpkg"])
def test_batches_with_geometry(vecs, fmt):
    gdf, paths = vecs
    tables = list(iter_vec_batches(paths[fmt], columns=["seg_id"], read_geometry=True,
                                   chunk_size=7))
    assert tables[0].column_names == ["seg_id", "geometry"]
    assert pa.types.is_binary(tables[0].schema.field("geometry").type) or (
        pa.types.is_large_binary(tables[0].schema.field("geometry").type)
    )
    df = _concat(tables)
    geoms = shapely.from_wkb(df["geometry"].to_numpy())
    assert all(g.equals(e) for g, e in zip(geoms, gdf.geometry))
    np.testing.assert_array_equal(df["seg_id"], gdf["seg_id"])


def test_batches_from_layer_info_and_errors(vecs):
    gdf, paths = vecs
    info = read_vec_layer_info(paths["gpkg"], "segs")
    assert info.n_features == N_FEATS
    df = _concat(iter_vec_batches(info, columns=["name"], chunk_size=4))
    assert df["name"].tolist() == gdf["name"].tolist()

    with pytest.raises(rsgispy.RSGISPyException, match="chunk_size"):
        next(iter_vec_batches(paths["parquet"], chunk_size=0))
    with pytest.raises(rsgispy.RSGISPyException, match="not found"):
        next(iter_vec_batches(paths["parquet"], columns=["seg_id", "nope"]))
    with pytest.raises(rsgispy.RSGISPyException, match="no geometry"):
        next(iter_vec_batches(paths["table"], read_geometry=True))
    with pytest.raises(rsgispy.RSGISPyException, match="does not exist"):
        next(iter_vec_batches(paths["parquet"] + ".missing"))
