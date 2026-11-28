"""
Unit tests for rsgispy.tools.tables.
"""

import importlib.util

import pandas as pd
import pytest

import rsgispy
import rsgispy.tools.tables as tools_tables


def test_resolve_table_output_spec():
    """Verify table format and compression resolution."""
    assert tools_tables.resolve_table_output_spec("out.csv") == ("csv", "infer")
    assert tools_tables.resolve_table_output_spec("out.csv.gz") == ("csv", "infer")
    assert tools_tables.resolve_table_output_spec(
        "out.tsv", out_compress=True
    ) == ("tsv", "gzip")
    assert tools_tables.resolve_table_output_spec(
        "out.csv", out_compress=False
    ) == ("csv", None)
    assert tools_tables.resolve_table_output_spec(
        "out.parquet", out_compress="zstd"
    ) == ("parquet", "zstd")
    assert tools_tables.resolve_table_output_spec("out.parquet.gzip") == (
        "parquet",
        "gzip",
    )
    assert tools_tables.resolve_table_output_spec(
        "out.dat", out_format="json"
    ) == ("json", None)

    with pytest.raises(rsgispy.RSGISPyException):
        tools_tables.resolve_table_output_spec("out.xml")
    with pytest.raises(rsgispy.RSGISPyException):
        tools_tables.resolve_table_output_spec("out.csv", out_format="invalid")
    with pytest.raises(rsgispy.RSGISPyException):
        tools_tables.resolve_table_output_spec(
            "out.parquet", out_compress="badcodec"
        )
    with pytest.raises(rsgispy.RSGISPyException):
        tools_tables.resolve_table_output_spec(
            "out.csv", out_compress=123  # type: ignore[arg-type]
        )


def test_write_dataframe_table(tmp_path):
    """Verify writing DataFrames to CSV, compressed CSV, TSV, JSON, Pickle, and Parquet."""
    df = pd.DataFrame({"id": [1, 2], "val": [3.5, 4.5]})

    csv_path = tmp_path / "sub" / "table.csv"
    tools_tables.write_dataframe_table(df, csv_path)
    assert tools_tables.read_dataframe_table(csv_path).equals(df)

    csv_gz_path = tmp_path / "table.csv.gz"
    tools_tables.write_dataframe_table(df, csv_gz_path)
    assert tools_tables.read_dataframe_table(csv_gz_path).equals(df)

    tsv_path = tmp_path / "table.tsv"
    tools_tables.write_dataframe_table(df, tsv_path)
    assert tools_tables.read_dataframe_table(tsv_path).equals(df)

    json_path = tmp_path / "table.json"
    tools_tables.write_dataframe_table(df, json_path)
    assert len(tools_tables.read_dataframe_table(json_path)) == 2

    pkl_path = tmp_path / "table.pkl"
    tools_tables.write_dataframe_table(df, pkl_path)
    assert tools_tables.read_dataframe_table(pkl_path).equals(df)

    with pytest.raises(rsgispy.RSGISPyException):
        tools_tables.read_dataframe_table(tmp_path / "missing.csv")

    if importlib.util.find_spec("pyarrow") is not None:
        import pyarrow.parquet as pq

        pq_path = tmp_path / "table.parquet"
        tools_tables.write_dataframe_table(df, pq_path, out_compress="zstd")
        assert pq.ParquetFile(pq_path).metadata.row_group(0).column(0).compression == "ZSTD"
        assert tools_tables.read_dataframe_table(pq_path).equals(df)


def test_read_dataframe_table_allow_pickle_guard(tmp_path):
    """read_dataframe_table rejects pickle inputs when allow_pickle=False."""
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    pkl_path = tmp_path / "sample.pkl"
    df.to_pickle(pkl_path)

    with pytest.raises(rsgispy.RSGISPyException, match="allow_pickle=False"):
        tools_tables.read_dataframe_table(pkl_path, allow_pickle=False)

    loaded = tools_tables.read_dataframe_table(pkl_path, allow_pickle=True)
    pd.testing.assert_frame_equal(loaded, df)
