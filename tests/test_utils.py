"""
Tests for rsgispy.tools utility modules split from utils
(colours, credentials, dates, dicts, jsonio, maths, strings, system, textio).
"""

import datetime
import json

import numpy as np
import pytest

import rsgispy
import rsgispy.tools.colours as rcolours
import rsgispy.tools.credentials as rcreds
import rsgispy.tools.dates as rdates
import rsgispy.tools.dicts as rdicts
import rsgispy.tools.jsonio as rjsonio
import rsgispy.tools.maths as rmaths
import rsgispy.tools.strings as rstrings
import rsgispy.tools.system as rsystem
import rsgispy.tools.textio as rtextio


class _CustomJsonObj:
    def to_json(self):
        return {"custom": 42}


def test_json_encoder_and_json_io(tmp_path):
    """Test RSGISNumpyArrayEncoder, JSON, and gzip-JSON read/write helpers."""
    payload = {
        "int": np.int64(7),
        "float": np.float32(1.5),
        "arr": np.array([[1, 2], [3, 4]]),
        "bool": np.bool_(True),
        "date": datetime.date(2024, 5, 1),
        "dt": datetime.datetime(2024, 5, 1, 12, 30, 0),
        "delta": datetime.timedelta(days=2),
        "custom": _CustomJsonObj(),
    }
    json_str = rjsonio.dict_to_json_str(payload)
    parsed = json.loads(json_str)
    assert parsed["int"] == 7
    assert parsed["arr"] == [[1, 2], [3, 4]]
    assert parsed["bool"] is True
    assert parsed["date"] == "2024-05-01"
    assert parsed["custom"] == {"custom": 42}

    json_file = tmp_path / "data.json"
    rjsonio.write_dict_to_json(payload, str(json_file))
    assert rjsonio.read_json_to_dict(str(json_file))["int"] == 7

    gz_file = tmp_path / "data.json.gz"
    rjsonio.write_dict_to_json_gz(payload, str(gz_file))
    assert rjsonio.read_gz_json_to_dict(str(gz_file))["arr"] == [[1, 2], [3, 4]]


def test_numeric_string_and_text_helpers(tmp_path, monkeypatch):
    """Test numeric checks, parsing, padding, strings, env vars, and text I/O."""
    uid = rstrings.uid_generator(10)
    assert len(uid) == 10
    assert uid.isalnum()

    assert rmaths.is_number("12.34")
    assert rmaths.is_number("3+4j")
    assert not rmaths.is_number("abc")
    assert not rmaths.is_number(True)

    assert rmaths.similar_numeric_vals([1.0, 2.0], [1.05, 2.05], eql_thres=0.1)
    assert not rmaths.similar_numeric_vals([1.0, 2.0], [1.5, 2.05], eql_thres=0.1)
    assert rmaths.similar_numeric_vals(5.0, 5.02, eql_thres=0.1)

    assert rstrings.str_to_float(" 3.5 ") == pytest.approx(3.5)
    assert rstrings.str_to_float("bad", err_val=-1.0) == pytest.approx(-1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        rstrings.str_to_float("bad")

    assert rstrings.str_to_int(" 4 ") == 4
    assert rstrings.str_to_int(" 4.6 ") == 5
    assert rstrings.str_to_int("bad", err_val=-9) == -9
    with pytest.raises(rsgispy.RSGISPyException):
        rstrings.str_to_int("bad")

    assert rstrings.zero_pad_num_str(7, str_len=4) == "0007"
    assert (
        rstrings.zero_pad_num_str(
            -1.256,
            str_len=4,
            round_num=True,
            round_n_digts=2,
            integerise=True,
            absolute=True,
            gain=100,
        )
        == "0126"
    )

    assert rmaths.in_bounds(np.array([1.0, 5.0]), 1.0, 5.0)
    assert not rmaths.in_bounds(np.array([1.0, 5.0]), 1.0, 5.0, upper_strict=True)
    assert rmaths.mixed_signs(np.array([-1.0, 0.0, 2.0]))
    assert not rmaths.mixed_signs(np.array([1.0, 2.0]))
    assert rmaths.negative(np.array([-3.0, -1.0]))
    assert not rmaths.negative(np.array([-3.0, 0.0]))
    assert rmaths.is_odd(3)
    assert not rmaths.is_odd(4)

    assert rcolours.hex_to_rgb("#b432be") == (180, 50, 190)
    assert rcolours.rgb_to_hex(180, 50, 190) == "#b432be"
    assert rcolours.rgb_to_hex(1.0, 0.0, 0.0, normalised=True) == "#ff0000"
    with pytest.raises(rsgispy.RSGISPyException):
        rcolours.hex_to_rgb("#123")
    with pytest.raises(rsgispy.RSGISPyException):
        rcolours.rgb_to_hex(300, 0, 0)

    assert rstrings.remove_repeated_chars("a__b___c", "_") == "a_b_c"
    assert (
        rstrings.check_str(
            " Héllo--World! 123 ",
            rm_non_ascii=True,
            rm_dashs=True,
            rm_spaces=True,
            rm_punc=True,
        )
        == "Hllo_World_123"
    )

    monkeypatch.setenv("RSGISPY_TEST_VAR", "ok")
    assert rsystem.get_environment_variable("RSGISPY_TEST_VAR") == "ok"
    assert rsystem.get_environment_variable("RSGISPY_MISSING_VAR") is None
    assert rsystem.num_process_cores() >= 1

    txt_file = tmp_path / "sample.txt"
    rtextio.write_list_to_file(["line1", "", "line2"], str(txt_file))
    assert rtextio.read_text_file(str(txt_file)) == "line1\n\nline2\n"
    assert rtextio.read_text_file_no_new_lines(str(txt_file)) == "line1line2"
    assert rtextio.read_text_file_to_list(str(txt_file)) == ["line1", "line2"]
    assert rtextio.read_text_file_to_list(str(txt_file), include_empty_lines=True) == [
        "line1",
        "",
        "line2",
    ]

    non_ascii_file = tmp_path / "non_ascii.txt"
    ascii_file = tmp_path / "ascii.txt"
    rtextio.write_data_to_file("café", str(non_ascii_file))
    rtextio.create_ascii_text_file(str(non_ascii_file), str(ascii_file))
    assert rtextio.read_text_file_to_list(str(ascii_file)) == ["caf"]


def test_combinatorics_dates_dict_struct_and_encoding(tmp_path):
    """Test powerset, create_var_list, dates, dict_struct_*, base64, and XML."""
    pset = rmaths.powerset_lst(["a", "b", "c"], min_items=2)
    assert len(pset) == 4
    assert ["a", "b", "c"] in pset

    combos = rmaths.create_var_list({"k": [5, 10], "d": [1, 2]})
    assert len(combos) == 4
    assert {"k": 5, "d": 1} in combos

    base_d = datetime.date(2020, 1, 1)
    assert rdates.get_days_since(2020, 10, base_d) == 9
    assert rdates.get_days_since_date(2020, 1, 10, base_d) == 9
    with pytest.raises(rsgispy.RSGISPyException):
        rdates.get_days_since(2019, 1, base_d)
    with pytest.raises(rsgispy.RSGISPyException):
        rdates.get_days_since_date(2019, 1, 1, base_d)

    assert rdates.find_month_end_date(2020, 2) == 29
    assert rdates.create_year_month_start_end_lst(2020, 11, 2021, 2) == [
        (2020, 11),
        (2020, 12),
        (2021, 1),
        (2021, 2),
    ]
    assert rdates.create_year_month_n_months_lst(2020, 12, 2, 1) == [
        (2020, 11),
        (2020, 12),
        (2021, 1),
        (2021, 2),
    ]
    assert rdates.is_summer_winter(52.0, datetime.date(2020, 7, 1)) == 1
    assert rdates.is_summer_winter(-30.0, datetime.date(2020, 7, 1)) == 2

    incs = rdates.get_datetime_increment_lst(
        datetime.datetime(2020, 1, 1),
        datetime.datetime(2020, 1, 3),
        datetime.timedelta(days=1),
    )
    assert incs == [datetime.datetime(2020, 1, 2), datetime.datetime(2020, 1, 3)]

    struct = {
        "meta": {
            "sensor": "S2",
            "active": True,
            "date": "2024-06-15",
            "dt": "2024-06-15T10:20:30.000Z",
            "bands": ["B02", "B03"],
            "scale": 10,
        }
    }
    assert rdicts.dict_struct_does_path_exist(struct, ["meta", "sensor"])
    assert not rdicts.dict_struct_does_path_exist(struct, ["meta", "missing"])
    assert rdicts.dict_struct_get_str_value(struct, ["meta", "sensor"], ["S2", "L8"]) == "S2"
    assert rdicts.dict_struct_get_boolean_value(struct, ["meta", "active"]) is True
    assert rdicts.dict_struct_get_date_value(struct, ["meta", "date"]) == datetime.date(2024, 6, 15)
    assert rdicts.dict_struct_get_datetime_value(
        struct, ["meta", "dt"], ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"]
    ) == datetime.datetime(2024, 6, 15, 10, 20, 30)
    assert rdicts.dict_struct_get_str_list_value(
        struct, ["meta", "bands"], ["B02", "B03", "B04"]
    ) == ["B02", "B03"]
    assert rdicts.dict_struct_get_numeric_value(struct, ["meta", "scale"], 0, 20) == 10
    assert rdicts.dict_struct_get_list_value(struct, ["meta", "bands"]) == ["B02", "B03"]

    with pytest.raises(rsgispy.RSGISPyException):
        rdicts.dict_struct_get_numeric_value(struct, ["meta", "scale"], valid_lower=20)

    creds = tmp_path / "creds.txt"
    rcreds.create_username_password_file("alice", "s3cr3t", str(creds))
    assert rcreds.get_username_password(str(creds)) == ("alice", "s3cr3t")

    raw_xml = tmp_path / "raw.xml"
    pretty_xml = tmp_path / "pretty.xml"
    raw_xml.write_text("<root><child>1</child></root>", encoding="utf-8")
    rtextio.prettify_xml_file(str(raw_xml), str(pretty_xml))
    assert "\n  <child>1</child>\n" in pretty_xml.read_text(encoding="utf-8")
