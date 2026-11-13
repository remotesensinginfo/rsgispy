"""
The tools.textio module provides utilities for reading and writing text and
XML files.
"""

from typing import Any, List, Sequence
import xml.etree.ElementTree as ET

import rsgispy.tools.strings

__all__ = [
    "read_text_file",
    "read_text_file_no_new_lines",
    "read_text_file_to_list",
    "write_list_to_file",
    "write_data_to_file",
    "create_ascii_text_file",
    "prettify_xml_file",
]


def read_text_file(input_file: str) -> str:
    """
    Read a text file into a single string.

    :param input_file: Path to the input text file.
    :return: File contents string.

    .. code:: python

        import rsgispy.tools.textio
        txt = rsgispy.tools.textio.read_text_file("notes.txt")
    """
    with open(input_file, "r", encoding="utf-8") as data_file:
        return data_file.read()


def read_text_file_no_new_lines(input_file: str) -> str:
    """
    Read a text file into a single string with line breaks and leading/trailing
    line whitespace stripped.

    :param input_file: Path to the input text file.
    :return: Concatenated file contents string.

    .. code:: python

        import rsgispy.tools.textio
        txt = rsgispy.tools.textio.read_text_file_no_new_lines("notes.txt")
    """
    with open(input_file, "r", encoding="utf-8") as data_file:
        return "".join(line.strip() for line in data_file)


def read_text_file_to_list(
    input_file: str,
    include_empty_lines: bool = False,
) -> List[str]:
    """
    Read a text file into a list of stripped lines.

    :param input_file: Path to the input text file.
    :param include_empty_lines: Include blank lines when True (Default = False).
    :return: List of line strings.

    .. code:: python

        import rsgispy.tools.textio
        lines = rsgispy.tools.textio.read_text_file_to_list("notes.txt")
    """
    out_list: List[str] = []
    with open(input_file, "r", encoding="utf-8") as data_file:
        for line in data_file:
            stripped = line.strip()
            if include_empty_lines or stripped != "":
                out_list.append(stripped)
    return out_list


def write_list_to_file(data_lst: Sequence[Any], out_file: str) -> None:
    """
    Write a sequence of values to a text file, one item per line.

    :param data_lst: Sequence of items to write.
    :param out_file: Path to the output text file.

    .. code:: python

        import rsgispy.tools.textio
        rsgispy.tools.textio.write_list_to_file(["a", "b", "c"], "out.txt")
    """
    with open(out_file, "w", encoding="utf-8") as out_fp:
        for item in data_lst:
            out_fp.write(f"{item}\n")


def write_data_to_file(data_val: Any, out_file: str) -> None:
    """
    Write ``str(data_val)`` followed by a newline to an output text file.

    :param data_val: Value to write.
    :param out_file: Path to the output text file.

    .. code:: python

        import rsgispy.tools.textio
        rsgispy.tools.textio.write_data_to_file("hello", "out.txt")
    """
    with open(out_file, "w", encoding="utf-8") as out_fp:
        out_fp.write(f"{data_val}\n")


def create_ascii_text_file(input_file: str, output_file: str) -> None:
    """
    Copy ``input_file`` to ``output_file`` while removing non-ASCII characters.

    :param input_file: Input text file path.
    :param output_file: Output ASCII text file path.

    .. code:: python

        import rsgispy.tools.textio
        rsgispy.tools.textio.create_ascii_text_file("in.txt", "out_ascii.txt")
    """
    txt_lst = read_text_file_to_list(input_file, True)
    out_txt_lst = [
        rsgispy.tools.strings.check_str(line, rm_non_ascii=True)
        for line in txt_lst
    ]
    write_list_to_file(out_txt_lst, output_file)


def prettify_xml_file(input_file: str, output_file: str) -> None:
    """
    Prettify an XML file by inserting standard indentation and newlines.

    :param input_file: Input XML file path.
    :param output_file: Output formatted XML file path.

    .. code:: python

        import rsgispy.tools.textio
        rsgispy.tools.textio.prettify_xml_file("raw.xml", "pretty.xml")
    """
    xml_str = read_text_file_no_new_lines(input_file)
    xml_elements = ET.XML(xml_str)
    ET.indent(xml_elements)
    pretty_xml = ET.tostring(xml_elements, encoding="unicode")
    write_data_to_file(pretty_xml, output_file)
