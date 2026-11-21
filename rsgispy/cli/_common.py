"""
Shared command-line argument parsing, input resolution, formatting, and error
handling utilities for ``rsgispy`` CLI tools.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Callable, Iterable, List, Optional, Sequence

import rsgispy
from rsgispy import RSGISPyException
from rsgispy.tools.files import convert_file_size_units, natural_sort_file_names

_GLOB_CHARS = frozenset("*?[")


def build_parser(
    prog: str,
    description: str,
    epilog: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Create a standardised :class:`argparse.ArgumentParser` with consistent
    help formatting and a ``--version`` option.

    :param prog: Program name displayed in usage text.
    :param description: Short description of the CLI tool.
    :param epilog: Optional epilog text (e.g., usage examples).
    :return: Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s (rsgispy {rsgispy.__version__})",
    )
    return parser


def add_input_file_arg(
    parser: argparse.ArgumentParser,
    help_text: str = "Input file path.",
    required: bool = True,
) -> None:
    """Add standardised single input file option (``-i`` / ``--input``)."""
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=required,
        metavar="PATH",
        help=help_text,
    )


def add_input_files_arg(
    parser: argparse.ArgumentParser,
    help_text: str = "Input file path(s) or glob pattern(s).",
    required: bool = True,
) -> None:
    """Add standardised multi-file / glob input option (``-i`` / ``--input``)."""
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=required,
        metavar="PATH_OR_GLOB",
        help=help_text,
    )


def add_output_arg(
    parser: argparse.ArgumentParser,
    help_text: str = "Output file path.",
    required: bool = True,
    default: Optional[str] = None,
    metavar: str = "PATH",
) -> None:
    """Add standardised output path option (``-o`` / ``--output``)."""
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=required,
        default=default,
        metavar=metavar,
        help=help_text,
    )


def add_layer_arg(
    parser: argparse.ArgumentParser,
    help_text: str = "Vector layer name (defaults to the first/file-named layer).",
) -> None:
    """Add standardised vector layer option (``-l`` / ``--layer``)."""
    parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        metavar="LAYER",
        help=help_text,
    )


def add_format_arg(
    parser: argparse.ArgumentParser,
    default: str = "GPKG",
    help_text: str = "Output vector format (e.g., GPKG, GeoJSON, Parquet).",
) -> None:
    """Add standardised output format option (``-f`` / ``--format``)."""
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default=default,
        metavar="FORMAT",
        help=help_text,
    )


def add_n_cores_arg(
    parser: argparse.ArgumentParser,
    default: int = 1,
    help_text: str = "Number of worker processes to use.",
) -> None:
    """Add standardised worker core count option (``-j`` / ``--n-cores``)."""
    parser.add_argument(
        "-j",
        "--n-cores",
        type=int,
        default=default,
        metavar="N",
        help=help_text,
    )


def add_no_pbar_arg(parser: argparse.ArgumentParser) -> None:
    """Add standardised progress-bar suppression flag (``--no-pbar``)."""
    parser.add_argument(
        "--no-pbar",
        action="store_true",
        default=False,
        help="Disable progress bar output.",
    )


def add_json_arg(
    parser: argparse.ArgumentParser,
    help_text: str = "Emit machine-readable JSON output.",
) -> None:
    """Add standardised JSON output flag (``--json``)."""
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help=help_text,
    )


def _has_glob_magic(pattern: str) -> bool:
    """Return True if ``pattern`` contains shell glob wildcard characters."""
    return any(ch in pattern for ch in _GLOB_CHARS)


def resolve_input_files(
    inputs: Sequence[str],
    *,
    natural_sort: bool = True,
    allow_empty: bool = False,
) -> List[str]:
    """
    Expand file paths and glob patterns into a deduplicated list of existing
    file paths.

    Handles both shell-expanded file lists (``tool -i *.tif``) and quoted glob
    expressions (``tool -i "*.tif"``).

    :param inputs: Sequence of file paths and/or glob patterns.
    :param natural_sort: If ``True``, sort matched files naturally.
    :param allow_empty: If ``False``, raise :class:`RSGISPyException` when no
                        files match.
    :return: List of resolved file paths.
    """
    matched: List[str] = []
    seen = set()

    for raw_item in inputs:
        item = os.path.expanduser(raw_item)
        if _has_glob_magic(item):
            expanded = sorted(glob.glob(item))
            candidates = expanded
        else:
            if not os.path.exists(item):
                raise RSGISPyException(f"Input file does not exist: '{raw_item}'")
            candidates = [item]

        for path in candidates:
            if path not in seen:
                seen.add(path)
                matched.append(path)

    if natural_sort and len(matched) > 1:
        matched = natural_sort_file_names(matched)

    if not matched and not allow_empty:
        joined = ", ".join(repr(p) for p in inputs)
        raise RSGISPyException(f"No input files matched: {joined}")

    return matched


def format_bytes(size_bytes: int) -> str:
    """
    Format a byte count into a compact human-readable string (e.g., ``23.86 MB``).

    :param size_bytes: File size in bytes.
    :return: Formatted size string.
    """
    if size_bytes < 0:
        return f"{size_bytes} B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    units = ("kb", "mb", "gb", "tb")
    labels = ("KB", "MB", "GB", "TB")
    val = float(size_bytes)
    chosen_val = val
    chosen_label = "B"
    for unit, label in zip(units, labels):
        converted = convert_file_size_units(size_bytes, "bytes", unit)
        chosen_val = converted
        chosen_label = label
        if converted < 1024.0:
            break
    return f"{chosen_val:.2f} {chosen_label}"


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignments: Optional[Sequence[str]] = None,
    indent: str = "  ",
) -> str:
    """
    Render a compact, space-padded table suitable for terminal output.

    :param headers: Sequence of column header labels.
    :param rows: Sequence of row sequences (each same length as ``headers``).
    :param alignments: Optional sequence of ``'l'`` or ``'r'`` per column.
    :param indent: Leading indentation string for each line.
    :return: Formatted multi-line table string.
    """
    n_cols = len(headers)
    if alignments is None:
        alignments = ["l"] * n_cols

    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def _fmt_row(cells: Iterable[str]) -> str:
        parts = []
        for idx, cell in enumerate(cells):
            if alignments[idx] == "r":
                parts.append(cell.rjust(widths[idx]))
            else:
                parts.append(cell.ljust(widths[idx]))
        return indent + "  ".join(parts).rstrip()

    lines = [_fmt_row(headers)]
    lines.append(indent + "  ".join("-" * w for w in widths))
    for row in str_rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def run_cli(
    main_fn: Callable[[Optional[Sequence[str]]], int],
    argv: Optional[Sequence[str]] = None,
) -> int:
    """
    Execute a CLI ``main`` function with consistent error reporting and exit
    codes.

    :param main_fn: Entry-point callable accepting ``argv`` and returning an
                    integer exit status.
    :param argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    :return: Integer exit code (``0`` on success, ``1`` on error, ``130`` on
             keyboard interrupt).
    """
    try:
        return int(main_fn(argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (RSGISPyException, ValueError, OSError) as err:
        msg = err.value if isinstance(err, RSGISPyException) else str(err)
        print(f"Error: {msg}", file=sys.stderr)
        return 1
