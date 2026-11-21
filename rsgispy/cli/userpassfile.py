"""
Command-line tool ``rsgisuserpassfile``: create a base64-encoded username and
password credentials file compatible with :mod:`rsgispy.tools.credentials`.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Optional, Sequence

from rsgispy import RSGISPyException
from rsgispy.cli._common import add_output_arg, build_parser, run_cli
from rsgispy.tools.credentials import create_username_password_file


def build_userpassfile_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisuserpassfile``."""
    parser = build_parser(
        prog="rsgisuserpassfile",
        description=(
            "Create a base64-encoded username/password file for use with "
            "rsgispy data access utilities."
        ),
    )
    add_output_arg(parser, "Output credentials file path.")
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=None,
        metavar="USER",
        help="Username (prompted interactively if omitted).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisuserpassfile`` and return an exit status code."""
    parser = build_userpassfile_parser()
    args = parser.parse_args(argv)

    out_file = os.path.expanduser(args.output)
    if os.path.exists(out_file) and not args.overwrite:
        raise RSGISPyException(
            f"Output file already exists: '{args.output}' "
            "(use --overwrite to replace it)."
        )

    username = args.username
    if username is None:
        username = input("Username: ").strip()
    if not username:
        raise RSGISPyException("Username cannot be empty.")

    password = getpass.getpass("Password: ")
    if not password:
        raise RSGISPyException("Password cannot be empty.")

    parent = os.path.dirname(os.path.abspath(out_file))
    if parent:
        os.makedirs(parent, exist_ok=True)

    create_username_password_file(
        username=username,
        password=password,
        out_file=out_file,
    )
    try:
        os.chmod(out_file, 0o600)
    except OSError:
        pass

    print(f"Credentials file written to: {out_file}")
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisuserpassfile``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
