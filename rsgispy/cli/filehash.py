"""
Command-line tool ``rsgisfilehash``: compute cryptographic hash digests for
one or more input files.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Dict, Optional, Sequence

from rsgispy.cli._common import (
    add_input_files_arg,
    add_json_arg,
    build_parser,
    resolve_input_files,
    run_cli,
)
from rsgispy.tools import hashing

_HASH_FUNCS: Dict[str, Callable[[str], str]] = {
    "md5": hashing.create_md5_hash,
    "sha1": hashing.create_sha1_hash,
    "sha224": hashing.create_sha224_hash,
    "sha256": hashing.create_sha256_hash,
    "sha384": hashing.create_sha384_hash,
    "sha512": hashing.create_sha512_hash,
    "sha3_224": hashing.create_sha3_224_hash,
    "sha3_256": hashing.create_sha3_256_hash,
    "sha3_384": hashing.create_sha3_384_hash,
    "sha3_512": hashing.create_sha3_512_hash,
    "blake2b": hashing.create_blake2b_hash,
    "blake2s": hashing.create_blake2s_hash,
}


def build_filehash_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``rsgisfilehash``."""
    parser = build_parser(
        prog="rsgisfilehash",
        description="Compute cryptographic hash digest(s) for one or more files.",
    )
    add_input_files_arg(parser, "Input file path(s) or glob pattern(s).")
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str.lower,
        choices=sorted(_HASH_FUNCS),
        default="md5",
        help="Cryptographic hash algorithm.",
    )
    add_json_arg(parser, "Emit digest(s) as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute ``rsgisfilehash`` and return an exit status code."""
    parser = build_filehash_parser()
    args = parser.parse_args(argv)

    files = resolve_input_files(args.input, natural_sort=True)
    hash_fn = _HASH_FUNCS[args.algorithm]
    results = [
        {"file": path, "algorithm": args.algorithm, "hash": hash_fn(path)}
        for path in files
    ]

    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
        return 0

    if len(results) == 1:
        print(results[0]["hash"])
    else:
        for item in results:
            print(f"{item['hash']}  {item['file']}")
    return 0


def cli_main() -> int:
    """Console script entry point for ``rsgisfilehash``."""
    return run_cli(main)


if __name__ == "__main__":
    sys.exit(cli_main())
