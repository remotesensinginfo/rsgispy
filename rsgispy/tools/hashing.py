"""
The tools.hashing module provides cryptographic and checksum hash utilities
for files.
"""

import hashlib

__all__ = [
    "create_sha1_hash",
    "create_sha224_hash",
    "create_sha256_hash",
    "create_sha384_hash",
    "create_sha512_hash",
    "create_md5_hash",
    "create_blake2b_hash",
    "create_blake2s_hash",
    "create_sha3_224_hash",
    "create_sha3_256_hash",
    "create_sha3_384_hash",
    "create_sha3_512_hash",
]


def _calc_file_hash(
    hash_name: str, input_file: str, block_size: int = 4096
) -> str:
    """Compute a hexadecimal digest of ``input_file`` using ``hashlib``."""
    hash_obj = hashlib.new(hash_name)
    with open(input_file, "rb") as fobj:
        for byte_block in iter(lambda: fobj.read(block_size), b""):
            hash_obj.update(byte_block)
    return hash_obj.hexdigest()


def create_sha1_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-1 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-1 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha1_hash("data.bin")
    """
    return _calc_file_hash("sha1", input_file, block_size)


def create_sha224_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-224 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-224 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha224_hash("data.bin")
    """
    return _calc_file_hash("sha224", input_file, block_size)


def create_sha256_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-256 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-256 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha256_hash("data.bin")
    """
    return _calc_file_hash("sha256", input_file, block_size)


def create_sha384_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-384 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-384 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha384_hash("data.bin")
    """
    return _calc_file_hash("sha384", input_file, block_size)


def create_sha512_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-512 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-512 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha512_hash("data.bin")
    """
    return _calc_file_hash("sha512", input_file, block_size)


def create_md5_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the MD5 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: MD5 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_md5_hash("data.bin")
    """
    return _calc_file_hash("md5", input_file, block_size)


def create_blake2b_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the BLAKE2b hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: BLAKE2b hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_blake2b_hash("data.bin")
    """
    return _calc_file_hash("blake2b", input_file, block_size)


def create_blake2s_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the BLAKE2s hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: BLAKE2s hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_blake2s_hash("data.bin")
    """
    return _calc_file_hash("blake2s", input_file, block_size)


def create_sha3_224_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-224 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-224 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha3_224_hash("data.bin")
    """
    return _calc_file_hash("sha3_224", input_file, block_size)


def create_sha3_256_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-256 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-256 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha3_256_hash("data.bin")
    """
    return _calc_file_hash("sha3_256", input_file, block_size)


def create_sha3_384_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-384 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-384 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha3_384_hash("data.bin")
    """
    return _calc_file_hash("sha3_384", input_file, block_size)


def create_sha3_512_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-512 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-512 hex digest string.

    .. code:: python

        import rsgispy.tools.hashing
        digest = rsgispy.tools.hashing.create_sha3_512_hash("data.bin")
    """
    return _calc_file_hash("sha3_512", input_file, block_size)
