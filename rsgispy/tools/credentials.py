"""
The tools.credentials module provides base64 text encoding/decoding and simple
username/password credential file utilities.
"""

import base64
from typing import Tuple

import rsgispy
import rsgispy.tools.textio

__all__ = [
    "encode_base64_text",
    "decode_base64_text",
    "create_username_password_file",
    "get_username_password",
]


def encode_base64_text(input_txt: str) -> str:
    """
    Encode ``input_txt`` as a base64 ASCII string.

    :param input_txt: Plain text string to encode.
    :return: Base64-encoded string.

    .. code:: python

        import rsgispy.tools.credentials
        encoded = rsgispy.tools.credentials.encode_base64_text("my_secret")
    """
    return base64.b64encode(input_txt.encode("utf-8")).decode("utf-8")


def decode_base64_text(in_encoded_txt: str) -> str:
    """
    Decode a base64 string produced by :func:`encode_base64_text`.

    :param in_encoded_txt: Base64-encoded string.
    :return: Decoded plain text string.

    .. code:: python

        import rsgispy.tools.credentials
        decoded = rsgispy.tools.credentials.decode_base64_text("bXlfc2VjcmV0")
    """
    return base64.b64decode(in_encoded_txt.encode("utf-8")).decode("utf-8")


def create_username_password_file(
    username: str,
    password: str,
    out_file: str,
) -> None:
    """
    Write a two-line base64-obfuscated username/password file.

    :param username: Username string.
    :param password: Password string.
    :param out_file: Output file path.

    .. code:: python

        import rsgispy.tools.credentials
        rsgispy.tools.credentials.create_username_password_file(
            "user1", "pass123", "creds.txt"
        )
    """
    out_data = [encode_base64_text(username), encode_base64_text(password)]
    rsgispy.tools.textio.write_list_to_file(out_data, out_file)


def get_username_password(input_file: str) -> Tuple[str, str]:
    """
    Read and decode a username/password file created by
    :func:`create_username_password_file`.

    :param input_file: Input credentials file path.
    :return: ``(username, password)`` tuple.

    .. code:: python

        import rsgispy.tools.credentials
        user, passwd = rsgispy.tools.credentials.get_username_password("creds.txt")
    """
    in_data_lst = rsgispy.tools.textio.read_text_file_to_list(input_file)
    if len(in_data_lst) != 2:
        raise rsgispy.RSGISPyException(
            "There should just be two lines in username/password file."
        )
    return (
        decode_base64_text(in_data_lst[0]),
        decode_base64_text(in_data_lst[1]),
    )
