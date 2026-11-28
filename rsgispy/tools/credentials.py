"""
The tools.credentials module provides base64 text encoding/decoding and simple
username/password credential file utilities.

.. warning::

   Base64 is an encoding, not encryption: anyone who can read a credentials
   file written by :func:`create_username_password_file` can decode the
   username and password with :func:`decode_base64_text` (or any base64
   tool). It only keeps them from being read at a glance. Protect the file
   with its permissions (it is created readable by its owner only), keep it
   out of version control and shared folders, and prefer the environment
   variables read by :func:`resolve_env_user_pass` or a secrets manager for
   anything sensitive.
"""

import base64
import os
from typing import Optional, Sequence, Tuple, Union

import rsgispy
import rsgispy.tools.system
import rsgispy.tools.textio

__all__ = [
    "encode_base64_text",
    "decode_base64_text",
    "create_username_password_file",
    "get_username_password",
    "resolve_env_credential",
    "resolve_env_user_pass",
]


def encode_base64_text(input_txt: str) -> str:
    """
    Encode ``input_txt`` as a base64 ASCII string. This is not encryption:
    the text can be recovered by anyone with :func:`decode_base64_text`.

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
    Write a two-line base64-obfuscated username/password file with restrictive
    owner-only permissions (``0o600``).

    The credentials are **not encrypted**: base64 only obfuscates them and is
    trivially reversed, so the file permissions are the only protection.

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
    fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out_fp:
        for line in out_data:
            out_fp.write(f"{line}\n")
    try:
        os.chmod(out_file, 0o600)
    except OSError:
        pass


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


def resolve_env_credential(
    value: Optional[str],
    env_vars: Union[str, Sequence[str]],
    *,
    required: bool = False,
    label: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a credential string, falling back to one or more environment
    variables when ``value`` is ``None``.

    :param value: Explicitly supplied credential string, or ``None`` to look up
                  environment variables.
    :param env_vars: Environment variable name or ordered sequence of fallback
                     environment variable names.
    :param required: When ``True``, raise :class:`rsgispy.RSGISPyException` if
                     the resolved value is ``None`` or blank.
    :param label: Human-readable credential label used in error messages when
                  ``required=True`` (e.g. ``"USGS username"``).
    :return: Resolved credential string, or ``None`` if unset and not required.

    .. code:: python

        import rsgispy.tools.credentials
        api_key = rsgispy.tools.credentials.resolve_env_credential(
            None, ("RSGIS_PLANET_API_KEY", "PL_API_KEY")
        )
    """
    var_names = [env_vars] if isinstance(env_vars, str) else list(env_vars)
    resolved = value
    if resolved is None:
        for var_name in var_names:
            resolved = rsgispy.tools.system.get_environment_variable(var_name)
            if resolved is not None:
                break
    if required and (not resolved or not str(resolved).strip()):
        desc = label or "Credential"
        vars_desc = " or ".join(var_names)
        raise rsgispy.RSGISPyException(
            f"{desc} must be provided or set via the {vars_desc} "
            f"environment variable."
        )
    return resolved


def resolve_env_user_pass(
    username: Optional[str],
    password: Optional[str],
    user_env_vars: Union[str, Sequence[str]],
    pass_env_vars: Union[str, Sequence[str]],
    *,
    required: bool = False,
    service_name: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve ``(username, password)`` credentials from explicit arguments or
    environment variables.

    :param username: Explicit username, or ``None`` to read ``user_env_vars``.
    :param password: Explicit password, or ``None`` to read ``pass_env_vars``.
    :param user_env_vars: Username environment variable name(s).
    :param pass_env_vars: Password environment variable name(s).
    :param required: When ``True``, raise :class:`rsgispy.RSGISPyException` if
                     either username or password is missing or blank.
    :param service_name: Optional service prefix for error messages (e.g.
                         ``"USGS"``).
    :return: ``(username, password)`` tuple.

    .. code:: python

        import rsgispy.tools.credentials
        user, passwd = rsgispy.tools.credentials.resolve_env_user_pass(
            None, None, "RSGIS_COP_USER", "RSGIS_COP_PASS"
        )
    """
    prefix = f"{service_name} " if service_name else ""
    resolved_user = resolve_env_credential(
        username,
        user_env_vars,
        required=required,
        label=f"{prefix}username",
    )
    resolved_pass = resolve_env_credential(
        password,
        pass_env_vars,
        required=required,
        label=f"{prefix}password",
    )
    return resolved_user, resolved_pass
