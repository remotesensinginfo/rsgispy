"""
The tools.system module provides operating system environment and hardware
resource helper utilities.
"""

import os
from typing import Optional

__all__ = [
    "get_environment_variable",
    "num_process_cores",
]


def get_environment_variable(var: str) -> Optional[str]:
    """
    Get an environment variable value, or ``None`` if it is not set.

    :param var: Environment variable name.
    :return: Variable value string or ``None``.

    .. code:: python

        import rsgispy.tools.system
        path_val = rsgispy.tools.system.get_environment_variable("PATH")
    """
    return os.environ.get(var)


def num_process_cores() -> int:
    """
    Return the number of CPU cores available on the machine.

    :return: Integer CPU count (defaults to 1 if undetectable).

    .. code:: python

        import rsgispy.tools.system
        n_cores = rsgispy.tools.system.num_process_cores()
    """
    return os.cpu_count() or 1
