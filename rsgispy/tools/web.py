"""
The tools.web module provides utilities for discovering linked images and
files on web pages.

Functions requiring ``beautifulsoup4`` and ``requests`` raise
:class:`rsgispy.RSGISPyException` at runtime if either optional dependency is
not installed.
"""

from typing import List
from urllib.parse import urljoin

import rsgispy

try:
    import bs4  # pylint: disable=import-error

    HAVE_BS4 = True
except ImportError:
    bs4 = None  # type: ignore[assignment]
    HAVE_BS4 = False

try:
    import requests  # pylint: disable=import-error

    HAVE_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAVE_REQUESTS = False

__all__: List[str] = [
    "find_web_imgs",
    "find_web_files",
]


def _find_web_elements(
    base_url: str,
    tag_name: str,
    attr_name: str,
    file_ext: str,
    timeout: float,
) -> List[str]:
    """Extract absolute URLs matching ``file_ext`` from HTML elements."""
    found_urls: List[str] = []
    ext_suffix = f".{file_ext.lstrip('.').lower()}"
    try:
        response = requests.get(base_url, timeout=timeout)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        find_kwargs = {attr_name: True} if tag_name == "a" else {}
        for elem in soup.find_all(tag_name, **find_kwargs):
            target = elem.get(attr_name)
            if target and target.lower().endswith(ext_suffix):
                found_urls.append(urljoin(base_url, target))
    except requests.exceptions.RequestException as err:
        print(f"Error fetching website: {err}")
    return found_urls


def find_web_imgs(
    base_url: str,
    file_ext: str = "jpg",
    timeout: float = 30.0,
) -> List[str]:
    """
    Extract all ``<img>`` URLs with ``file_ext`` from a web page.

    :param base_url: URL of the web page to search.
    :param file_ext: File extension filter (Default = ``'jpg'``).
    :param timeout: Request timeout in seconds (Default = ``30.0``).
    :return: List of matching absolute image URLs.

    .. code:: python

        import rsgispy.tools.web
        img_urls = rsgispy.tools.web.find_web_imgs(
            "https://example.com/gallery", file_ext="png"
        )
    """
    rsgispy.check_optional_dep(HAVE_BS4, "find_web_imgs", "beautifulsoup4", "dataaccess")
    rsgispy.check_optional_dep(HAVE_REQUESTS, "find_web_imgs", "requests", "dataaccess")
    return _find_web_elements(base_url, "img", "src", file_ext, timeout)


def find_web_files(
    base_url: str,
    file_ext: str = "pdf",
    timeout: float = 30.0,
) -> List[str]:
    """
    Extract all ``<a>`` link URLs with ``file_ext`` from a web page.

    :param base_url: URL of the web page to search.
    :param file_ext: File extension filter (Default = ``'pdf'``).
    :param timeout: Request timeout in seconds (Default = ``30.0``).
    :return: List of matching absolute file URLs.

    .. code:: python

        import rsgispy.tools.web
        pdf_urls = rsgispy.tools.web.find_web_files(
            "https://example.com/docs", file_ext="pdf"
        )
    """
    rsgispy.check_optional_dep(HAVE_BS4, "find_web_files", "beautifulsoup4", "dataaccess")
    rsgispy.check_optional_dep(HAVE_REQUESTS, "find_web_files", "requests", "dataaccess")
    return _find_web_elements(base_url, "a", "href", file_ext, timeout)
