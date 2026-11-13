"""
The tools.webtools module provides utilities for discovering linked images and
files on web pages.

Functions requiring ``bs4`` and ``requests`` are only defined and exported when
both packages are installed.
"""

from typing import List
from urllib.parse import urljoin

try:
    import bs4  # pylint: disable=import-error

    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False

try:
    import requests  # pylint: disable=import-error

    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

__all__: List[str] = []

if HAVE_BS4 and HAVE_REQUESTS:
    __all__.extend(
        [
            "find_web_imgs",
            "find_web_files",
        ]
    )


if HAVE_BS4 and HAVE_REQUESTS:

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
        """
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
        """
        return _find_web_elements(base_url, "a", "href", file_ext, timeout)
