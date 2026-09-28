"""Safely fetch pages from discovered web source URLs."""

from http import HTTPStatus
from time import sleep
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from financial_advisor.config import (
    DEFAULT_WEB_TIMEOUT_SECONDS,
    WEB_FETCH_ATTEMPTS,
    WEB_FETCH_RETRY_DELAY_SECONDS,
    WEB_FETCH_USER_AGENT,
)

SUPPORTED_WEB_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "application/pdf"})


def fetch_page(url: str) -> tuple[str, str, bytes] | None:
    """Return supported content from an HTTP/S URL without embedded credentials."""

    for attempt_index in range(WEB_FETCH_ATTEMPTS):
        try:
            parsed_url = urlsplit(url)
            if (
                parsed_url.scheme not in {"http", "https"}
                or parsed_url.hostname is None
                or parsed_url.username is not None
                or parsed_url.password is not None
            ):
                return None

            with urlopen(  # noqa: S310
                Request(url, headers={"User-Agent": WEB_FETCH_USER_AGENT}),
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            ) as response:
                content_type = response.headers.get_content_type().lower()
                url = response.geturl()
                if content_type not in SUPPORTED_WEB_CONTENT_TYPES:
                    return None
                return url, content_type, response.read()
        except ValueError:
            return None
        except OSError as error:
            if (
                isinstance(error, HTTPError)
                and error.code != HTTPStatus.TOO_MANY_REQUESTS
                and error.code < HTTPStatus.INTERNAL_SERVER_ERROR
            ):
                return None

        if attempt_index < WEB_FETCH_ATTEMPTS - 1:
            sleep(WEB_FETCH_RETRY_DELAY_SECONDS)
    return None
