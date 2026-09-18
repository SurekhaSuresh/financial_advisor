"""Shared URL validation for search-result filtering and original-page fetching."""

from urllib.parse import urlsplit


def url_matches_allowed_domain(url: str, allowed_domains: tuple[str, ...]) -> bool:
    """Return whether a URL host exactly matches or is a subdomain of an allowed host."""

    hostname = urlsplit(url).hostname
    if hostname is None:
        return False
    host = hostname.lower().rstrip(".")
    return any(
        host == domain.lower().rstrip(".") or host.endswith(f".{domain.lower().rstrip('.')}")
        for domain in allowed_domains
    )


def validate_http_url_policy(url: str, allowed_domains: tuple[str, ...]) -> None:
    """Require HTTP/S and, when configured, a URL from an approved domain."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("URL must use HTTP or HTTPS.")
    if allowed_domains and not url_matches_allowed_domain(url, allowed_domains):
        raise ValueError("URL is outside the approved domain policy.")
