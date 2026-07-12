"""
Lexical & structural feature extraction for URL phishing classification.

All features are computed directly from the raw URL string -- no external
lookups (WHOIS, DNS, live requests) -- so the pipeline stays fast, offline,
and reproducible.
"""

import re
import math
from urllib.parse import urlparse
import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "signin", "bank", "password", "pay", "billing", "suspend",
    "webscr", "ebayisapi", "wallet", "recover", "unlock", "alert",
]

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "shorte.st", "rebrand.ly",
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> dict:
    url = str(url).strip()

    # Ensure a scheme so urlparse behaves consistently
    parse_target = url if "://" in url else "http://" + url
    parsed = urlparse(parse_target)

    domain = parsed.netloc.split(":")[0].lower()
    path = parsed.path or ""
    query = parsed.query or ""

    domain_no_www = domain[4:] if domain.startswith("www.") else domain
    subdomain_parts = domain_no_www.split(".")
    tld = subdomain_parts[-1] if len(subdomain_parts) > 1 else ""

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),

        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_letters": sum(c.isalpha() for c in url),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9./:_-]", url)),
        "num_params": query.count("&") + (1 if query else 0),
        "num_fragments": 1 if parsed.fragment else 0,

        "has_at_symbol": int("@" in url),
        "has_ip_address": int(bool(IP_PATTERN.match(domain))),
        "has_https": int(parsed.scheme == "https"),
        "has_http_in_path": int("http" in path.lower()),
        "has_double_slash_redirect": int("//" in path),
        "has_port": int(":" in parsed.netloc and not IP_PATTERN.match(domain)),

        "num_subdomains": max(len(subdomain_parts) - 2, 0),
        "tld_length": len(tld),

        "is_shortened": int(domain_no_www in SHORTENERS),
        "suspicious_word_count": sum(
            1 for w in SUSPICIOUS_WORDS if w in url.lower()
        ),

        "digit_ratio": (sum(c.isdigit() for c in url) / len(url)) if url else 0,
        "letter_ratio": (sum(c.isalpha() for c in url) / len(url)) if url else 0,
        "domain_entropy": _shannon_entropy(domain),

        "domain_has_digits": int(any(c.isdigit() for c in domain_no_www)),
        "domain_hyphen_count": domain.count("-"),
    }
    return features


def build_feature_dataframe(urls: pd.Series) -> pd.DataFrame:
    """Vectorized-ish wrapper: apply extract_features across a Series of URLs."""
    records = urls.apply(extract_features)
    return pd.DataFrame(list(records))


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify-account.php",
        "http://paypal-secure-login.verify-account.tk/webscr?cmd=login",
        "https://bit.ly/3xYzAbC",
    ]
    for s in samples:
        print(s)
        print(extract_features(s))
        print()
