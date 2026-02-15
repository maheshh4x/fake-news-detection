from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "apnews.com", "thehindu.com", "ndtv.com",
    "timesofindia.indiatimes.com", "cnn.com", "nytimes.com", "washingtonpost.com"
]

SUSPICIOUS_DOMAINS = [
    "healthtruthexposed.info", "globaltruthers.biz", "worldnewssource.xyz"
]

def check_domain(url: str):
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
    except:
        return "unknown"

    if any(td in domain for td in TRUSTED_DOMAINS):
        return "trusted"
    if any(sd in domain for sd in SUSPICIOUS_DOMAINS):
        return "suspicious"

    return "unknown"
