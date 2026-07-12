"""
Hard-negative augmentation: the model was flagging ordinary legitimate
deep links (github.com/x, docs.python.org/x, stackoverflow.com/questions/x)
as phishing because the legit training URLs skewed toward an older
torrent/wiki-style corpus. This generates realistic legit URLs across a
broad set of well-known domains and common real-world path patterns, to
teach the model what a normal legitimate deep link looks like.
"""

import random
import pandas as pd

random.seed(42)

TOP_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "instagram.com", "twitter.com",
    "x.com", "wikipedia.org", "amazon.com", "reddit.com", "linkedin.com",
    "github.com", "stackoverflow.com", "microsoft.com", "apple.com", "netflix.com",
    "yahoo.com", "office.com", "live.com", "bing.com", "cnn.com", "nytimes.com",
    "bbc.com", "espn.com", "twitch.tv", "pinterest.com", "wordpress.com",
    "blogspot.com", "adobe.com", "salesforce.com", "zoom.us", "dropbox.com",
    "spotify.com", "ebay.com", "walmart.com", "target.com", "bestbuy.com",
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com", "irs.gov",
    "usa.gov", "medium.com", "quora.com", "imdb.com", "yelp.com", "tripadvisor.com",
    "booking.com", "airbnb.com", "uber.com", "lyft.com", "doordash.com",
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org", "docs.python.org",
    "developer.mozilla.org", "w3schools.com", "npmjs.com", "pypi.org", "docker.com",
    "kubernetes.io", "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
    "nature.com", "sciencedirect.com", "researchgate.net", "arxiv.org", "ieee.org",
    "who.int", "un.org", "worldbank.org", "nasa.gov", "cdc.gov", "nih.gov",
    "weather.com", "accuweather.com", "craigslist.org", "indeed.com", "glassdoor.com",
    "monster.com", "coinbase.com", "binance.com", "shopify.com", "etsy.com",
    "steamcommunity.com", "epicgames.com", "playstation.com", "xbox.com",
    "samsung.com", "sony.com", "dell.com", "hp.com", "intel.com", "nvidia.com",
    "slack.com", "notion.so", "trello.com", "atlassian.com", "gitlab.com",
    "bitbucket.org", "digitalocean.com", "heroku.com", "cloudflare.com",
    "mail.google.com", "drive.google.com", "calendar.google.com", "docs.google.com",
    "outlook.com", "protonmail.com", "icloud.com", "dropbox.com", "box.com",
    "figma.com", "canva.com", "airtable.com", "asana.com", "monday.com",
    "webmd.com", "mayoclinic.org", "healthline.com", "npr.org", "reuters.com",
    "bloomberg.com", "wsj.com", "forbes.com", "techcrunch.com", "theverge.com",
    "wired.com", "arstechnica.com", "engadget.com", "cnbc.com", "marketwatch.com",
    "investopedia.com", "coursera.org", "skillshare.com", "duolingo.com",
    "codecademy.com", "freecodecamp.org", "leetcode.com", "hackerrank.com",
    "kaggle.com", "huggingface.co", "openai.com", "anthropic.com", "meta.com",
    "tiktok.com", "snapchat.com", "discord.com", "telegram.org", "whatsapp.com",
    "vimeo.com", "soundcloud.com", "bandcamp.com", "goodreads.com", "audible.com",
    "chess.com", "duckduckgo.com", "yandex.com", "baidu.com", "alibaba.com",
    "aliexpress.com", "wish.com", "zillow.com", "redfin.com", "realtor.com",
    "expedia.com", "kayak.com", "hotels.com", "marriott.com", "hilton.com",
    "delta.com", "united.com", "southwest.com", "americanexpress.com",
    "capitalone.com", "discover.com", "venmo.com", "cashapp.com", "robinhood.com",
    "fidelity.com", "vanguard.com", "schwab.com", "etrade.com", "mint.com",
    "turbotax.com", "hrblock.com", "sec.gov", "fbi.gov", "state.gov",
    "congress.gov", "supremecourt.gov", "loc.gov", "archives.gov",
    "smithsonian.org", "metmuseum.org", "nps.gov", "noaa.gov", "usgs.gov",
]

EXTRA_TLD_DOMAINS = [
    "example.io", "startup.co", "myapp.dev", "service.app", "cloudhost.net",
    "university.edu", "college.edu", "hospital.org", "clinic.org", "museum.org",
    "council.gov.uk", "parliament.uk", "gov.au", "canada.ca", "europa.eu",
]
TOP_DOMAINS = TOP_DOMAINS + EXTRA_TLD_DOMAINS

PATH_TEMPLATES = [
    "/about", "/contact", "/help", "/support", "/faq", "/terms", "/privacy",
    "/blog/{slug}", "/news/{slug}", "/article/{slug}", "/{year}/{month:02d}/{day:02d}/{slug}",
    "/user/{id}", "/profile/{id}", "/u/{id}", "/product/{id}", "/item/{id}",
    "/p/{id}", "/dp/{id}", "/questions/{id}/{slug}", "/wiki/{slug}",
    "/docs/{slug}", "/api/{slug}", "/reference/{slug}", "/tutorial/{slug}",
    "/search?q={slug}", "/category/{slug}", "/tag/{slug}", "/topic/{slug}",
    "/watch?v={id}", "/status/{id}", "/posts/{id}", "/videos/{id}",
    "/jobs/{id}", "/careers", "/pricing", "/download", "/signup", "/login",
    "/settings", "/dashboard", "/{slug}/{id}", "/repos/{slug}", "/issues/{id}",
    "/pull/{id}", "/releases/{slug}",
]

SLUGS = [
    "getting-started", "how-to-reset-password", "top-10-tips", "annual-report-2025",
    "machine-learning-basics", "climate-change-impact", "new-features-announcement",
    "quarterly-earnings", "product-launch", "developer-guide", "installation-guide",
    "python-tutorial", "data-structures", "world-news-today", "market-update",
    "health-and-wellness", "travel-guide-2026", "recipe-collection", "movie-review",
    "electric-vehicles", "space-exploration", "startup-funding-news",
    "Machine_learning", "Artificial_intelligence", "Python_(programming_language)",
    "United_States", "World_War_II", "Climate_change", "pandas-merge-dataframes",
    "react-hooks-explained", "docker-compose-tutorial", "rest-api-best-practices",
    "understanding-oauth2", "css-grid-vs-flexbox", "kubernetes-deployment-guide",
]


def _rand_slug():
    s = random.choice(SLUGS)
    if random.random() < 0.3:
        s = s + "-" + str(random.randint(1, 99))
    return s


def _rand_id():
    choices = [
        str(random.randint(1000, 999999)),
        f"{random.randint(1,999)}x{random.randint(1,999)}",
        "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(6, 11))),
        str(random.randint(1, 50)),
    ]
    return random.choice(choices)


def generate_legit_urls(n=8000):
    urls = set()
    attempts = 0
    while len(urls) < n and attempts < n * 5:
        attempts += 1
        domain = random.choice(TOP_DOMAINS)
        template = random.choice(PATH_TEMPLATES)
        path = template.format(
            slug=_rand_slug(),
            id=_rand_id(),
            year=random.randint(2018, 2026),
            month=random.randint(1, 12),
            day=random.randint(1, 28),
        )
        scheme = "https"
        www = random.choice(["www.", ""])
        url = f"{scheme}://{www}{domain}{path}"
        urls.add(url)
    return list(urls)


if __name__ == "__main__":
    urls = generate_legit_urls(30000)
    df = pd.DataFrame({"url": urls, "label": 0})
    df.to_csv("data/augmented_legit_urls.csv", index=False)
    print(f"Generated {len(df)} augmented legit URLs")
    print(df.sample(10, random_state=1))
