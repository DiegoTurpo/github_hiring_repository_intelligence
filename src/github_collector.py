"""GitHub API extraction — collects repository-level signals.

Sampling strategy
-----------------
We do NOT assign categories here. The goal of sampling is *diversity*: we pull
repositories across several strata so the dataset contains a wide range of
engineering maturity. The LLM (later stage) is what assigns the actual labels.

Strata (search queries) are chosen to span the maturity spectrum:
  - very popular, multi-contributor projects  -> likely senior / lead
  - mid-popularity projects                   -> likely junior / senior
  - tiny / inactive projects                  -> likely intern / low-value
  - boilerplate / starter templates           -> likely template/boilerplate

For each repository we extract 13 signals via the REST API.

Run:
    python src/github_collector.py
Output:
    data/raw/repos.csv
"""

import os
import re
import time
import base64
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Each stratum: (search query, how many repos to pull). Tune REPOS_PER_QUERY
# to control dataset size and API usage.
SEARCH_STRATA = [
    ("stars:>20000 language:python", 85),       # mature / lead-level
    ("stars:1000..5000 language:python", 85),   # senior-level
    ("stars:50..300 language:python", 90),      # junior-level
    ("stars:0..3 language:python", 90),         # intern / low-value
    ("boilerplate in:name stars:>20", 80),      # templates / boilerplate
    ("starter-template in:name", 80),           # starter kits / replicas
]

PER_PAGE = 50


def _check_rate_limit(resp):
    """Sleep until reset if we are about to hit the rate limit."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None and int(remaining) <= 1:
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - time.time(), 0) + 2
        print(f"  rate limit reached, sleeping {wait:.0f}s ...")
        time.sleep(wait)


def gh_get(url, params=None):
    """GET with auth, retry on 403 secondary rate limit."""
    for attempt in range(4):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            _check_rate_limit(resp)
            continue
        _check_rate_limit(resp)
        return resp
    return resp


def search_repositories(query, max_repos):
    """Return a list of repo objects for a search query."""
    repos = []
    page = 1
    while len(repos) < max_repos:
        resp = gh_get(
            f"{API}/search/repositories",
            params={"q": query, "per_page": PER_PAGE, "page": page,
                    "sort": "updated", "order": "desc"},
        )
        if resp.status_code != 200:
            print(f"  search failed ({resp.status_code}) for: {query}")
            break
        items = resp.json().get("items", [])
        if not items:
            break
        repos.extend(items)
        page += 1
        time.sleep(2)  # respect search API secondary limits
    return repos[:max_repos]


def count_via_link_header(url):
    """Many list endpoints expose total count through the Link 'last' page.
    We request per_page=1 and read the last page number = total count."""
    resp = gh_get(url, params={"per_page": 1})
    if resp.status_code != 200:
        return 0
    link = resp.headers.get("Link", "")
    for part in link.split(","):
        if 'rel="last"' in part:
            # match page= preceded by ? or & so we skip 'per_page='
            m = re.search(r"[?&]page=(\d+)", part)
            if m:
                return int(m.group(1))
    # no 'last' page -> the single page holds all items
    data = resp.json()
    return len(data) if isinstance(data, list) else 0


def get_readme_length(full_name):
    resp = gh_get(f"{API}/repos/{full_name}/readme")
    if resp.status_code != 200:
        return 0
    content = resp.json().get("content", "")
    try:
        return len(base64.b64decode(content).decode("utf-8", errors="ignore"))
    except Exception:
        return 0


def has_ci_workflows(full_name):
    resp = gh_get(f"{API}/repos/{full_name}/contents/.github/workflows")
    return resp.status_code == 200 and len(resp.json()) > 0


def days_between(iso_a, iso_b):
    a = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
    b = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
    return (b - a).days


def extract_signals(repo):
    """Extract the 13 repository-level signals for one repo object."""
    full = repo["full_name"]
    now = datetime.now(timezone.utc).isoformat()

    contributors = count_via_link_header(f"{API}/repos/{full}/contributors")
    commits = count_via_link_header(f"{API}/repos/{full}/commits")
    releases = count_via_link_header(f"{API}/repos/{full}/releases")

    return {
        "full_name": full,
        "url": repo["html_url"],
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "contributors": contributors,
        "commits": commits,
        "releases": releases,
        "has_ci": int(has_ci_workflows(full)),
        "readme_length": get_readme_length(full),
        "n_topics": len(repo.get("topics", [])),
        "has_license": int(repo.get("license") is not None),
        "language": repo.get("language") or "unknown",
        "age_days": days_between(repo["created_at"], now),
        "days_since_push": days_between(repo["pushed_at"], now),
    }


def main():
    if not GITHUB_TOKEN or GITHUB_TOKEN == "your_github_token_here":
        raise SystemExit("Set GITHUB_TOKEN in your .env file first.")

    # 1) gather candidate repos from all strata (dedup by full_name)
    seen = set()
    candidates = []
    for query, n in SEARCH_STRATA:
        print(f"Searching: {query}  (target {n})")
        for repo in search_repositories(query, n):
            if repo["full_name"] not in seen:
                seen.add(repo["full_name"])
                candidates.append(repo)
    print(f"\nTotal unique candidate repos: {len(candidates)}")

    # 2) enrich each repo with the per-repo signals
    rows = []
    for repo in tqdm(candidates, desc="Extracting signals"):
        try:
            rows.append(extract_signals(repo))
        except Exception as e:
            print(f"  skipped {repo['full_name']}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/repos.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} repositories -> {out}")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
