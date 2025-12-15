#!/usr/bin/env python3
"""
OpenAlex multi-pass dataset builder for:
Continual Inference Caching Across Sessions (LLM serving)

Fixes:
- avoids "freezing" by limiting per-pass downloads
- narrows overly-broad passes (e.g., "hidden state")
- prints progress
- checkpoints after each pass
- merges + deduplicates by DOI/OpenAlex ID
- creates output folders automatically

Run:
  python openalex_script.py
Outputs:
  data_raw/openalex_llm_caching_2010_2025_merged.csv
  data_raw/_checkpoint_pass*.csv
  data_raw/_checkpoint_merged_so_far.csv
"""

import os
import csv
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# OpenAlex endpoints
# -------------------------
BASE_WORKS = "https://api.openalex.org/works"
BASE_CONCEPTS = "https://api.openalex.org/concepts"

# -------------------------
# Settings
# -------------------------
MAILTO = "berke.palamutcu10@euvt.ro"
PER_PAGE = 200
SLEEP = 0.15

# Hard safety limits (prevents “freeze”)
MAX_RECORDS_PER_PASS = 3000          # increase later (e.g., 10000) if you want
PRINT_EVERY_PAGES = 10
CHECKPOINT_EVERY_PAGES = 25

# -------------------------
# Your constraints
# -------------------------
YEAR_FILTER = "publication_year:2010-2025"
LANG_FILTER = "language:en"

# Subject area: Computer Science + Artificial Intelligence (OpenAlex concept filter)
USE_CONCEPT_FILTER = True
CONCEPT_NAMES = ["computer science", "artificial intelligence"]

# -------------------------
# Multi-pass search strategy
# -------------------------
# OpenAlex "search" is not strict Boolean like Scopus.
# So we use multiple passes and later merge + dedupe.
SEARCH_PASSES = [
    # Pass 1: Broad, still relevant
    "large language model LLM transformer inference serving cache caching",

    # Pass 2: KV cache terms
    '"KV cache" OR "key value cache" OR "key-value cache" OR "attention cache"',

    # Pass 3: Prefix/prompt caching
    '"prefix cache" OR "prefix caching" OR "prompt cache" OR "prompt caching" OR "prefix sharing"',

    # Pass 4: Intermediate/activation reuse but anchored to LLM inference/serving
    '("activation cache" OR "activation caching" OR "intermediate state" OR "hidden state reuse") '
    '(LLM OR "large language model" OR transformer) (inference OR serving OR deployment)',

    # Pass 5: Cross-session / multi-tenant / shared cache (your novelty)
    '"cross-session" OR "across sessions" OR "multi-tenant" OR "shared cache" OR "global cache"',

    # Pass 6: Serving frameworks/inference engines often cited in systems work
    'vLLM OR "TensorRT-LLM" OR "inference engine" OR "model serving" OR "LLM serving" OR "continuous batching"',

    # Pass 7: Safety / privacy / poisoning
    'privacy OR "differential privacy" OR leakage OR "information leakage" OR "cache poisoning" OR isolation OR coherence',
]

# Output paths
OUT_PATH = "data_raw/openalex_llm_caching_2010_2025_merged.csv"
MERGED_CHECKPOINT_PATH = "data_raw/_checkpoint_merged_so_far.csv"


# -------------------------
# HTTP session with retries
# -------------------------
def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(session: requests.Session, url: str, params: dict, timeout: int = 60) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------------
# Concept resolution (CS + AI)
# -------------------------
def find_top_concept_id(session: requests.Session, concept_query: str) -> str:
    params = {"search": concept_query, "per-page": 5, "mailto": MAILTO}
    data = get_json(session, BASE_CONCEPTS, params)
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No concept results for query: {concept_query}")

    q_lower = concept_query.lower().strip()
    chosen = results[0]
    for c in results:
        name = (c.get("display_name") or "").lower().strip()
        if name == q_lower:
            chosen = c
            break

    return chosen["id"].split("/")[-1]


def build_filter(session: requests.Session) -> str:
    filters = [YEAR_FILTER, LANG_FILTER]

    if USE_CONCEPT_FILTER:
        concept_ids = []
        for name in CONCEPT_NAMES:
            cid = find_top_concept_id(session, name)
            concept_ids.append(cid)
            print(f"[concept] {name} -> {cid}")
        # OR between concepts: "|"
        filters.append("concept.id:" + "|".join(concept_ids))

    # AND between different filters: ","
    return ",".join(filters)


# -------------------------
# Flatten OpenAlex work record
# -------------------------
def flatten(work: dict) -> dict:
    authorships = work.get("authorships", []) or []
    concepts = work.get("concepts", []) or []
    keywords = work.get("keywords", []) or []

    # Venue
    venue = ""
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if isinstance(source, dict):
        venue = source.get("display_name") or ""

    # Authors
    authors = "; ".join(
        (a.get("author", {}) or {}).get("display_name", "")
        for a in authorships
        if (a.get("author", {}) or {}).get("display_name")
    )

    # Institutions / countries
    institutions = set()
    countries = set()
    for a in authorships:
        for inst in (a.get("institutions") or []):
            if inst.get("display_name"):
                institutions.add(inst["display_name"])
            if inst.get("country_code"):
                countries.add(inst["country_code"])

    concept_names = "; ".join(c.get("display_name", "") for c in concepts if c.get("display_name"))
    keyword_names = "; ".join(k.get("display_name", "") for k in keywords if k.get("display_name"))

    work_id = (work.get("id", "") or "").split("/")[-1]
    doi = (work.get("doi") or "")
    if isinstance(doi, str):
        doi = doi.lower().strip()

    return {
        "openalex_id": work_id,
        "doi": doi,
        "title": work.get("display_name", "") or "",
        "publication_year": work.get("publication_year", "") or "",
        "publication_date": work.get("publication_date", "") or "",
        "cited_by_count": work.get("cited_by_count", 0) or 0,
        "type": work.get("type", "") or "",
        "venue": venue,
        "authors": authors,
        "institutions": "; ".join(sorted(institutions)),
        "countries": "; ".join(sorted(countries)),
        "concepts": concept_names,
        "keywords": keyword_names,
    }


# -------------------------
# Deduplication
# -------------------------
def dedupe(rows: list[dict]) -> list[dict]:
    """
    Prefer DOI if available; else OpenAlex ID.
    Keeps first occurrence.
    """
    seen = set()
    out = []
    for r in rows:
        key = r["doi"] if r.get("doi") else f"OA:{r.get('openalex_id')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# -------------------------
# CSV writing
# -------------------------
def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "openalex_id", "doi", "title", "publication_year", "publication_date",
        "cited_by_count", "type", "venue", "authors", "institutions", "countries",
        "concepts", "keywords"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Download a single pass (cursor paging + caps + progress + checkpoints)
# -------------------------
def download_pass(session: requests.Session, filt: str, search: str, pass_index: int) -> list[dict]:
    # Test first
    test_params = {"filter": filt, "search": search, "per-page": 5, "mailto": MAILTO}
    test = get_json(session, BASE_WORKS, test_params)
    count = (test.get("meta") or {}).get("count", 0)
    print(f"[pass test] #{pass_index} count={count} search={search[:80]}...")

    rows = []
    cursor = "*"
    page = 0

    while True:
        page += 1
        params = {
            "filter": filt,
            "search": search,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": MAILTO,
        }

        data = get_json(session, BASE_WORKS, params)
        results = data.get("results", []) or []
        meta = data.get("meta") or {}
        next_cursor = meta.get("next_cursor")

        if page == 1:
            print(f"[pass {pass_index}] page1 got={len(results)}")

        if not results:
            break

        for w in results:
            rows.append(flatten(w))
            if len(rows) >= MAX_RECORDS_PER_PASS:
                print(f"[pass {pass_index}] reached MAX_RECORDS_PER_PASS={MAX_RECORDS_PER_PASS}, stopping pass early.")
                return rows

        if page % PRINT_EVERY_PAGES == 0:
            print(f"[pass {pass_index}] page={page} downloaded={len(rows)}")

        if page % CHECKPOINT_EVERY_PAGES == 0:
            tmp_path = f"data_raw/_checkpoint_pass{pass_index}.csv"
            write_csv(tmp_path, dedupe(rows))
            print(f"[pass {pass_index}] checkpoint saved: {tmp_path}")

        if not next_cursor:
            break

        cursor = next_cursor
        time.sleep(SLEEP)

    return rows


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("data_raw", exist_ok=True)

    session = make_session()
    filt = build_filter(session)

    all_rows: list[dict] = []

    for i, search in enumerate(SEARCH_PASSES, start=1):
        print(f"\n=== PASS {i}/{len(SEARCH_PASSES)} ===")
        rows = download_pass(session, filt, search, i)
        print(f"[pass {i}] downloaded rows={len(rows)}")

        all_rows.extend(rows)

        merged_so_far = dedupe(all_rows)
        write_csv(MERGED_CHECKPOINT_PATH, merged_so_far)
        print(f"[merged checkpoint] records so far: {len(merged_so_far)}")

    merged = dedupe(all_rows)
    write_csv(OUT_PATH, merged)
    print(f"\nSaved: {OUT_PATH} records: {len(merged)}")


if __name__ == "__main__":
    main()
