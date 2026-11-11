#!/usr/bin/env python3
"""Fetch merged perf-related PRs from vllm/sglang via the GitHub API.

Two modes:
  1. (default) search mode : discover candidate PRs (performance label +
     perf-related title keywords), save raw jsonl per repo.
  2. --enrich <jsonl>      : fill merged_at/additions/deletions/changed_files
     for the PRs listed in an existing jsonl (used for the major tier before
     evidence-card generation).

Token is read from the GITHUB_TOKEN env var and never written to disk.
Stdlib only; resume-safe (already-fetched PR numbers are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

REPOS = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
}

# Extra labels worth capturing beyond `performance`.
EXTRA_LABELS = {
    "vllm": [
        "quantization",
        "speculative-decoding",
        "torch.compile",
        "kv-cache-manager",
        "kv-connector",
        "scheduler",
    ],
    "sglang": [
        "kernel",
        "sgl-kernel",
        "quant",
        "quantization",
        "release-highlight",
        "flashinfer",
        "jit-kernel",
    ],
}

# Title keywords. The legacy issues search API does not support parentheses
# or reliable OR precedence, so each keyword runs as its own query.
TITLE_KEYWORDS = [
    "perf",
    "speedup",
    "faster",
    "optimize",
    "optimization",
    "optimized",
    "fused",
    "fusion",
    "overlap",
    "cuda graph",
    "flashinfer",
    "deepep",
    "speculative",
    "eagle",
    "mtp",
    "quantization",
    "quantize",
    "kernel",
]

SEARCH_WINDOW_START = date(2023, 1, 1)
PAGE_SIZE = 100
MAX_SEARCH_PAGES = 10  # per query window
SEARCH_INTERVAL_S = 2.1  # 30 searches/min authenticated budget
BODY_LIMIT = 30000  # chars of PR body kept in raw data

API = "https://api.github.com"


class GitHub:
    def __init__(self, token: str) -> None:
        self.token = token
        self.search_calls = 0
        self.core_calls = 0

    def _request(self, url: str, accept: str = "application/vnd.github+json"):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", accept)
        req.add_header("User-Agent", "perf-pr-index-fetcher")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        for attempt in range(5):
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                payload = resp.read()
                return json.loads(payload.decode("utf-8")), remaining, reset
            except urllib.error.HTTPError as e:
                if e.code in (403, 429) or e.code >= 500:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    reset = e.headers.get("X-RateLimit-Reset") if e.headers else None
                    if e.code == 403 and (e.headers or {}).get(
                        "X-RateLimit-Remaining"
                    ) != "0":
                        # Secondary limit or transient; short backoff.
                        time.sleep(10 + attempt * 10)
                        continue
                    wait = _wait_seconds(retry_after, reset)
                    print(f"  rate limited, sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if e.code == 404:
                    return None, "0", None
                if e.code == 422:
                    raise RuntimeError(f"GitHub rejected the query (422): {url}")
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  network error ({e}), retrying...", flush=True)
                time.sleep(5 + attempt * 5)
        raise RuntimeError(f"request failed after retries: {url}")

    def get(self, path_or_url: str):
        url = path_or_url if path_or_url.startswith("http") else API + path_or_url
        data, remaining, _ = self._request(url)
        self.core_calls += 1
        if remaining and remaining.isdigit() and int(remaining) < 100:
            reset_ts = int(_core_reset or time.time() + 900)
            wait = max(1, reset_ts - int(time.time()))
            print(
                f"  core rate limit low ({remaining} left), sleeping {wait}s",
                flush=True,
            )
            time.sleep(min(wait, 3600))
        return data

    def search(self, query: str):
        """Yield all issue items for a search query, respecting the 1000-result
        cap by splitting on merged-date windows when needed."""
        total = self._search_count(query)
        if total <= 950:
            yield from self._search_page_all(query)
            return
        # Split by merged-date quarters until each slice fits.
        for start, end in _quarter_slices():
            sub = f"{query} merged:{start}..{end}"
            n = self._search_count(sub)
            if n > 950:
                for m_start, m_end in _month_slices(start, end):
                    sub2 = f"{query} merged:{m_start}..{m_end}"
                    n2 = self._search_count(sub2)
                    if n2 > 950:
                        print(
                            f"  WARNING: {sub2} still has {n2} results, "
                            "truncating to first 1000",
                            flush=True,
                        )
                    yield from self._search_page_all(sub2)
            else:
                yield from self._search_page_all(sub)

    def _search_count(self, query: str) -> int:
        url = f"{API}/search/issues?per_page=1&q={urllib.parse.quote(query)}"
        data, _, _ = self._request(url)
        self.search_calls += 1
        time.sleep(SEARCH_INTERVAL_S)
        return data.get("total_count", 0)

    def _search_page_all(self, query: str):
        q = urllib.parse.quote(query)
        for page in range(1, MAX_SEARCH_PAGES + 1):
            url = f"{API}/search/issues?per_page={PAGE_SIZE}&page={page}&q={q}"
            data, _, _ = self._request(url)
            self.search_calls += 1
            items = data.get("items", [])
            yield from items
            if len(items) < PAGE_SIZE:
                break
            time.sleep(SEARCH_INTERVAL_S)
        time.sleep(SEARCH_INTERVAL_S)


_core_reset: str | None = None


def _wait_seconds(retry_after: str | None, reset: str | None) -> int:
    if retry_after:
        try:
            return int(retry_after) + 2
        except ValueError:
            pass
    if reset:
        try:
            return max(5, int(reset) - int(time.time()) + 2)
        except ValueError:
            pass
    return 60


def _quarter_slices():
    out = []
    year = SEARCH_WINDOW_START.year
    month = 1
    today = date.today()
    while date(year, month, 1) <= today:
        # quarter = [month, month+2]; end is the 1st day of the following month
        end = date(year + (month + 2) // 12, (month + 2) % 12 + 1, 1)
        out.append(
            (SEARCH_WINDOW_START.isoformat() if len(out) == 0 else f"{year}-{month:02d}-01", end.isoformat())
        )
        year, month = (year + 1, 1) if month == 10 else (year, month + 3)
    return out


def _month_slices(start_iso: str, end_iso: str):
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    cur = start
    while cur < end:
        nxt = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield cur.isoformat(), min(nxt, end).isoformat()
        cur = nxt


def build_search_queries(repo_full: str) -> list[str]:
    base = f"repo:{repo_full} is:pr is:merged"
    queries = [f"{base} label:performance"]
    queries += [f'{base} "{kw}" in:title' for kw in TITLE_KEYWORDS]
    queries += [f"{base} label:{lab}" for lab in EXTRA_LABELS.get(repo_key(repo_full), [])]
    return queries


def repo_key(repo_full: str) -> str:
    return "vllm" if "vllm" in repo_full else "sglang"


def item_to_record(item: dict, repo_full: str) -> dict:
    labels = [l["name"] for l in item.get("labels", [])]
    body = item.get("body") or ""
    if len(body) > BODY_LIMIT:
        body = body[:BODY_LIMIT]
    return {
        "repo": repo_key(repo_full),
        "repo_full": repo_full,
        "number": item["number"],
        "title": item.get("title") or "",
        "author": (item.get("user") or {}).get("login") or "",
        "labels": labels,
        "state": item.get("state"),
        "created_at": item.get("created_at"),
        "closed_at": item.get("closed_at"),
        "merged_at": None,  # filled by --enrich for the major tier
        "comments": item.get("comments", 0),
        "reactions_total": (item.get("reactions") or {}).get("total_count", 0),
        "additions": None,
        "deletions": None,
        "changed_files": None,
        "merge_commit_sha": None,
        "body": body,
        "html_url": item.get("html_url"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_existing(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:  # iterate; don't use splitlines() (U+2028 traps)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line from an interrupted run
                records[rec["number"]] = rec
    return records


def _dumps(rec: dict) -> str:
    # U+2028/U+2029 are legal JSON chars that break naive line splitting later
    return (
        json.dumps(rec, ensure_ascii=False)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def append_records(path: Path, records: dict[int, dict]) -> int:
    """Append-only write of records not yet on disk (by number)."""
    existing = load_existing(path)
    new = {n: r for n, r in records.items() if n not in existing}
    if not new:
        return 0
    tmp = path.with_suffix(".tmp")
    with tmp.open("a", encoding="utf-8") as f:
        for rec in new.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # merge tmp into the canonical file (existing + new), deduped, sorted
    merged = {**existing, **new}
    for attempt in range(5):  # Windows: AV/indexer may briefly lock the file
        try:
            with path.open("w", encoding="utf-8") as f:
                for num in sorted(merged):
                    f.write(json.dumps(merged[num], ensure_ascii=False) + "\n")
            break
        except PermissionError:
            if attempt == 4:
                raise
            print(f"    ! target locked, retrying save ({attempt + 2}/5)", flush=True)
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return len(new)


def run_search(repo_full: str, gh: GitHub) -> dict[int, dict]:
    key = repo_key(repo_full)
    out_path = RAW_DIR / f"{key}-prs.jsonl"
    records = load_existing(out_path)
    print(f"[{key}] {len(records)} PRs already in {out_path.name}, resuming", flush=True)
    queries = build_search_queries(repo_full)
    print(f"[{key}] {len(queries)} search queries", flush=True)
    seen_this_run = 0
    for i, q in enumerate(queries, 1):
        print(f"[{key}] query {i}/{len(queries)}: {q}", flush=True)
        for item in gh.search(q):
            if item.get("pull_request") is None:
                continue
            rec = item_to_record(item, repo_full)
            num = rec["number"]
            if num in records:
                # merge newly-seen labels into the existing record
                old = records[num]
                merged_labels = sorted(set(old["labels"]) | set(rec["labels"]))
                old["labels"] = merged_labels
                continue
            records[num] = rec
            seen_this_run += 1
            if seen_this_run % 200 == 0:
                print(f"[{key}] {len(records)} candidates so far", flush=True)
        print(f"[{key}] candidates after query {i}: {len(records)}", flush=True)
        append_records(out_path, records)
    n = append_records(out_path, records)
    print(f"[{key}] saved, +{n} new, total {len(records)} -> {out_path}", flush=True)
    return records


def run_enrich(jsonl_path: Path, gh: GitHub) -> None:
    records = load_existing(jsonl_path)
    todo = [
        r
        for r in records.values()
        if r.get("merged_at") is None or r.get("additions") is None
    ]
    print(f"enrich {jsonl_path.name}: {len(todo)} of {len(records)} PRs need enrichment", flush=True)
    done = 0
    for rec in records.values():
        if rec.get("merged_at") is not None and rec.get("additions") is not None:
            continue
        pr = gh.get(f"/repos/{rec['repo_full']}/pulls/{rec['number']}")
        if pr is None:
            print(f"  WARN: PR #{rec['number']} not found", flush=True)
            continue
        rec["merged_at"] = pr.get("merged_at")
        rec["merge_commit_sha"] = pr.get("merge_commit_sha")
        rec["additions"] = pr.get("additions")
        rec["deletions"] = pr.get("deletions")
        rec["changed_files"] = pr.get("changed_files")
        done += 1
        if done % 50 == 0:
            print(f"  enriched {done}/{len(todo)}", flush=True)
            append_records(jsonl_path, records)
    append_records(jsonl_path, records)
    print(f"enrich done: {done} updated -> {jsonl_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--enrich", type=Path, help="jsonl to enrich with /pulls data")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: set the GITHUB_TOKEN env var (token is never written to disk)")

    gh = GitHub(token)
    t0 = time.time()
    try:
        if args.enrich:
            run_enrich(args.enrich, gh)
        else:
            repos = list(REPOS.values()) if args.repo == "both" else [REPOS[args.repo]]
            for repo_full in repos:
                run_search(repo_full, gh)
    finally:
        dt = time.time() - t0
        print(
            f"done in {dt/60:.1f} min | search calls: {gh.search_calls} | "
            f"core calls: {gh.core_calls}",
            flush=True,
        )


if __name__ == "__main__":
    main()
