#!/usr/bin/env python3
"""Fetch ALL merged bug-fix PRs (label:bug UNION 'fix' in:title UNION 'bug'
in:title) from vllm-project/vllm and sgl-project/sglang.

Output: data/raw/<repo>-bugfix-prs.jsonl (one PR per line, body capped at 30k)

Search API caps one query at 1000 results -> every query is sliced into
quarter windows (merged:A..B); a quarter over 1000 is auto-split into months.
Data is saved after EVERY query (crash-safe, resume by PR number).
The search API occasionally answers 422 under secondary rate limiting —
retries use long backoff and the response body is logged for diagnosis.
Token via GITHUB_TOKEN env var only — never written to disk.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

REPOS = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
}

QUERIES = [  # (source-name, query tail) — union = the bug-fix set
    ("label", "label:bug"),
    ("fix", "fix in:title"),
    ("bug", "bug in:title"),
]

START = date(2023, 1, 1)  # both repos' first merges are within 2023
REQUEST_SPACING = 3.5     # s between search requests (limit: 30/min)
RETRY_WAITS = [30, 60, 90, 120, 180, 240]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    # U+2028/U+2029 would corrupt line-delimited readers that use splitlines()
    return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def load_existing(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["number"]] = r
    return out


def save(repo_key: str, merged: dict[int, dict], out_path: Path) -> None:
    recs = sorted(merged.values(), key=lambda r: r["number"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(_dumps(r) + "\n")
    tmp.replace(out_path)
    print(f"    [saved {len(recs)} -> {out_path.name}]", flush=True)


def quarter_windows(start: date, end: date):
    cur_y, cur_m = start.year, (start.month - 1) // 3 * 3 + 1
    while date(cur_y, cur_m, 1) <= end:
        nxt_y, nxt_m = (cur_y + 1, 1) if cur_m == 10 else (cur_y, cur_m + 3)
        q_end = date(nxt_y, nxt_m, 1) - timedelta(days=1)
        yield date(cur_y, cur_m, 1), min(q_end, end)
        cur_y, cur_m = nxt_y, nxt_m


def month_windows(q_start: date, q_end: date):
    cur = q_start
    while cur <= q_end:
        nxt_y, nxt_m = (cur.year + 1, 1) if cur.month == 12 else (cur.year, cur.month + 1)
        m_end = date(nxt_y, nxt_m, 1) - timedelta(days=1)
        yield cur, min(m_end, q_end)
        cur = m_end + timedelta(days=1)


def search_page(url: str, token: str) -> dict:
    """One search request with long-backoff retries; raises after exhaustion."""
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        if wait:
            print(f"    ! retry in {wait}s (attempt {attempt + 1})", flush=True)
            time.sleep(wait)
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("User-Agent", "bugfix-index-fetch")
        req.add_header("Accept", "application/vnd.github+json")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            print(f"    ! HTTP {e.code}: {body}", flush=True)
        except Exception as e:  # noqa: BLE001 - network hiccups
            print(f"    ! {e}", flush=True)
    raise RuntimeError(f"search failed permanently: {url[:120]}")


def search_slice(repo_full: str, tail: str, win: tuple[date, date], token: str):
    """Yield (total_count, items) per page for one merged-window slice."""
    lo, hi = win
    q = (f"repo:{repo_full} is:pr is:merged {tail} "
         f"merged:{lo.isoformat()}..{hi.isoformat()}")
    base = ("https://api.github.com/search/issues?q="
            + urllib.parse.quote(q, safe="")
            + "&per_page=100&sort=created&order=asc")
    page = 1
    while True:
        data = search_page(f"{base}&page={page}", token)
        items = data.get("items", [])
        total = data.get("total_count", 0)
        yield total, items
        # GitHub serves only the first 1000 search results; page 11+ answers 422.
        # Callers must split windows before reaching the cap (quarter->month).
        if len(items) < 100 or page * 100 >= min(total, 1000):
            break
        page += 1
        time.sleep(REQUEST_SPACING)


def slim(item: dict) -> dict:
    labels = [
        l.get("name", "") if isinstance(l, dict) else str(l)
        for l in item.get("labels", [])
    ]
    reactions = item.get("reactions") or {}
    return {
        "number": item["number"],
        "title": (item.get("title") or "").strip(),
        "state": item.get("state", ""),
        "author": (item.get("user") or {}).get("login", ""),
        "created_at": item.get("created_at", ""),
        "closed_at": item.get("closed_at", ""),  # for merged PRs == merged_at
        "updated_at": item.get("updated_at", ""),
        "labels": labels,
        "comments": item.get("comments", 0),
        "reactions_total": reactions.get("total_count", 0),
        "html_url": item.get("html_url", ""),
        "body": (item.get("body") or "")[:30000],
    }


def merge_items(merged: dict[int, dict], items: list[dict], qname: str) -> None:
    for it in items:
        rec = slim(it)
        prev = merged.get(rec["number"])
        rec["sources"] = list(prev.get("sources", [])) if prev else []
        if qname not in rec["sources"]:
            rec["sources"] = rec["sources"] + [qname]
        if prev:  # union labels across queries / runs
            rec["labels"] = sorted(set(prev.get("labels", [])) | set(rec["labels"]))
            rec["fetched_at"] = prev.get("fetched_at", now())
        else:
            rec["fetched_at"] = now()
        merged[rec["number"]] = rec


def fetch_window_items(repo_full: str, qtail: str, win: tuple[date, date],
                       token: str) -> tuple[int, list[dict]]:
    """All items of one window; if the window exceeds 1000, auto-split into months."""
    pages = list(search_slice(repo_full, qtail, win, token))
    total = pages[0][0] if pages else 0
    if total <= 1000:
        return total, [it for _, items in pages for it in items]
    print(f"    {win[0]}..{win[1]}: {total} > 1000 -> splitting into months", flush=True)
    items, raw = [], 0
    for mwin in month_windows(win[0], win[1]):
        mtotal, mitems = 0, []
        for t, page_items in search_slice(repo_full, qtail, mwin, token):
            mtotal = t
            mitems.extend(page_items)
        if mtotal > 1000:
            print(f"      !! month {mwin} still {mtotal} > 1000 — results truncated", flush=True)
        items.extend(mitems)
        raw += mtotal
        time.sleep(REQUEST_SPACING)
    return total, items


def fetch_query(repo_key: str, repo_full: str, qname: str, qtail: str,
                token: str, merged: dict[int, dict], out_path: Path) -> None:
    print(f"  query[{qname}]: {qtail}", flush=True)
    for q_start, q_end in quarter_windows(START, date(2026, 12, 31)):
        win = (q_start, q_end)
        n_before = len(merged)
        total, items = fetch_window_items(repo_full, qtail, win, token)
        merge_items(merged, items, qname)
        got = len(items)
        if total > 1000:
            print(f"    {q_start}..{q_end}: total {total} (split), +{len(merged) - n_before} new "
                  f"(raw {got})", flush=True)
        elif total and got != total:
            print(f"    !! {q_start}..{q_end}: got {got} != total {total}", flush=True)
        else:
            print(f"    {q_start}..{q_end}: +{len(merged) - n_before} new (raw {got})", flush=True)
        save(repo_key, merged, out_path)
        time.sleep(REQUEST_SPACING)


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("set GITHUB_TOKEN env var (export GITHUB_TOKEN=...)")

    for repo_key, full in REPOS.items():
        out_path = RAW / f"{repo_key}-bugfix-prs.jsonl"
        merged = load_existing(out_path)
        print(f"[{repo_key}] resuming with {len(merged)} records", flush=True)
        for qname, qtail in QUERIES:
            fetch_query(repo_key, full, qname, qtail, token, merged, out_path)
        save(repo_key, merged, out_path)
        recs = sorted(merged.values(), key=lambda r: r["number"])
        n = {s: sum(1 for r in recs if s in r["sources"]) for s, _ in QUERIES}
        print(f"[{repo_key}] total {len(recs)} merged fix PRs {n} -> {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
