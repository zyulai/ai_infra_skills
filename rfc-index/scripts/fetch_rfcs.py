#!/usr/bin/env python3
"""Fetch ALL RFC issues (label:rfc UNION 'RFC' in:title) from vllm-project/vllm
and sgl-project/sglang via the GitHub Search API.

Output: data/raw/<repo>-rfcs.jsonl  (one issue per line, body capped at 30k chars)

Resume-safe: re-running merges with the existing file by issue number
(labels / sources are unioned). Token via GITHUB_TOKEN env var only —
never written to disk.

Usage:
  export GITHUB_TOKEN=<token>
  python scripts/fetch_rfcs.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

REPOS = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
}

# Union of both queries = the RFC set. (issues search API has no OR grouping,
# so run one query per source and merge client-side.)
QUERIES = [
    ("label", "label:rfc"),
    ("title", "RFC in:title"),
]


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


def search_pages(query: str, token: str):
    """Yield (total_count, items) per page. 100/page, created asc = stable paging."""
    q = urllib.parse.quote(query, safe="")
    base = ("https://api.github.com/search/issues"
            f"?q={q}&per_page=100&sort=created&order=asc")
    page = 1
    while True:
        req = urllib.request.Request(f"{base}&page={page}")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("User-Agent", "rfc-index-fetch")
        req.add_header("Accept", "application/vnd.github+json")
        data = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:  # noqa: BLE001 - retry with backoff
                wait = 30 * (attempt + 1)
                print(f"    ! {e} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        if data is None:
            raise RuntimeError(f"search failed permanently: {base}&page={page}")
        items = data.get("items", [])
        total = data.get("total_count", 0)
        yield total, items
        if len(items) < 100 or page * 100 >= total:
            break
        page += 1
        time.sleep(2)


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
        "closed_at": item.get("closed_at", ""),
        "updated_at": item.get("updated_at", ""),
        "labels": labels,
        "comments": item.get("comments", 0),
        "reactions_total": reactions.get("total_count", 0),
        "html_url": item.get("html_url", ""),
        "body": (item.get("body") or "")[:30000],
    }


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("set GITHUB_TOKEN env var (export GITHUB_TOKEN=...)")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for repo_key, full in REPOS.items():
        out_path = RAW / f"{repo_key}-rfcs.jsonl"
        merged = load_existing(out_path)

        for qname, qtail in QUERIES:
            query = f"repo:{full} is:issue {qtail}"
            print(f"[{repo_key}] query[{qname}]: {query}", flush=True)
            got = 0
            total = None
            for total, items in search_pages(query, token):
                for it in items:
                    rec = slim(it)
                    prev = merged.get(rec["number"])
                    rec["sources"] = list(prev.get("sources", [])) if prev else []
                    if qname not in rec["sources"]:
                        rec["sources"] = rec["sources"] + [qname]
                    if prev:  # union labels across queries / runs
                        rec["labels"] = sorted(set(prev.get("labels", [])) | set(rec["labels"]))
                        rec["fetched_at"] = prev.get("fetched_at", now)
                    else:
                        rec["fetched_at"] = now
                    merged[rec["number"]] = rec
                got += len(items)
                print(f"    +{len(items)} (cum {got}/{total})", flush=True)
            if total is not None and got != total:
                print(f"    !! WARNING: got {got} != total_count {total}", flush=True)

        recs = sorted(merged.values(), key=lambda r: r["number"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(_dumps(r) + "\n")
        tmp.replace(out_path)
        n_label = sum(1 for r in recs if "label" in r["sources"])
        n_title = sum(1 for r in recs if "title" in r["sources"])
        print(f"[{repo_key}] total {len(recs)} RFC issues "
              f"(label:{n_label} title:{n_title}) -> {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
