#!/usr/bin/env python3
"""Query the perf PR index knowledge base.

Examples:
  python scripts/query.py --list
  python scripts/query.py "cuda graph"
  python scripts/query.py --framework vllm "fused moe"
  python scripts/query.py --framework sglang --tier major --category "Attention / Kernel"
  python scripts/query.py --min-speedup 30 --limit 50
  python scripts/query.py --author yhyang --since 2025-01-01
  python scripts/query.py --card 11061 --repo sglang     # show the evidence card

Stdlib only; reads data/classified/*.jsonl and index/*/cards/*.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIED_DIR = ROOT / "data" / "classified"
INDEX_DIR = ROOT / "index"

CATEGORIES = [
    "Speculative Decoding",
    "MoE",
    "Quantization",
    "CUDA Graph / Compiler",
    "Overlap / Scheduler",
    "Attention / Kernel",
    "KV Cache / Memory",
    "Parallelism / Communication",
    "Sampler / Detokenization",
    "Frontend / Serving",
    "Model-specific Fast Path",
    "CI / Test-Perf",
    "Other",
]


def load_records(repos: list[str]) -> list[dict]:
    recs: list[dict] = []
    for repo in repos:
        src = CLASSIFIED_DIR / f"{repo}-classified.jsonl"
        if not src.exists():
            continue
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines() (U+2028 traps)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"warn: skipping malformed line in {src.name}", file=sys.stderr)
    return recs


def load_card_text(repo: str, number: int) -> str:
    cards_dir = INDEX_DIR / repo / "cards"
    if not cards_dir.is_dir():
        return ""
    for p in cards_dir.glob(f"pr-{number}-*.md"):
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def tokenize(query: str) -> list[str]:
    return list(dict.fromkeys(t for t in re.split(r"\s+", query.lower()) if t))


def score_record(rec: dict, terms: list[str], use_cards: bool) -> int:
    """All terms must match somewhere; weight: title x3, claim x2, body/card x1."""
    title = rec.get("title", "").lower()
    body = (rec.get("body") or "").lower()
    claims = " ".join(rec.get("speedup_claims", [])).lower()
    card = load_card_text(rec["repo"], rec["number"]).lower() if use_cards else ""
    total = 0
    for t in terms:
        s = 0
        if t in title:
            s += 3
        if t in claims:
            s += 2
        if t in body:
            s += 1
        if card and t in card:
            s += 1
        if s == 0:
            return 0
        total += s
    return total


def display_date(rec: dict) -> str:
    return ((rec.get("merged_at") or rec.get("closed_at")) or "")[:10]


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no results")
        return
    print(
        f"{'repo':<7} {'PR':>7}  {'tier':<6} {'category':<28} {'date':<10} title"
    )
    print("-" * 110)
    for r in rows:
        tier = "major" if r["tier"] == "major" else "-"
        title = r["title"].replace("\n", " ")
        if len(title) > 68:
            title = title[:67] + "…"
        claim = ""
        if r.get("speedup_numbers"):
            claim = " | " + r["speedup_claims"][0][:40] if r.get("speedup_claims") else ""
        print(
            f"{r['repo']:<7} #{r['number']:<6} {tier:<6} {r['category']:<28} "
            f"{display_date(r):<10} {title}{claim}"
        )
    print(f"\n{len(rows)} result(s). Links: {rows[0]['html_url'].rsplit('/', 1)[0]}/<PR#>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="keywords (AND semantics)")
    ap.add_argument("--repo", "--framework", dest="repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--category", help=f"one of: {'; '.join(CATEGORIES)}")
    ap.add_argument("--tier", choices=["major", "normal"])
    ap.add_argument("--author")
    ap.add_argument("--since", help="YYYY-MM-DD, on merged/closed date")
    ap.add_argument("--until", help="YYYY-MM-DD")
    ap.add_argument("--min-speedup", type=float, help="min claimed number (%% or x)")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--list", action="store_true", help="show category counts and exit")
    ap.add_argument("--card", type=int, metavar="PR#", help="print the evidence card for a PR")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-cards", action="store_true", help="skip evidence cards in search")
    args = ap.parse_args()

    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    recs = load_records(repos)
    if not recs:
        sys.exit("no classified data found; run fetch_prs.py + classify.py first")

    if args.card:
        repo = args.repo if args.repo != "both" else None
        matches = [r for r in recs if r["number"] == args.card and (repo is None or r["repo"] == repo)]
        if not matches:
            sys.exit(f"PR #{args.card} not found in index")
        rec = matches[0]
        text = load_card_text(rec["repo"], rec["number"])
        if text:
            print(text)
        else:
            print(f"no evidence card yet for #{rec['number']} [{rec['repo']}] {rec['title']}")
            print(rec["html_url"])
        return

    if args.list:
        from collections import Counter
        for repo in repos:
            sub = [r for r in recs if r["repo"] == repo]
            print(f"[{repo}] total {len(sub)}, major {sum(1 for r in sub if r['tier'] == 'major')}")
            for cat, n in Counter(r["category"] for r in sub).most_common():
                print(f"    {cat:<32} {n}")
        return

    out = recs
    if args.category:
        out = [r for r in out if r["category"].lower() == args.category.lower()]
    if args.tier:
        out = [r for r in out if r["tier"] == args.tier]
    if args.author:
        out = [r for r in out if r["author"].lower() == args.author.lower()]
    if args.since:
        out = [r for r in out if display_date(r) >= args.since]
    if args.until:
        out = [r for r in out if display_date(r) <= args.until]
    if args.min_speedup is not None:
        out = [r for r in out if any(n >= args.min_speedup for n in r.get("speedup_numbers", []))]

    terms = tokenize(" ".join(args.query))
    if terms:
        scored = [(score_record(r, terms, not args.no_cards), r) for r in out]
        out = [r for s, r in scored if s > 0]
        score_of = {id(r): s for s, r in scored if s > 0}
        out.sort(key=lambda r: (-score_of[id(r)], display_date(r), -r["number"]))
    else:
        out.sort(key=lambda r: (display_date(r), r["number"]), reverse=True)

    out = out[: args.limit]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_table(out)


if __name__ == "__main__":
    main()
