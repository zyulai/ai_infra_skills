#!/usr/bin/env python3
"""Search the bug-fix index. Stdlib only, offline.

Usage:
  python scripts/query.py --list                                  # 分类/类型统计
  python scripts/query.py "cuda graph"                            # 关键词检索(标题×3, 正文×1, AND)
  python scripts/query.py --repo vllm --tier major --type crash   # 组合过滤
  python scripts/query.py --type correctness --since 2026-01-01
  python scripts/query.py --card 12345 --repo sglang              # 打印一张修复卡
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIED_DIR = ROOT / "data" / "classified"
CARDS = ROOT / "index"

BUG_TYPE_ZH = {
    "correctness": "正确性", "crash": "崩溃", "hang": "挂死",
    "resource": "资源泄漏/OOM", "regression": "回归", "startup": "启动/加载",
    "general": "一般",
}


def load(repos: list[str]) -> list[dict]:
    recs = []
    for repo in repos:
        src = CLASSIFIED_DIR / f"{repo}-bugfix-classified.jsonl"
        if not src.exists():
            continue
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
            recs.extend(json.loads(l) for l in f if l.strip())
    return recs


def find_card(repo: str, number: int):
    d = CARDS / repo / "fix-cards"
    if d.is_dir():
        for p in d.glob(f"pr-{number}-*.md"):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("keyword", nargs="?", default="", help="关键词(空格分词, AND)")
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--category", default=None)
    ap.add_argument("--type", dest="bug_type", default=None,
                    help="correctness/crash/hang/resource/regression/startup/general")
    ap.add_argument("--tier", choices=["major", "normal"], default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD(按合并日期)")
    ap.add_argument("--until", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="只打印分类/类型统计")
    ap.add_argument("--card", type=int, default=None, help="打印某个 PR 的修复卡全文")
    args = ap.parse_args()

    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    recs = load(repos)

    if args.card is not None:
        found = [r for r in recs if r["number"] == args.card]
        if not found:
            raise SystemExit(f"PR #{args.card} not in index")
        p = None
        for repo in repos:
            p = find_card(repo, args.card)
            if p:
                break
        if p:
            print(p.read_text(encoding="utf-8"))
        else:
            print(f"(PR #{args.card} 无修复卡) {found[0]['html_url']}")
        return

    if args.list:
        for repo in repos:
            src = CLASSIFIED_DIR / f"{repo}-bugfix-classified.jsonl"
            if not src.exists():
                continue
            with src.open(encoding="utf-8") as f:
                sub = [json.loads(l) for l in f if l.strip()]
            cats = Counter(r["category"] for r in sub)
            types = Counter(r["bug_type"] for r in sub)
            majors = sum(1 for r in sub if r["tier"] == "major")
            print(f"\n[{repo}] total {len(sub)} | 🔥 major {majors}")
            print("  分类:", ", ".join(f"{k}={v}" for k, v in cats.most_common()))
            print("  类型:", ", ".join(f"{BUG_TYPE_ZH.get(k, k)}={v}" for k, v in types.most_common()))
        return

    # filters
    terms = [t.lower() for t in args.keyword.split() if t]
    hit = []
    for r in recs:
        d = (r.get("closed_at") or "")[:10]
        if args.since and d and d < args.since:
            continue
        if args.until and d and d > args.until:
            continue
        if args.category and r["category"] != args.category:
            continue
        if args.bug_type and r["bug_type"] != args.bug_type:
            continue
        if args.tier and r["tier"] != args.tier:
            continue
        if args.author and r.get("author", "") != args.author:
            continue
        if terms:
            hay_title = r.get("title", "").lower()
            hay_body = (r.get("body") or "").lower()
            ok = all(t in hay_title or t in hay_body for t in terms)
            if not ok:
                continue
            score = sum(3 * (t in hay_title) + (t in hay_body) for t in terms)
        else:
            score = 0
        hit.append((score, r))

    hit.sort(key=lambda t: (-t[0], t[1].get("closed_at") or ""), )
    hit = hit[: args.limit]

    print(f"{'PR':>7} | {'合并':<10} | {'类型':<12} | 档位 | 仓库 | 标题 | 修复卡")
    print("-" * 100)
    for score, r in hit:
        repo = "vllm" if "vllm-project" in r.get("html_url", "") else "sglang"
        card = find_card(repo, r["number"])
        card_s = f"{card.relative_to(CARDS).as_posix()}" if card else ""
        tier = "🔥" if r["tier"] == "major" else ""
        print(f"[{r['number']:>6}] | {r.get('closed_at', '')[:10]:<10} "
              f"| {BUG_TYPE_ZH.get(r['bug_type'], r['bug_type']):<12} | {tier:>2} | {repo} "
              f"| {r['title'][:60]} | {card_s}")
    print(f"\n({len(hit)} shown of {len(recs)} indexed; --limit 调整)")


if __name__ == "__main__":
    main()
