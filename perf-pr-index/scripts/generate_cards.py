#!/usr/bin/env python3
"""Prepare material packs for evidence-card authoring (major tier).

For each selected PR this fetches the PR diff (public patch-diff URL, no API
quota) and writes one self-contained pack per PR:

    data/cards-workdir/{repo}/packs/pr-<N>.pack.md

plus batch manifests:

    data/cards-workdir/{repo}/batches/batch-<category>-<k>.json

The pack contains everything a writer (human or agent) needs: metadata,
PR body, and the diff. Card authoring then happens outside this script.
Resume-safe: existing packs are skipped.

Token: GITHUB_TOKEN via env var (used only for the diff fetch fallback via
api.github.com; primary source needs no token).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAJOR_DIR = ROOT / "data" / "major"
WORKDIR = ROOT / "data" / "cards-workdir"

BATCH_SIZE = 40
DIFF_LIMIT = 40000  # chars of diff kept per pack
BODY_LIMIT = 25000

# file paths whose diff hunks are kept verbatim first (rest summarized by count)
SOURCE_HINT = re.compile(r"(sgl|sglang|vllm|csrc|kernels|layers|models|attention|"
                         r"schedul|quant|specul|moe|cache|worker|executor|engine)", re.I)


def fetch_diff(repo_full: str, number: int, token: str) -> str:
    """PR diff via the public patch URL; falls back to the API endpoint."""
    urls = [
        f"https://patch-diff.githubusercontent.com/raw/{repo_full}/pull/{number}.diff",
        f"https://api.github.com/repos/{repo_full}/pulls/{number}",
    ]
    for i, url in enumerate(urls):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "perf-pr-index-cards")
            if i == 1:
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                req.add_header("Accept", "application/vnd.github.v3.diff")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - report and try next source
            print(f"    diff fetch failed for #{number} via {url.split('/')[2]}: {e}")
            time.sleep(2)
    return ""


def trim_diff(diff: str) -> tuple[str, int]:
    """Keep source-file hunks up to DIFF_LIMIT; count dropped files."""
    if len(diff) <= DIFF_LIMIT:
        return diff, 0
    files = diff.split("diff --git ")
    kept: list[str] = []
    dropped = 0
    budget = DIFF_LIMIT
    header = files[0]
    files = files[1:]
    # source-ish files first
    files.sort(key=lambda f: 0 if SOURCE_HINT.search(f.split("\n", 1)[0]) else 1)
    for f in files:
        chunk = "diff --git " + f
        if len(chunk) <= budget:
            kept.append(chunk)
            budget -= len(chunk)
        else:
            if budget > 500:
                kept.append(chunk[:budget] + "\n... [truncated]\n")
                budget = 0
            dropped += 1
    return header + "".join(kept), dropped


def build_pack(rec: dict, token: str) -> Path:
    key = rec["repo"]
    packs_dir = WORKDIR / key / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    out = packs_dir / f"pr-{rec['number']}.pack.md"
    if out.exists():
        return out
    diff = fetch_diff(rec["repo_full"], rec["number"], token)
    diff, dropped = trim_diff(diff)
    body = (rec.get("body") or "")[:BODY_LIMIT]
    claims = "\n".join(f"- {c}" for c in rec.get("speedup_claims", [])) or "- (none detected by regex)"
    parts = [
        f"# MATERIAL PACK — {rec['repo_full']} PR #{rec['number']}",
        "",
        f"- title: {rec['title']}",
        f"- author: {rec['author']}",
        f"- merged/closed: {(rec.get('merged_at') or rec.get('closed_at') or '')[:10]}",
        f"- labels: {', '.join(rec.get('labels') or []) or '(none)'}",
        f"- category: {rec.get('category')} | tier: {rec.get('tier')}",
        f"- diffstat: +{rec.get('additions')}/-{rec.get('deletions')} across {rec.get('changed_files')} files",
        f"- url: {rec['html_url']}",
        "",
        "## perf claims detected in body (regex, unverified)",
        claims,
        "",
        "## PR body",
        "```markdown",
        body,
        "```",
        "",
        "## diff",
        "```diff",
        diff or "(diff unavailable)",
        "```",
    ]
    if dropped:
        parts.append(f"\n> {dropped} file(s) dropped from diff for size (see PR for full diff).")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def emit_batches(repo: str, recs: list[dict]) -> list[Path]:
    batches_dir = WORKDIR / repo / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[dict]] = {}
    for r in recs:
        by_cat.setdefault(r.get("category") or "Other", []).append(r)
    paths = []
    for cat, items in sorted(by_cat.items()):
        items.sort(key=lambda r: r["number"])
        slug = re.sub(r"[^a-z0-9]+", "-", cat.lower()).strip("-")
        for k in range(0, len(items), BATCH_SIZE):
            chunk = items[k : k + BATCH_SIZE]
            path = batches_dir / f"batch-{slug}-{k // BATCH_SIZE:02d}.json"
            path.write_text(
                json.dumps(
                    {
                        "repo": repo,
                        "category": cat,
                        "count": len(chunk),
                        "prs": [
                            {
                                "number": r["number"],
                                "title": r["title"],
                                "pack": f"data/cards-workdir/{repo}/packs/pr-{r['number']}.pack.md",
                                "card_out": f"index/{repo}/cards/pr-{r['number']}-<slug>.md",
                            }
                            for r in chunk
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--category", help="only pack PRs of this category")
    ap.add_argument("--limit", type=int, help="max PRs to pack this run")
    ap.add_argument("--refresh", action="store_true", help="rebuild existing packs")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    for repo in repos:
        src = MAJOR_DIR / f"{repo}-major.jsonl"
        if not src.exists():
            print(f"[{repo}] no major data ({src}); run classify.py first")
            continue
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines() (U+2028 traps)
            recs = [json.loads(l) for l in f if l.strip()]
        if args.category:
            recs = [r for r in recs if r.get("category") == args.category]
        if args.limit:
            recs = recs[: args.limit]
        print(f"[{repo}] packing {len(recs)} major PRs")
        done = 0
        for r in recs:
            if args.refresh:
                (WORKDIR / repo / "packs" / f"pr-{r['number']}.pack.md").unlink(missing_ok=True)
            build_pack(r, token)
            done += 1
            if done % 25 == 0:
                print(f"  packed {done}/{len(recs)}", flush=True)
            time.sleep(0.3)  # be gentle with patch-diff
        batches = emit_batches(repo, recs)
        print(f"[{repo}] packs: {done} | batches: {len(batches)}")


if __name__ == "__main__":
    main()
