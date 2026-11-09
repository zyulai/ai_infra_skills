#!/usr/bin/env python3
"""Generate material packs (diff + body) + agent batch lists for major
bug-fix PRs.

For every 🔥 fix PR:
  1. fetch the PR diff (patch-diff.githubusercontent.com — no API quota;
     falls back to the API with token), trim to 40k chars at file granularity
     (source files kept before test/doc files);
  2. write pack  data/cards-workdir/<repo>/packs/pr-<N>.pack.md
     = metadata + body (≤20k) + trimmed diff;
  3. score-sorted selection of the next `--batch-size` unfinished PRs,
     round-robin across `--agents` agents ->
     data/cards-workdir/<repo>/agent-batches/batch-XXX-agent-K.md
     with one line per card: `PR #N | title | pack=... | out=...`

Resume-safe: PRs with an existing card (index/<repo>/fix-cards/pr-<N>-*.md)
are skipped; packs only written when missing. Use --refresh to rebuild packs.

Usage:
  python scripts/make_packs.py --repo both --batch-size 45 --agents 3
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
MAJOR = ROOT / "data" / "major"
WORKDIR = ROOT / "data" / "cards-workdir"
CARDS = ROOT / "index"
SLUG_RE = re.compile(r"[^a-z0-9]+")
DIFF_CAP = 40000
BODY_CAP = 20000

REPOS = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
}

SEVERITY_SCORE = {
    "correctness": 30, "crash": 20, "hang": 20,
    "resource": 15, "regression": 20, "startup": 5, "general": 0,
}
STRONG_COMPONENT = re.compile(
    r"kernel|attention|flashinfer|\bmoe\b|deepep|quant|cuda graph|schedul|kv cache"
    r"|speculative|\beagle\b|\bmtp\b|parallel|allreduce|nccl|tokenizer|detokeniz"
    r"|prefix cach|radix|logits|sampling|paged", re.I)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def slugify(title: str) -> str:
    return SLUG_RE.sub("-", title.lower()).strip("-")[:40].strip("-")


def out_path(repo: str, rec: dict) -> Path:
    return CARDS / repo / "fix-cards" / f"pr-{rec['number']}-{slugify(rec['title'])}.md"


def score(rec: dict) -> int:
    return (SEVERITY_SCORE.get(rec.get("bug_type"), 0)
            + rec.get("reactions_total", 0) * 2 + rec.get("comments", 0)
            + (5 if STRONG_COMPONENT.search(rec.get("title", "")) else 0))


def _http_get(url: str, headers: dict, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - retry with backoff
            print(f"    ! GET {url.rsplit('/', 1)[-1]}: {e} (attempt {attempt + 1})", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def fetch_diff(full: str, number: int, token: str) -> str:
    """PR diff without burning API quota; API fallback with token."""
    data = _http_get(
        f"https://patch-diff.githubusercontent.com/raw/{full}/pull/{number}.diff",
        {"User-Agent": "bugfix-index-packs"})
    if data is None and token:
        data = _http_get(
            f"https://api.github.com/repos/{full}/pulls/{number}",
            {"User-Agent": "bugfix-index-packs",
             "Authorization": f"Bearer {token}",
             "Accept": "application/vnd.github.diff"})
    return data.decode("utf-8", errors="replace") if data else ""


def trim_diff(diff: str, cap: int = DIFF_CAP) -> str:
    """File-granularity trim; source files kept before test/doc files."""
    if len(diff) <= cap:
        return diff
    chunks = [c for c in re.split(r"(?m)^(?=diff --git )", diff) if c.strip()]
    src, test = [], []
    for c in chunks:
        m = re.match(r"diff --git a/(\S+)", c)
        path = m.group(1) if m else ""
        tgt = test if re.search(r"(^|/)(tests?|benchmarks?|docs)/|\.md$|\.rst$|\.json$", path) else src
        tgt.append(c)
    out, used, truncated = [], 0, False
    for c in src + test:
        if used + len(c) > cap:
            truncated = True
            break
        out.append(c)
        used += len(c)
    text = "".join(out) if out else diff[:cap]
    if truncated:
        text += f"\n<!-- ...diff truncated at {cap} chars (file granularity)... -->"
    return text


def write_pack(repo: str, rec: dict, path: Path, token: str) -> None:
    full = REPOS[repo]
    body = (rec.get("body") or "").strip()
    if len(body) > BODY_CAP:
        body = body[:BODY_CAP] + "\n\n<!-- ...body truncated at 20k chars... -->"
    diff = trim_diff(fetch_diff(full, rec["number"], token))
    diff_note = "" if diff.strip() else "(diff 获取失败 — 卡片如实注明)"
    labels = ", ".join(rec.get("labels", [])) or "(none)"
    md = f"""<!-- pack for PR #{rec['number']} ({repo}) — source material for the fix card -->
# PR #{rec['number']}: {rec['title']}

| 字段 | 值 |
| --- | --- |
| 仓库 | {full} |
| 编号 | [#{rec['number']}]({rec['html_url']}) |
| 作者 | {rec.get('author', '')} |
| 合并日期 | {(rec.get('closed_at') or '')[:10]} |
| labels | {labels} |
| 热度 | 👍 {rec.get('reactions_total', 0)} · 💬 {rec.get('comments', 0)} |
| 分类 | {rec.get('category', '')} |
| bug 类型 | {rec.get('bug_type', '')} |

## 正文(PR 描述)

{body if body else '(正文为空 — 卡片如实注明)'}

## Diff

{diff_note}

```diff
{diff}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=45)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="re-fetch missing diffs in existing packs")
    args = ap.parse_args()
    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    for repo in repos:
        recs = read_jsonl(MAJOR / f"{repo}-bugfix-major.jsonl")
        if not recs:
            print(f"[{repo}] no major fix PRs")
            continue
        cards_dir = CARDS / repo / "fix-cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        done = {int(m.group(1)) for p in cards_dir.glob("pr-*.md")
                if (m := re.match(r"pr-(\d+)-", p.name))}
        pending = sorted((r for r in recs if r["number"] not in done),
                         key=lambda r: (-score(r), r["number"]))

        packs_dir = WORKDIR / repo / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)

        batch = pending[: args.batch_size]
        if not batch:
            print(f"[{repo}] all {len(recs)} majors have cards — nothing to do")
            continue

        n_packs = 0
        for r in batch:  # packs for the selected batch only (idempotent)
            p = packs_dir / f"pr-{r['number']}.pack.md"
            if p.exists():
                if args.refresh and "(diff 获取失败" in p.read_text(encoding="utf-8"):
                    write_pack(repo, r, p, token)
                    n_packs += 1
                continue
            write_pack(repo, r, p, token)
            n_packs += 1
        bdir = WORKDIR / repo / "agent-batches"
        bdir.mkdir(parents=True, exist_ok=True)
        existing = [int(m.group(1)) for p in bdir.glob("batch-*-agent-*.md")
                    if (m := re.match(r"batch-(\d+)-", p.name))]
        bno = (max(existing) + 1) if existing else 1

        agent_lists: dict[int, list[str]] = {k: [] for k in range(1, args.agents + 1)}
        for i, r in enumerate(batch):
            op = out_path(repo, r)
            pk = packs_dir / f"pr-{r['number']}.pack.md"
            agent_lists[i % args.agents + 1].append(
                f"PR #{r['number']} | {r['title']} | pack={pk.as_posix()} | out={op.as_posix()}"
            )
        for k, lines in agent_lists.items():
            if not lines:
                continue
            f = bdir / f"batch-{bno:03d}-agent-{k}.md"
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[{repo}] batch-{bno:03d}-agent-{k}: {len(lines)} cards -> {f.name}")
        print(f"[{repo}] wrote {n_packs} new packs | batch {bno:03d} covers {len(batch)} "
              f"| remaining after this batch: {max(0, len(pending) - len(batch))} "
              f"(done {len(done)}/{len(recs)})")


if __name__ == "__main__":
    main()
