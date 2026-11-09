#!/usr/bin/env python3
"""Generate material packs + agent batch lists for major RFC summary cards.

For every 🔥 RFC: write a pack (metadata + body ≤20k chars) to
  data/cards-workdir/<repo>/packs/rfc-<N>.pack.md
Pick the next `--batch-size` unfinished RFCs (score-sorted), split
round-robin across `--agents` agents, and emit
  data/cards-workdir/<repo>/agent-batches/batch-XXX-agent-K.md
with one line per card: `RFC #N | title | pack=... | out=...`

Resume-safe: RFCs that already have a card (index/<repo>/rfc-cards/rfc-<N>-*.md)
are skipped; packs are only written when missing.

Usage:
  python scripts/make_batches.py --repo both --batch-size 45 --agents 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAJOR = ROOT / "data" / "major"
WORKDIR = ROOT / "data" / "cards-workdir"
CARDS = ROOT / "index"
SLUG_RE = re.compile(r"[^a-z0-9]+")
PACK_BODY_CAP = 20000

REPOS = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
        return [json.loads(l) for l in f if l.strip()]


def slugify(title: str) -> str:
    return SLUG_RE.sub("-", title.lower()).strip("-")[:40].strip("-")


def out_path(repo: str, rec: dict) -> Path:
    return CARDS / repo / "rfc-cards" / f"rfc-{rec['number']}-{slugify(rec['title'])}.md"


def write_pack(repo: str, rec: dict, path: Path) -> None:
    body = (rec.get("body") or "").strip()
    if len(body) > PACK_BODY_CAP:
        body = body[:PACK_BODY_CAP] + "\n\n<!-- ...body truncated at 20k chars... -->"
    labels = ", ".join(rec.get("labels", [])) or "(none)"
    closed = f"({rec['closed_at'][:10]})" if rec.get("closed_at") else ""
    md = f"""<!-- pack for RFC #{rec['number']} ({repo}) — source material for the summary card -->
# RFC #{rec['number']}: {rec['title']}

| 字段 | 值 |
| --- | --- |
| 仓库 | {REPOS[repo]} |
| 编号 | [#{rec['number']}]({rec['html_url']}) |
| 作者 | {rec.get('author', '')} |
| 创建日期 | {rec['created_at'][:10]} |
| 状态 | {'🟢 open' if rec.get('state') == 'open' else '⚪ closed'} {closed} |
| labels | {labels} |
| 热度 | 👍 {rec.get('reactions_total', 0)} · 💬 {rec.get('comments', 0)} |
| 分类 | {rec.get('category', '')} |

## 正文(PRD / 提案全文)

{body if body else '(正文为空 — 卡片如实注明,只写元信息,不编造)'}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(path)


def score(rec: dict) -> int:
    return (rec.get("reactions_total", 0) * 2 + rec.get("comments", 0)
            + 8 * len(rec.get("speedup_evidence", []))
            + (5 if rec.get("perf") else 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=45)
    ap.add_argument("--agents", type=int, default=3)
    args = ap.parse_args()
    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]

    for repo in repos:
        recs = read_jsonl(MAJOR / f"{repo}-rfcs-major.jsonl")
        if not recs:
            print(f"[{repo}] no major RFCs")
            continue
        cards_dir = CARDS / repo / "rfc-cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        done = {int(m.group(1)) for p in cards_dir.glob("rfc-*.md")
                if (m := re.match(r"rfc-(\d+)-", p.name))}
        pending = sorted((r for r in recs if r["number"] not in done),
                         key=lambda r: (-score(r), r["number"]))

        packs_dir = WORKDIR / repo / "packs"
        n_packs = 0
        for r in recs:  # packs for all majors (idempotent)
            p = packs_dir / f"rfc-{r['number']}.pack.md"
            if not p.exists():
                write_pack(repo, r, p)
                n_packs += 1

        batch = pending[: args.batch_size]
        if not batch:
            print(f"[{repo}] all {len(recs)} majors have cards — nothing to do")
            continue
        bdir = WORKDIR / repo / "agent-batches"
        bdir.mkdir(parents=True, exist_ok=True)
        existing = [int(m.group(1)) for p in bdir.glob("batch-*-agent-*.md")
                    if (m := re.match(r"batch-(\d+)-", p.name))]
        bno = (max(existing) + 1) if existing else 1

        agent_lists: dict[int, list[str]] = {k: [] for k in range(1, args.agents + 1)}
        for i, r in enumerate(batch):
            op = out_path(repo, r)
            pk = packs_dir / f"rfc-{r['number']}.pack.md"
            agent_lists[i % args.agents + 1].append(
                f"RFC #{r['number']} | {r['title']} | pack={pk.as_posix()} | out={op.as_posix()}"
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
