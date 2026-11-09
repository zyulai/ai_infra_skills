#!/usr/bin/env python3
"""Select the next batch of major-tier PRs that lack evidence cards and emit
per-agent work lists for card authoring.

Usage:
  python scripts/next_batch.py --repo sglang [--batch 40] [--agents 4]

Multi-worker split (two sessions writing cards concurrently):
  python scripts/next_batch.py --repo vllm --make-split 2
      -> data/cards-workdir/vllm/split-0.json / split-1.json: disjoint
         fixed PR lists (rank parity of the score-ranked remaining pool).
  python scripts/next_batch.py --repo vllm --from-split 0 --worker main
      -> emit the next 51 un-done PRs of split-0 as agent lists under
         agent-batches-<worker>/ (no interference with --from-split 1).

Writes data/cards-workdir/<repo>/agent-batches[-<worker>]/batch-<id>-agent-<k>.md
where each line is: PR# | title | pack path | out path. Resume-safe: PRs that
already have a card (index/<repo>/cards/pr-<N>-*.md) or no pack are skipped;
ranked by the same importance score as the first batch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(r: dict) -> float:
    s = 0.0
    ev = r.get("speedup_evidence") or []
    if ev:
        s += 40 + max(e["num"] for e in ev)
    s += 3 * r.get("reactions_total", 0) + min(r.get("comments", 0), 40)
    labels = set(r.get("labels") or [])
    if "release-highlight" in labels:
        s += 30
    if "performance" in labels:
        s += 15
    if r.get("speedup_claims"):
        s += 8
    return s


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:40].strip("-")


def load_pool(repo: str) -> tuple[list[dict], Path]:
    major_path = ROOT / "data" / "major" / f"{repo}-major.jsonl"
    with major_path.open(encoding="utf-8") as f:  # iterate; splitlines() U+2028 trap
        major = [json.loads(l) for l in f if l.strip()]
    cards_dir = ROOT / "index" / repo / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.name.split("-")[1]) for p in cards_dir.glob("pr-*.md")}
    packs_dir = ROOT / "data" / "cards-workdir" / repo / "packs"
    candidates = [
        r
        for r in major
        if r["number"] not in done and (packs_dir / f"pr-{r['number']}.pack.md").exists()
    ]
    return sorted(candidates, key=score, reverse=True), packs_dir


def emit_agent_lists(repo: str, ranked: list[dict], packs_dir: Path, out_dir: Path, agents: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_id = 1 + max(
        (int(p.stem.split("-")[1]) for p in out_dir.glob("batch-*-agent-*.md")),
        default=0,
    )
    per_agent = -(-len(ranked) // agents)
    print(f"[{repo}] batch {batch_id}: {len(ranked)} cards, {agents} agents -> {out_dir.name}/")
    for k in range(agents):
        chunk = ranked[k * per_agent : (k + 1) * per_agent]
        if not chunk:
            continue
        lines = []
        for r in chunk:
            pack = (packs_dir / f"pr-{r['number']}.pack.md").relative_to(ROOT).as_posix()
            out = f"index/{repo}/cards/pr-{r['number']}-{slug(r['title'])}.md"
            lines.append(f"PR #{r['number']} | {r['title']} | pack={pack} | out={out}")
        f = out_dir / f"batch-{batch_id:03d}-agent-{k + 1}.md"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  agent {k + 1}: {len(chunk)} cards -> {f.relative_to(ROOT).as_posix()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, choices=["sglang", "vllm"])
    ap.add_argument("--batch", type=int, default=51)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--make-split", type=int, metavar="N", help="partition the remaining pool into N disjoint fixed lists and exit")
    ap.add_argument("--from-split", type=int, metavar="K", help="draw the next batch from fixed split list K instead of re-ranking")
    ap.add_argument("--worker", default="", help="worker tag; batch lists go to agent-batches-<worker>/")
    args = ap.parse_args()

    repo = args.repo
    workdir = ROOT / "data" / "cards-workdir" / repo

    if args.make_split:
        ranked, _ = load_pool(repo)
        n = args.make_split
        for k in range(n):
            part = ranked[k::n]
            path = workdir / f"split-{k}.json"
            path.write_text(json.dumps([r["number"] for r in part]), encoding="utf-8")
            print(f"[{repo}] split-{k}: {len(part)} PRs -> {path.relative_to(ROOT).as_posix()}")
        return

    if args.from_split is not None:
        split_path = workdir / f"split-{args.from_split}.json"
        numbers = json.loads(split_path.read_text(encoding="utf-8"))
        by_number = {}
        major_path = ROOT / "data" / "major" / f"{repo}-major.jsonl"
        with major_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_number[r["number"]] = r
        cards_dir = ROOT / "index" / repo / "cards"
        done = {int(p.name.split("-")[1]) for p in cards_dir.glob("pr-*.md")}
        packs_dir = workdir / "packs"
        ranked = [
            by_number[n]
            for n in numbers
            if n not in done and (packs_dir / f"pr-{n}.pack.md").exists()
        ]
        total_in_split = len(numbers)
        done_in_split = sum(1 for n in numbers if n in done)
        ranked = ranked[: args.batch]
        print(
            f"[{repo}] split-{args.from_split}: total {total_in_split}, done {done_in_split}, "
            f"taking {len(ranked)} now, left after this {total_in_split - done_in_split - len(ranked)}"
        )
        out_dir = workdir / ("agent-batches" if not args.worker else f"agent-batches-{args.worker}")
        emit_agent_lists(repo, ranked, packs_dir, out_dir, args.agents)
        return

    ranked, packs_dir = load_pool(repo)
    remaining = len(ranked)
    ranked = ranked[: args.batch]
    out_dir = ROOT / "data" / "cards-workdir" / repo / "agent-batches"
    print(f"[{repo}] {remaining} cards in pool, taking {len(ranked)}, {remaining - len(ranked)} left after this")
    emit_agent_lists(repo, ranked, packs_dir, out_dir, args.agents)


if __name__ == "__main__":
    main()
