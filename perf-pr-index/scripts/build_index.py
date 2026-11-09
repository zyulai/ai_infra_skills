#!/usr/bin/env python3
"""Build the human-readable markdown index from classified PR data.

Input : data/classified/{repo}-classified.jsonl
Output: index/{repo}/PERF-PRS.md   (per-category index tables)
        index/STATS.md             (cross-repo statistics)

Stdlib only, offline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIED_DIR = ROOT / "data" / "classified"
INDEX_DIR = ROOT / "index"
SNAPSHOT = ROOT / "data" / "snapshot.json"


def snapshot_lines() -> list[str]:
    """Render the data snapshot (upstream HEAD commit ids, fetch window) if present."""
    if not SNAPSHOT.exists():
        return []
    try:
        snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    lines = ["", "## 数据快照(下次更新的基线)", ""]
    lines.append(f"- 快照记录时间: {snap.get('generated_at', '?')}")
    for repo, e in snap.get("repos", {}).items():
        sha = e.get("head_sha") or "?"
        head = e.get("head_date", "")[:10]
        url = f"https://github.com/{e['repo_full']}/commit/{sha}" if sha != "?" else "?"
        lo, hi = e.get("fetch_window_utc", ["?", "?"])
        lines.append(
            f"- **{repo}** @{sha[:12]} (upstream HEAD {head}) — "
            f"commit: {url} | API 抓取窗口: {lo} ~ {hi} | "
            f"raw {e.get('raw_prs')} → 重点档 {e.get('major_prs')} → 证据卡 {e.get('evidence_cards')}"
        )
    return lines

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

REPO_URLS = {
    "vllm": "https://github.com/vllm-project/vllm",
    "sglang": "https://github.com/sgl-project/sglang",
}

# 分类中文显示名(jsonl 里的 category 键保持英文,仅渲染层双语)
CATEGORY_ZH = {
    "Speculative Decoding": "投机解码",
    "MoE": "混合专家",
    "Quantization": "量化",
    "CUDA Graph / Compiler": "CUDA Graph 与编译",
    "Overlap / Scheduler": "重叠与调度",
    "Attention / Kernel": "注意力与算子",
    "KV Cache / Memory": "KV Cache 与显存",
    "Parallelism / Communication": "并行与通信",
    "Sampler / Detokenization": "采样与反分词",
    "Frontend / Serving": "前端与服务",
    "Model-specific Fast Path": "模型专属快路径",
    "CI / Test-Perf": "CI 与测试性能",
    "Other": "其他",
}


def cat_display(cat: str) -> str:
    zh = CATEGORY_ZH.get(cat)
    return f"{cat}({zh})" if zh else cat


def gh_anchor(text: str) -> str:
    """GitHub heading-anchor algorithm: lowercase, drop punctuation (CJK stays),
    spaces -> hyphens."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)  # \w covers ASCII word chars and CJK letters
    return re.sub(r"\s", "-", t)


SLUG_RE = re.compile(r"[^a-z0-9]+")


def card_path(repo: str, number: int, title: str) -> Path:
    slug = SLUG_RE.sub("-", title.lower()).strip("-")[:40].strip("-")
    return INDEX_DIR / repo / "cards" / f"pr-{number}-{slug}.md"


def find_card(repo: str, number: int) -> Path | None:
    cards_dir = INDEX_DIR / repo / "cards"
    if not cards_dir.is_dir():
        return None
    prefix = f"pr-{number}-"
    for p in cards_dir.glob(f"pr-{number}-*.md"):
        return p
    return None


def md_cell(text: str, limit: int = 90) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def pr_link(repo: str, rec: dict) -> str:
    return f"[#{rec['number']}]({rec['html_url']})"


def date_str(rec: dict) -> str:
    d = rec.get("merged_at") or rec.get("closed_at") or ""
    return (d or "")[:10]


def repo_table(repo: str, recs: list[dict]) -> str:
    lines = [
        "| PR | 标题 | 作者 | 合并日期 | 档位 | 加速声明 | 证据卡 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in recs:
        tier = "🔥 重点" if rec["tier"] == "major" else "一般"
        card = find_card(repo, rec["number"])
        card_cell = (
            f"[卡片]({card.relative_to(INDEX_DIR / repo).as_posix()})" if card else ""
        )
        claim = md_cell(rec["speedup_claims"][0], 60) if rec.get("speedup_claims") else ""
        lines.append(
            f"| {pr_link(repo, rec)} | {md_cell(rec['title'])} | {rec['author']} "
            f"| {date_str(rec)} | {tier} | {claim} | {card_cell} |"
        )
    return "\n".join(lines)


def build_repo_page(repo: str, recs: list[dict]) -> Path:
    recs = sorted(recs, key=lambda r: (date_str(r), r["number"]), reverse=True)
    major = [r for r in recs if r["tier"] == "major"]
    out = [
        f"# {repo.upper()} 性能优化 PR 全量索引",
        "",
        f"- 仓库: {REPO_URLS[repo]}",
        f"- 收录总数: **{len(recs)}** (重点 🔥 **{len(major)}** / 一般 {len(recs) - len(major)})",
        f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 目录",
        "",
    ]
    counts = Counter(r["category"] for r in recs)
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        if n:
            heading = f"{cat_display(cat)} ({n})"
            out.append(f"- [{heading}](#{gh_anchor(heading)})")
    for cat in CATEGORIES:
        subset = [r for r in recs if r["category"] == cat]
        if not subset:
            continue
        out += ["", f"## {cat_display(cat)} ({len(subset)})", "", repo_table(repo, subset)]
    out += snapshot_lines()
    path = INDEX_DIR / repo / "PERF-PRS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def yearly(recs: list[dict]) -> Counter:
    return Counter(date_str(r)[:4] for r in recs if date_str(r))


def top_speedups(recs: list[dict], n: int = 20) -> list[tuple[dict, dict]]:
    rows = []
    for r in recs:
        for e in r.get("speedup_evidence", []):
            rows.append((e, r))
    rows.sort(key=lambda t: -t[0]["num"])
    return rows[:n]


def build_stats(data: dict[str, list[dict]]) -> Path:
    out = ["# Perf PR 索引统计总览", ""]
    out.append(f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append("## 总量")
    out.append("")
    out.append("| 仓库 | 收录总数 | 🔥 重点 | 一般 | 有加速声明 |")
    out.append("| --- | --- | --- | --- | --- |")
    for repo, recs in data.items():
        claims = sum(1 for r in recs if r.get("speedup_claims"))
        major = sum(1 for r in recs if r["tier"] == "major")
        out.append(
            f"| {repo} | {len(recs)} | {major} | {len(recs) - major} | {claims} |"
        )
    for repo, recs in data.items():
        major = [r for r in recs if r["tier"] == "major"]
        out += ["", f"## {repo}:分类分布", ""]
        out.append("| 分类 | 总数 | 🔥 重点 |")
        out.append("| --- | --- | --- |")
        counts = Counter(r["category"] for r in recs)
        mcounts = Counter(r["category"] for r in major)
        for cat in CATEGORIES:
            if counts.get(cat):
                out.append(f"| {cat_display(cat)} | {counts[cat]} | {mcounts.get(cat, 0)} |")
        out += ["", f"## {repo}:年度分布", ""]
        out.append("| 年份 | 总数 | 🔥 重点 |")
        out.append("| --- | --- | --- |")
        yc, ym = yearly(recs), yearly(major)
        for y in sorted(yc, reverse=True):
            out.append(f"| {y} | {yc[y]} | {ym.get(y, 0)} |")
        authors = Counter(r["author"] for r in recs if r["author"])
        out += ["", f"## {repo}:Top 作者 (前 15)", ""]
        out.append("| 作者 | PR 数 |")
        out.append("| --- | --- |")
        for a, n in authors.most_common(15):
            out.append(f"| [{a}](https://github.com/{a}) | {n} |")
        out += ["", f"## {repo}:加速声明 Top (数字最大前 20)", ""]
        out.append("| 声明 | PR | 标题 | 声明原文 |")
        out.append("| --- | --- | --- | --- |")
        for ev, rec in top_speedups(recs):
            out.append(
                f"| {ev['num']:g}{ev['unit']} | {pr_link(repo, rec)} "
                f"| {md_cell(rec['title'], 60)} | {md_cell(ev['claim'], 70)} |"
            )
        out += ["", f"> 注: 声明数字由正则从 PR 正文提取, 未经核实; 请点击 PR 查看测试条件 (模型/硬件/workload)。", ""]
    out += snapshot_lines()
    path = INDEX_DIR / "STATS.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    args = ap.parse_args()
    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    data: dict[str, list[dict]] = {}
    for repo in repos:
        src = CLASSIFIED_DIR / f"{repo}-classified.jsonl"
        if not src.exists():
            print(f"[{repo}] no classified data ({src}), skipping")
            continue
        recs = []
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines() (U+2028 traps)
            recs = [json.loads(l) for l in f if l.strip()]
        data[repo] = recs
        p = build_repo_page(repo, recs)
        print(f"[{repo}] {len(recs)} PRs -> {p}")
    if data:
        print(f"stats -> {build_stats(data)}")


if __name__ == "__main__":
    main()
