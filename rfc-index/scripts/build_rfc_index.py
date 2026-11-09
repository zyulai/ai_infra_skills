#!/usr/bin/env python3
"""Build the human-readable RFC index from classified RFC issues.

Input : data/classified/{repo}-rfcs-classified.jsonl
Output: index/{repo}/RFC-INDEX.md   (per-category tables, 🔥 tier marks, card links)
        index/STATS.md              (cross-repo statistics)

Stdlib only, offline. Atomic writes (tmp + replace).
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

CATEGORIES = [
    "Speculative Decoding",
    "MoE",
    "Quantization",
    "Disaggregated Serving",
    "CUDA Graph / Compiler",
    "Overlap / Scheduler",
    "Attention / Kernel",
    "KV Cache / Memory",
    "Parallelism / Communication",
    "Sampler / Detokenization",
    "Frontend / Serving / API",
    "Architecture / Engine",
    "Hardware / Backend",
    "Feature / Ecosystem",
    "CI / Infra / Process",
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
    "Disaggregated Serving": "分离式部署",
    "CUDA Graph / Compiler": "CUDA Graph 与编译",
    "Overlap / Scheduler": "重叠与调度",
    "Attention / Kernel": "注意力与算子",
    "KV Cache / Memory": "KV Cache 与显存",
    "Parallelism / Communication": "并行与通信",
    "Sampler / Detokenization": "采样与反分词",
    "Frontend / Serving / API": "前端与服务 API",
    "Architecture / Engine": "架构与引擎",
    "Hardware / Backend": "硬件与后端",
    "Feature / Ecosystem": "功能与生态",
    "CI / Infra / Process": "CI 与基础流程",
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


def snapshot_lines() -> list[str]:
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
            f"RFC {e.get('raw_rfc_issues')} → 重点档 {e.get('major_rfc_issues')} → 摘要卡 {e.get('summary_cards')}"
        )
    return lines


def find_card(repo: str, number: int) -> Path | None:
    cards_dir = INDEX_DIR / repo / "rfc-cards"
    if not cards_dir.is_dir():
        return None
    for p in cards_dir.glob(f"rfc-{number}-*.md"):
        return p
    return None


def md_cell(text: str, limit: int = 90) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def date_str(rec: dict, key: str = "created_at") -> str:
    return (rec.get(key) or "")[:10]


def state_cell(rec: dict) -> str:
    return "🟢 开放" if rec.get("state") == "open" else "⚪ 已关闭"


def repo_table(repo: str, recs: list[dict]) -> str:
    lines = [
        "| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in sorted(recs, key=lambda r: r["number"]):
        card = find_card(repo, rec["number"])
        card_cell = (
            f"[卡片]({card.relative_to(INDEX_DIR / repo).as_posix()})" if card else ""
        )
        tier = "🔥 重点" if rec["tier"] == "major" else "一般"
        claim = md_cell(rec["speedup_claims"][0], 60) if rec.get("speedup_claims") else ""
        lines.append(
            f"| [#{rec['number']}]({rec['html_url']}) | {md_cell(rec['title'])} "
            f"| {rec['author']} | {date_str(rec)} | {state_cell(rec)} | {tier} "
            f"| {claim} | {card_cell} |"
        )
    return "\n".join(lines)


def build_repo_page(repo: str, recs: list[dict]) -> Path:
    major = [r for r in recs if r["tier"] == "major"]
    open_n = sum(1 for r in recs if r.get("state") == "open")
    out = [
        f"# {repo.upper()} RFC 索引(label:rfc ∪ 标题含 RFC,全量)",
        "",
        f"- 仓库: {REPO_URLS[repo]}",
        f"- 收录总数: **{len(recs)}** (重点 🔥 **{len(major)}** / 一般 {len(recs) - len(major)})"
        f" | 开放 {open_n} / 已关闭 {len(recs) - open_n}",
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
    path = INDEX_DIR / repo / "RFC-INDEX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def engagement(rec: dict) -> int:
    return rec.get("reactions_total", 0) * 2 + rec.get("comments", 0)


def build_stats(data: dict[str, list[dict]]) -> Path:
    out = ["# RFC 索引统计总览", ""]
    out.append(f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out += ["", "## 总量", ""]
    out.append("| 仓库 | RFC 总数 | 🔥 重点 | 开放 | 已关闭 | 有收益声明 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for repo, recs in data.items():
        major = sum(1 for r in recs if r["tier"] == "major")
        open_n = sum(1 for r in recs if r.get("state") == "open")
        claims = sum(1 for r in recs if r.get("speedup_claims"))
        out.append(
            f"| {repo} | {len(recs)} | {major} | {open_n} | {len(recs) - open_n} | {claims} |"
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
        out += ["", f"## {repo}:年度分布(按创建时间)", ""]
        out.append("| 年份 | 总数 | 🔥 重点 |")
        out.append("| --- | --- | --- |")
        yc = Counter(date_str(r)[:4] for r in recs)
        ym = Counter(date_str(r)[:4] for r in major)
        for y in sorted(yc, reverse=True):
            out.append(f"| {y} | {yc[y]} | {ym.get(y, 0)} |")
        out += ["", f"## {repo}:热度 Top 20 (reactions×2 + comments)", ""]
        out.append("| 热度 | RFC | 标题 | 状态 |")
        out.append("| --- | --- | --- | --- |")
        for rec in sorted(recs, key=engagement, reverse=True)[:20]:
            out.append(
                f"| {engagement(rec)} | [#{rec['number']}]({rec['html_url']}) "
                f"| {md_cell(rec['title'], 60)} | {state_cell(rec)} |"
            )
        out += ["", f"## {repo}:Top 作者 (前 15)", ""]
        out.append("| 作者 | RFC 数 |")
        out.append("| --- | --- |")
        authors = Counter(r["author"] for r in recs if r["author"])
        for a, n in authors.most_common(15):
            out.append(f"| [{a}](https://github.com/{a}) | {n} |")
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
        src = CLASSIFIED_DIR / f"{repo}-rfcs-classified.jsonl"
        if not src.exists():
            print(f"[{repo}] no classified data ({src}), skipping")
            continue
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
            recs = [json.loads(l) for l in f if l.strip()]
        data[repo] = recs
        p = build_repo_page(repo, recs)
        print(f"[{repo}] {len(recs)} RFCs -> {p}")
    if data:
        print(f"stats -> {build_stats(data)}")


if __name__ == "__main__":
    main()
