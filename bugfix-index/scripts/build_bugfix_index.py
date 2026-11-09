#!/usr/bin/env python3
"""Build the human-readable bug-fix index from classified PR data.

Input : data/classified/{repo}-bugfix-classified.jsonl
Output: index/{repo}/BUGFIX-INDEX.md   (per-category tables, 🔥 marks, card links)
        index/STATS.md                 (cross-repo statistics incl. bug-type distribution)

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

BUG_TYPE_ZH = {
    "correctness": "正确性",
    "crash": "崩溃",
    "hang": "挂死",
    "resource": "资源泄漏/OOM",
    "regression": "回归",
    "startup": "启动/加载",
    "general": "一般",
}

REPO_URLS = {
    "vllm": "https://github.com/vllm-project/vllm",
    "sglang": "https://github.com/sgl-project/sglang",
}


def cat_display(cat: str) -> str:
    zh = CATEGORY_ZH.get(cat)
    return f"{cat}({zh})" if zh else cat


def type_display(t: str) -> str:
    zh = BUG_TYPE_ZH.get(t)
    return f"{t}({zh})" if zh else t


def gh_anchor(text: str) -> str:
    """GitHub heading-anchor algorithm: lowercase, drop punctuation (CJK stays),
    spaces -> hyphens."""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)
    return re.sub(r"\s", "-", t)


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
            f"raw {e.get('raw_prs')} → 重点档 {e.get('major_prs')} → 修复卡 {e.get('evidence_cards')}"
        )
    return lines


def find_card(repo: str, number: int) -> Path | None:
    cards_dir = INDEX_DIR / repo / "fix-cards"
    if not cards_dir.is_dir():
        return None
    for p in cards_dir.glob(f"pr-{number}-*.md"):
        return p
    return None


def md_cell(text: str, limit: int = 90) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def date_str(rec: dict) -> str:
    return (rec.get("closed_at") or rec.get("created_at") or "")[:10]  # merged PR: closed==merged


def repo_table(repo: str, recs: list[dict]) -> str:
    lines = [
        "| PR | 标题 | 作者 | 合并日期 | 类型 | 档位 | 修复卡 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in sorted(recs, key=lambda r: (date_str(r), r["number"]), reverse=True):
        card = find_card(repo, rec["number"])
        card_cell = (
            f"[卡片]({card.relative_to(INDEX_DIR / repo).as_posix()})" if card else ""
        )
        tier = "🔥 重点" if rec["tier"] == "major" else "一般"
        lines.append(
            f"| [{rec['number']}]({rec['html_url']}) | {md_cell(rec['title'])} "
            f"| {rec['author']} | {date_str(rec)} | {type_display(rec['bug_type'])} "
            f"| {tier} | {card_cell} |"
        )
    return "\n".join(lines)


def build_repo_page(repo: str, recs: list[dict]) -> Path:
    major = [r for r in recs if r["tier"] == "major"]
    out = [
        f"# {repo.upper()} Bug Fix 全量索引(已合并修复 PR)",
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
    path = INDEX_DIR / repo / "BUGFIX-INDEX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def yearly(recs: list[dict]) -> Counter:
    return Counter(date_str(r)[:4] for r in recs if date_str(r))


def build_stats(data: dict[str, list[dict]]) -> Path:
    out = ["# Bug Fix 索引统计总览", ""]
    out.append(f"- 生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out += ["", "## 总量", ""]
    out.append("| 仓库 | 收录总数 | 🔥 重点 | 正确性 | 崩溃 | 挂死 | 资源 | 回归 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for repo, recs in data.items():
        major = sum(1 for r in recs if r["tier"] == "major")
        tc = Counter(r["bug_type"] for r in recs)
        out.append(
            f"| {repo} | {len(recs)} | {major} | {tc.get('correctness', 0)} "
            f"| {tc.get('crash', 0)} | {tc.get('hang', 0)} | {tc.get('resource', 0)} "
            f"| {tc.get('regression', 0)} |"
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
        out += ["", f"## {repo}:bug 类型分布", ""]
        out.append("| 类型 | 总数 | 🔥 重点 |")
        out.append("| --- | --- | --- |")
        tcounts = Counter(r["bug_type"] for r in recs)
        tmcounts = Counter(r["bug_type"] for r in major)
        for t in BUG_TYPE_ZH:
            if tcounts.get(t):
                out.append(f"| {type_display(t)} | {tcounts[t]} | {tmcounts.get(t, 0)} |")
        out += ["", f"## {repo}:年度分布(按合并时间)", ""]
        out.append("| 年份 | 总数 | 🔥 重点 |")
        out.append("| --- | --- | --- |")
        yc, ym = yearly(recs), yearly(major)
        for y in sorted(yc, reverse=True):
            out.append(f"| {y} | {yc[y]} | {ym.get(y, 0)} |")
        out += ["", f"## {repo}:热度 Top 20 (reactions×2 + comments)", ""]
        out.append("| 热度 | PR | 标题 | 类型 |")
        out.append("| --- | --- | --- | --- |")
        eng = lambda r: r.get("reactions_total", 0) * 2 + r.get("comments", 0)
        for rec in sorted(recs, key=eng, reverse=True)[:20]:
            out.append(
                f"| {eng(rec)} | [{rec['number']}]({rec['html_url']}) "
                f"| {md_cell(rec['title'], 60)} | {type_display(rec['bug_type'])} |"
            )
        out += ["", f"## {repo}:Top 作者 (前 15)", ""]
        out.append("| 作者 | 修复 PR 数 |")
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
        src = CLASSIFIED_DIR / f"{repo}-bugfix-classified.jsonl"
        if not src.exists():
            print(f"[{repo}] no classified data ({src}), skipping")
            continue
        with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
            recs = [json.loads(l) for l in f if l.strip()]
        data[repo] = recs
        p = build_repo_page(repo, recs)
        print(f"[{repo}] {len(recs)} fix PRs -> {p}")
    if data:
        print(f"stats -> {build_stats(data)}")


if __name__ == "__main__":
    main()
