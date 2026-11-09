#!/usr/bin/env python3
"""Classify + tier merged bug-fix PRs for vllm/sglang.

Input : data/raw/<repo>-bugfix-prs.jsonl
Output: data/classified/<repo>-bugfix-classified.jsonl  (all)
        data/major/<repo>-bugfix-major.jsonl            (🔥 tier only)

Two orthogonal dimensions:
- category  : 13 component buckets (same taxonomy as perf-pr-index)
- bug_type  : correctness / crash / hang / resource / regression / startup / general

Tier 🔥 major rules (any one, CI / Test-Perf never major):
  1. correctness bug (wrong output/NaN/accuracy...) — always major
  2. crash/hang/resource fix in a core component, or with real heat
  3. regression fix
  4. reactions >= 5 or comments >= 25
  5. release-highlight / critical / security label
Stdlib only, offline, resume-safe (full rewrite each run).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLASSIFIED = ROOT / "data" / "classified"
MAJOR = ROOT / "data" / "major"

REPOS = ["vllm", "sglang"]

# ---- component categories (first match wins; same taxonomy as perf-pr-index)
CATEGORY_RULES: list[tuple[str, str | None]] = [
    ("CI / Test-Perf",
     r"^(test|bench|ci)[\s/:_\]]|workflow|github actions|\blint\b|format\.|black\b|spell"),
    ("Speculative Decoding",
     r"speculative|eagle|mtp|medusa|ngram|draft (model|token)|lookahead"),
    ("MoE",
     r"\bmoe\b|deepep|expert|all-to-all|grouped gemm|fused moe"),
    ("Quantization",
     r"quant|fp8|fp4|int8|int4|awq|gptq|marlin|modelopt|nvfp4|w4a8|compressed.tensors"),
    ("CUDA Graph / Compiler",
     r"cuda graph|piecewise|torch\.compile|inductor|\bjit\b|compil"),
    ("Overlap / Scheduler",
     r"overlap|schedul|two-batch|\btbo\b|chunked prefill|zero-busy|batch(ing)? (size|order)"),
    ("Attention / Kernel",
     r"attention|flashinfer|flash.attn|\bmla\b|\bnsa\b|triton|kernel|gemm|rmsnorm|\brope\b|\bfa3\b"),
    ("KV Cache / Memory",
     r"kv cache|kv connector|paged|radix|prefix cach|lmcache|hicache|memory pool|offload|kv reuse"),
    ("Parallelism / Communication",
     r"tensor parallel|pipeline parallel|data parallel|allreduce|nccl|nvshmem|sequence parallel|shard"),
    ("Sampler / Detokenization",
     r"sampler|sampling|detokeniz|logits|penalt"),
    ("Frontend / Serving",
     r"api server|openai|router|load balanc|tokenizer|frontend|http|grpc|metrics|serving"),
    ("Model-specific Fast Path",
     r"deepseek|qwen|llama|glm|gpt.oss|gemma|phi.3|mixtral|mistral|internvl|llava|minicpm"
     r"|jamba|granite|olmo|kimi|minimax|seed.oss|baichuan|chatglm|longcat|mamba|hybrid"
     r"|hunyuan|ernie|exaone|arctic|falcon|yi\b"),
    ("Other", None),
]

# ---- bug type (title-based, first match wins)
BUG_TYPE_RULES: list[tuple[str, str]] = [
    ("correctness",
     r"incorrect|wrong (result|output|answer|value|token)|mismatch|inaccura|accuracy"
     r"|silent|corrupt|numerical|\bnan\b|precision|diverge|invalid (output|result|token)"
     r"|garbage|hallucinat|wrong positional|off.by"),
    ("crash",
     r"crash|segfault|\babort\b|core dump|assert(ion)? fail|terminate|c\+\+ exception"
     r"|illegal (memory|instruction)|device.side assert"),
    ("hang",
     r"hang|deadlock|stuck|freeze|infinite (loop|wait)|unresponsive|never (return|finish)"
     r"|block(s|ed|ing)? forever"),
    ("resource",
     r"\boom\b|out of memory|memory leak|\bleak\b|fd leak|unbounded|vram|cuda error"),
    ("regression",
     r"regression|revert|broken (since|in|after)|broke(n)? (in|after|by)"),
    ("startup",
     r"startup|launch fail|fail(s|ed)? (to )?(start|launch|init|load|import)|boot|weight loading"),
]
BUG_TYPE_ZH = {
    "correctness": "正确性",
    "crash": "崩溃",
    "hang": "挂死",
    "resource": "资源泄漏/OOM",
    "regression": "回归",
    "startup": "启动/加载",
    "general": "一般",
}

STRONG_COMPONENT = re.compile(
    r"kernel|attention|flashinfer|\bmoe\b|deepep|quant|cuda graph|schedul|kv cache"
    r"|speculative|\beagle\b|\bmtp\b|parallel|allreduce|nccl|tokenizer|detokeniz"
    r"|prefix cach|radix|logits|sampling|paged", re.I)

HIGH_VALUE_LABELS = {"release-highlight", "critical", "security"}


def classify(rec: dict) -> dict:
    title = rec.get("title", "") or ""
    out = dict(rec)

    cat = "Other"
    for name, pat in CATEGORY_RULES:
        if pat is None:
            cat = name
            break
        if re.search(pat, title, re.I):
            cat = name
            break
    out["category"] = cat

    btype = "general"
    for name, pat in BUG_TYPE_RULES:
        if re.search(pat, title, re.I):
            btype = name
            break
    out["bug_type"] = btype

    react = rec.get("reactions_total", 0)
    comments = rec.get("comments", 0)
    labels = {l.lower() for l in rec.get("labels", [])}
    heat = react >= 5 or comments >= 25
    sev = btype in {"correctness", "crash", "hang", "resource"}
    ci = cat == "CI / Test-Perf"

    if ci:
        tier = "normal"
    elif btype == "correctness" or btype == "regression":
        tier = "major"
    elif sev and (STRONG_COMPONENT.search(title) or react >= 3 or comments >= 10):
        tier = "major"
    elif heat or (labels & HIGH_VALUE_LABELS):
        tier = "major"
    else:
        tier = "normal"
    out["tier"] = tier
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            s = json.dumps(r, ensure_ascii=False)
            s = s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            f.write(s + "\n")
    tmp.replace(path)


def main() -> None:
    for repo in REPOS:
        recs = read_jsonl(RAW / f"{repo}-bugfix-prs.jsonl")
        if not recs:
            print(f"[{repo}] no raw data, skip")
            continue
        out = [classify(r) for r in recs]
        write_jsonl(CLASSIFIED / f"{repo}-bugfix-classified.jsonl", out)
        majors = [r for r in out if r["tier"] == "major"]
        write_jsonl(MAJOR / f"{repo}-bugfix-major.jsonl", majors)

        cats = Counter(r["category"] for r in out)
        types = Counter(r["bug_type"] for r in out)
        print(f"[{repo}] {len(out)} fix PRs | 🔥 major {len(majors)}")
        print("    categories:", dict(cats.most_common()))
        print("    bug types :", dict(types.most_common()))


if __name__ == "__main__":
    main()
