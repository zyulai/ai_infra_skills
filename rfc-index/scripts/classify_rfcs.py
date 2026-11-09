#!/usr/bin/env python3
"""Classify + tier RFC issues for vllm/sglang.

Input : data/raw/<repo>-rfcs.jsonl
Output: data/classified/<repo>-rfcs-classified.jsonl  (all)
        data/major/<repo>-rfcs-major.jsonl            (🔥 tier only)

Categories are tuned for design proposals (superset of the perf-pr-index
categories so cross-referencing stays natural). Tier rules:
  🔥 major  = (perf signal AND engagement) OR high engagement OR strong title
  normal    = everything else
Stdlib only, offline, resume-safe (full rewrite each run).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLASSIFIED = ROOT / "data" / "classified"
MAJOR = ROOT / "data" / "major"

REPOS = ["vllm", "sglang"]

# (category, title-regex | None for fallback). First match wins.
CATEGORY_RULES: list[tuple[str, str | None]] = [
    ("CI / Infra / Process",
     r"^(test|bench|ci)\b|benchmark|release process|contribut|roadmap|governance|community|repo structure|monorepo"),
    ("Speculative Decoding",
     r"speculative|eagle|mtp|medusa|ngram|draft (model|token)|lookahead"),
    ("MoE",
     r"\bmoe\b|deepep|expert|all-to-all|grouped gemm|fused moe"),
    ("Quantization",
     r"quant|fp8|fp4|int8|int4|awq|gptq|marlin|modelopt|nvfp4|w4a8|compressed.tensors|weight.only"),
    ("Disaggregated Serving",
     r"disaggregat|pd disagg|prefill.decode (separation|split)|kv transfer|mooncake|nixl"),
    ("CUDA Graph / Compiler",
     r"cuda graph|piecewise|torch\.compile|inductor|\bjit\b|compil"),
    ("Overlap / Scheduler",
     r"overlap|schedul|two-batch|\btbo\b|chunked prefill|zero-busy|batching"),
    ("Attention / Kernel",
     r"attention|flashinfer|flash.attn|\bmla\b|\bnsa\b|triton|kernel|gemm|rmsnorm|\brope\b|\bfa3\b"),
    ("KV Cache / Memory",
     r"kv cache|kv connector|paged|radix|prefix cach|lmcache|hicache|memory pool|offload|kv reuse|memory"),
    ("Parallelism / Communication",
     r"tensor parallel|pipeline parallel|data parallel|allreduce|nccl|nvshmem|sequence parallel|shard"),
    ("Sampler / Detokenization",
     r"sampler|sampling|detokeniz|logits|penalt"),
    ("Frontend / Serving / API",
     r"api server|openai|router|load balanc|tokenizer|frontend|http|grpc|metrics|structured output|function call"),
    ("Architecture / Engine",
     r"engine|worker|multiproc|asyncio|refactor|architecture|\bv1\b|loopy|lifecycle|startup|core loop"),
    ("Hardware / Backend",
     r"\bnpu\b|ascend|rocm|\btpu\b|\bcpu\b|xpu|hardware|pluggable|device layer"),
    ("Feature / Ecosystem",
     r"support|integrat|adopt|enable|add |adding|new (model|backend|feature)|dependency|survey|wish"),
    ("Other", None),
]

PERF_TITLE = re.compile(
    r"perf|\bspeedup\b|faster|throughput|latency|optimiz|fused|overlap"
    r"|cuda graph|overhead|efficient|accelerat|\bfast\b|\bgpu util", re.I)

# RFCs whose subject is inherently a hot perf topic -> major even without engagement
STRONG_TITLE = re.compile(
    r"fused|\bkernel|cuda graph|deepep|flashinfer|\bmla\b|\bnsa\b|disaggregat"
    r"|speculative|\beagle\b|\bmtp\b|allreduce|zero-busy|two-batch|pd disagg", re.I)

PERF_WORD = re.compile(r"speed ?up|faster|throughput|latency|performance|slow|overhead|gpu util", re.I)
NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|x)\b", re.I)


def _clean_line(s: str) -> str:
    return s.strip().lstrip("#>-* ").strip()


def body_claims(body: str) -> list[dict]:
    """Extract speedup-ish claims: perf word + (%/x)-adjacent number, same sentence.
    Rejects markdown-table / code-block fragments. Numbers sanity-bounded."""
    if not body:
        return []
    body = re.sub(r"```.*?```", " ", body, flags=re.S)  # drop code blocks
    claims = []
    seen = set()
    for chunk in re.split(r"[.!?\n]", body):
        if "|" in chunk:
            continue
        if not PERF_WORD.search(chunk):
            continue
        for m in NUM_UNIT.finditer(chunk):
            num, unit = float(m.group(1)), m.group(2).lower()
            if unit == "%" and not (1 <= num <= 200):
                continue
            if unit == "x" and not (1.05 <= num <= 50):
                continue
            key = (num, unit)
            if key in seen:
                continue
            seen.add(key)
            claims.append({
                "num": num,
                "unit": unit,
                "claim": chunk.strip()[:200],
            })
    return claims


def classify(rec: dict) -> dict:
    title = rec.get("title", "") or ""
    body = rec.get("body", "") or ""
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

    claims = body_claims(body)
    out["speedup_claims"] = [c["claim"] for c in claims]
    out["speedup_evidence"] = claims

    perf = bool(PERF_TITLE.search(title)) or bool(claims) \
        or "performance" in [l.lower() for l in rec.get("labels", [])]
    out["perf"] = perf

    react = rec.get("reactions_total", 0)
    comments = rec.get("comments", 0)
    if react >= 10 or comments >= 30:
        tier = "major"
    elif perf and (react >= 5 or comments >= 10):
        tier = "major"
    elif STRONG_TITLE.search(title):
        tier = "major"
    else:
        tier = "normal"
    out["tier"] = tier
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:  # iterate; don't use splitlines()
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
        recs = read_jsonl(RAW / f"{repo}-rfcs.jsonl")
        if not recs:
            print(f"[{repo}] no raw data, skip")
            continue
        out = [classify(r) for r in recs]
        write_jsonl(CLASSIFIED / f"{repo}-rfcs-classified.jsonl", out)
        majors = [r for r in out if r["tier"] == "major"]
        write_jsonl(MAJOR / f"{repo}-rfcs-major.jsonl", majors)

        from collections import Counter
        cats = Counter(r["category"] for r in out)
        open_n = sum(1 for r in out if r.get("state") == "open")
        print(f"[{repo}] {len(out)} RFCs | 🔥 major {len(majors)} | open {open_n} "
              f"/ closed {len(out) - open_n}")
        for c, n in cats.most_common():
            print(f"    {n:5d}  {c}")


if __name__ == "__main__":
    main()
