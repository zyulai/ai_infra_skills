#!/usr/bin/env python3
"""Classify fetched PRs into perf categories and mark the major tier.

Input : data/raw/{repo}-prs.jsonl            (from fetch_prs.py)
Output: data/classified/{repo}-classified.jsonl  (all PRs, with category/tier)
        data/major/{repo}-major.jsonl            (major-tier subset)

Stdlib only, fully deterministic (keyword rules, no network).
Rules are documented in docs/methodology.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLASSIFIED_DIR = ROOT / "data" / "classified"
MAJOR_DIR = ROOT / "data" / "major"

# ---------------------------------------------------------------------------
# Category rules: first match in this priority order wins.
# Each entry: (category, compiled regex applied to title; a second list is
# applied to title+body only when the title has no match).
# ---------------------------------------------------------------------------

CI_TITLE = re.compile(
    r"^(test|bench|ci)\b|^\[(test|ci|bench)\]|\bbenchmark (suite|harness|runner|job|tool|script)\b"
    r"|\b(nightly|perf test|performance test|regression test)\b"
    r"|\bbenchmark[s]?\b|vllm bench|\bdocs?\b|documentation|markdown|pre-commit|spdx|readme",
    re.I,
)
OPT_TITLE = re.compile(r"fused|optim|speedup|speed up|faster|accelerat|throughput|latency", re.I)

CATEGORY_RULES: list[tuple[str, list[re.Pattern], list[re.Pattern]]] = [
    (
        "Speculative Decoding",
        [re.compile(r"speculative|eagle(?!scope)|\bmtp\b|\bmedusa\b|\bngram\b|n-gram|draft (model|token)|lookahead|\bscoop\b|dflash|\bmpp?\b|parallel sampling", re.I)],
        [re.compile(r"speculative decod|draft model|target model verif|accept(ance)? (length|rate)", re.I)],
    ),
    (
        "MoE",
        [re.compile(r"\bmoe\b|mixture[- ]of[- ]experts|\bdeepep\b|deep_ep|fused[- ]moe|expert[- ]parallel|\ball[- ]to[- ]all\b|\ba2a\b|expert (load|weight|specialization)|moe (kernel|quant|runner|fused|ep)", re.I)],
        [re.compile(r"experts?\b.*(kernel|dispatch|combine|routing|topk|grouped gemm)|grouped (gemm|masked)", re.I)],
    ),
    (
        "Quantization",
        [re.compile(r"\bquant\b|quantiz|fp8|fp4|int8|int4|\bw4a8\b|\bw8a8\b|\bawq\b|\bgptq\b|modelopt|compressed[- ]tensors|\bmarlin\b|\bmachete\b|\bgguf\b|nvfp4|mxfp4|\bfbgemm\b|\bquark\b|per[-_ ](channel|token|block)|block[- ]wise scale|\bdsv3\b fingerprint|e4m3|e5m2", re.I)],
        [re.compile(r"weight (only |)quant|activation quant|scaled mm|fused (quant|dequant)|quant per", re.I)],
    ),
    (
        "CUDA Graph / Compiler",
        [re.compile(r"cuda[- ]?graph|piecewise|\btorch\.compile\b|torch compile|\bjit\b|aot(?!oriz)|depyf|\binductor\b|graph (capture|replay|pad)|multi[- ]graph", re.I)],
        [re.compile(r"compiled (graph|kernel)|compilation (cache|time)|capture cuda", re.I)],
    ),
    (
        "Overlap / Scheduler",
        [re.compile(r"overlap|scheduler|scheduling|\basync (schedule|output|sched)|two[- ]batch|\btbo\b|disaggregat|\bpd (disagg|split)|prefill[- ]decode|zero[- ]busy|\bdbd\b|multi[- ]thread|thread (pool|safe)|batch (scheduler|manager)|chunked (prefill|request)|\bvllm scheduler\b|epd|nixl", re.I)],
        [re.compile(r"overlap (schedule|compute|communication|cpu)|cpu[- ]gpu overlap|schedule (batch|policy)", re.I)],
    ),
    (
        "Attention / Kernel",
        [re.compile(r"attention|flashinfer|flash[- ]?attn|flashattention|\bfa3\b|\bfa4\b|\bmla\b|\bnsa\b|indexer|paged attention|\btriton\b|kernel|\bgemm\b|\bgemv\b|rmsnorm|layernorm|\brope\b|\bsilu\b|custom allreduce|softmax|fused (op|kernel|add|rms|rope|act)|elementwise|vectoriz|\bsgl[- ]kernel\b|aiter|ck tile|\btl\.\b", re.I)],
        [re.compile(r"kernel (launch|fusion|fuse|optim|tuning|cache)|autotune|tuning (config|space)|cuda core|cutlass", re.I)],
    ),
    (
        "KV Cache / Memory",
        [re.compile(r"kv[- ]?cache|paged|radix (cache|tree)|prefix cach|kv connector|lmcache|\bhicache\b|memory (pool|alloc|usage|leak)|block (manager|pool|size|allocation)|token pool|host cache|offload|swa\b|sliding window|kv (share|transfer|reuse|layout|retrieval)|hierarchical cache|free slot|eviction", re.I)],
        [re.compile(r"cache (hit|miss|reuse|management)|reduce (memory|gpu mem)|memory (footprint|saving|efficien)", re.I)],
    ),
    (
        "Parallelism / Communication",
        [re.compile(r"tensor[- ]parallel|\btp\b (parallel|attention|degree)|pipeline[- ]parallel|\bpp\b (parallel|stage)|data[- ]parallel|\ball[- ]?reduce\b|\bnccl\b|\bnvshmem\b|communicat(or|ion)|sequence[- ]parallel|\bdp (attention|size|rank)|custom (allreduce|comm)|symmetric memory|mnnvl|multi[- ]node|single[- ]node (multi|nvlink)|\bipc\b|\bzs?b\b|zero bubble", re.I)],
        [re.compile(r"parallel (strategy|degree|group)|comm (fusion|overlap|kernel)|communication (overlap|efficien)", re.I)],
    ),
    (
        "Sampler / Detokenization",
        [re.compile(r"sampler|sampling|detokeniz|logits|logit processor|\btop[- ]?k\b|\btop[- ]?p\b|topk|topp|\bpenalt|\bgreedy\b|softmax (sampling)?|logit (processor|bias)|speculative sampler|argmax", re.I)],
        [re.compile(r"sampling (speed|perform|efficien)|batch (sampling|logits)", re.I)],
    ),
    (
        "Frontend / Serving",
        [re.compile(r"\brouter\b|load balanc|api server|openai (server|api|compatible)|frontend|tokenizer|detokeniz|http|websocket|\bqps\b|max[- ]concurrency|serving (layer|stack)|service (layer)|\bsgl[- ]router\b|request (queue|routing)", re.I)],
        [],
    ),
]

MODEL_NAMES = re.compile(
    r"deepseek|qwen|llama|\bglm\b|gpt[- ]?oss|hunyuan|ernie|kimi|minimax|gemma|\bphi\b|mistral|mixtral|internlm|baichuan|yi[- ]\d|stablelm|olmo|\bjamba\b|falcon|\bmpt\b|opt[- ]\d|bloom|chatglm| granite|llava|qwen[- ]vl|glm[- ]v|kimi[- ]vl|dots|ernie|seed[- ]oss|lfm|apertus|exaone|pixtral|molmo|cobra| significance",
    re.I,
)

# Fallback order preserved for docs.
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

# ---------------------------------------------------------------------------
# Tier rules
# ---------------------------------------------------------------------------

HIGH_VALUE_LABELS = {
    "performance",
    "sgl-kernel",
    "quant",
    "quantization",
    "speculative-decoding",
    "torch.compile",
    "release-highlight",
    "flashinfer",
    "jit-kernel",
    "kv-cache-manager",
    "kv-connector",
    "kernel",
}

# labels that on their own (with strong title) mark a release-worthy optimization
TOP_LABELS = {"release-highlight", "performance"}

TITLE_STRONG = re.compile(
    r"\bfused\b|\bfusion\b|cuda[- ]?graph|\boverlap\b|\bdeepep\b|deep_ep|flashinfer"
    r"|\beagle\b|\bmtp\b|speculative|\btriton\b|\bmla\b|\bnsa\b|flash[- ]?attn|\bquant\b|quantiz"
    r"|\brope\b|rmsnorm|\ballreduce\b|all[- ]reduce|\bv1\b engine|zero[- ]busy|two[- ]batch"
    r"|chunked prefill|prefix cach|paged attention|\bmarlin\b|\bmachete\b|piecewise",
    re.I,
)

# Numbered perf claims inside the PR body.
CLAIM_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(faster|speedup|speed[- ]up|higher|improvement|improvement|more (throughput|tokens?)|throughput|performance)", re.I),
    re.compile(r"(faster|speedup|speed[- ]up|speed up)[^.]{0,60}?(\d+(?:\.\d+)?)\s*[xX×]", re.I),
    re.compile(r"up to (\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"((?:throughput|latency|ttft|tpot|itl|e2e|performance|speed)[^.]{0,80}?(?:improve|increase|reduce|boost|gain|speedup|speed[- ]up)[^.]{0,40}?\d+(?:\.\d+)?\s*%)", re.I),
    re.compile(r"(\d+(?:\.\d+)?\s*%[^.]{0,50}?(?:faster|improvement|improvement|higher|speedup|throughput|latency (?:reduction|improvement)))", re.I),
    re.compile(r"(speedup|speed[- ]up) (?:of |by |: )?(\d+(?:\.\d+)?)[xX× ]", re.I),
]


def extract_claims(body: str) -> tuple[list[str], list[float], list[str], list[dict]]:
    """Return (claim snippets, numbers, units, evidence pairs). A number is
    trusted only when directly adjacent to a % or x/X/× marker inside the
    snippet, with sanity bounds (percent 1..200, multiplier 1.05..50) to
    reject token-count noise; markdown-table fragments are dropped."""
    if not body:
        return [], [], [], []
    snippets: list[str] = []
    for pat in CLAIM_PATTERNS:
        for m in pat.finditer(body):
            text = m.group(0).strip().replace("\n", " ")
            if len(text) > 160:
                text = text[:160]
            snippets.append(text)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in snippets:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        # benchmark-table fragments (markdown rows / code fences) are noise
        if "|" in s or "```" in s:
            continue
        uniq.append(s)
    evidence: list[dict] = []
    for s in uniq:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", s):
            v = float(m.group(1))
            if 1 <= v <= 200:
                evidence.append({"num": v, "unit": "%", "claim": s})
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[xX×](?![a-zA-Z])", s):
            v = float(m.group(1))
            if 1.05 <= v <= 50:
                evidence.append({"num": v, "unit": "x", "claim": s})
    evidence = evidence[:8]
    return (
        uniq[:5],
        [e["num"] for e in evidence],
        [e["unit"] for e in evidence],
        evidence,
    )


def classify_category(rec: dict) -> str:
    title = rec["title"]
    if CI_TITLE.search(title) and not OPT_TITLE.search(title):
        return "CI / Test-Perf"
    body = rec.get("body") or ""
    for _, title_res, body_res in CATEGORY_RULES:
        for pat in title_res:
            if pat.search(title):
                return _norm(cat := _lookup_cat(pat))
        for pat in body_res:
            if pat.search(title) or pat.search(body[:12000]):
                return _lookup_cat(pat)
    if MODEL_NAMES.search(title):
        return "Model-specific Fast Path"
    # model name in body with optimization wording
    if MODEL_NAMES.search(body[:6000]) and OPT_TITLE.search(body[:6000]):
        return "Model-specific Fast Path"
    return "Other"


def _lookup_cat(pat: re.Pattern) -> str:
    for cat, tres, bres in CATEGORY_RULES:
        if pat in tres or pat in bres:
            return cat
    return "Other"  # pragma: no cover


def _norm(cat: str) -> str:
    return cat


def assign_tier(rec: dict, category: str, claims: list[str], numbers: list[float]) -> str:
    if category == "CI / Test-Perf":
        return "normal"
    labels = set(rec.get("labels") or [])
    strong_title = bool(TITLE_STRONG.search(rec["title"]))
    # 1. explicit benchmark numbers in the body
    if numbers:
        return "major"
    # 2. community heat
    if rec.get("reactions_total", 0) >= 10 or rec.get("comments", 0) >= 25:
        return "major"
    # 3. release-worthy label + strong optimization title
    if labels & TOP_LABELS and strong_title:
        return "major"
    # 4. perf wording in body + strong optimization title
    if claims and strong_title:
        return "major"
    return "normal"


def run(repo: str) -> dict:
    src = RAW_DIR / f"{repo}-prs.jsonl"
    if not src.exists():
        raise SystemExit(f"missing {src}; run fetch_prs.py first")
    records = []
    with src.open(encoding="utf-8") as f:  # iterate; don't use splitlines() (U+2028 traps)
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[{repo}] WARN: skipping malformed line in {src.name}")

    cat_count: dict[str, int] = {}
    tier_count = {"major": 0, "normal": 0}
    classified = []
    for rec in records:
        cat = classify_category(rec)
        claims, numbers, units, evidence = extract_claims(rec.get("body") or "")
        tier = assign_tier(rec, cat, claims, numbers)
        rec2 = dict(rec)
        rec2["category"] = cat
        rec2["tier"] = tier
        rec2["speedup_claims"] = claims
        rec2["speedup_numbers"] = numbers
        rec2["speedup_units"] = units
        rec2["speedup_evidence"] = evidence
        classified.append(rec2)
        cat_count[cat] = cat_count.get(cat, 0) + 1
        tier_count[tier] += 1

    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    MAJOR_DIR.mkdir(parents=True, exist_ok=True)
    out_all = CLASSIFIED_DIR / f"{repo}-classified.jsonl"
    out_major = MAJOR_DIR / f"{repo}-major.jsonl"
    for path, subset in (
        (out_all, classified),
        (out_major, [r for r in classified if r["tier"] == "major"]),
    ):
        with path.open("w", encoding="utf-8") as f:
            for rec in sorted(subset, key=lambda r: r["number"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[{repo}] total {len(classified)} | major {tier_count['major']} | normal {tier_count['normal']}")
    for cat in CATEGORIES:
        print(f"    {cat:<32} {cat_count.get(cat, 0)}")
    return {"total": len(classified), "major": tier_count["major"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", choices=["vllm", "sglang", "both"], default="both")
    args = ap.parse_args()
    repos = ["vllm", "sglang"] if args.repo == "both" else [args.repo]
    for repo in repos:
        run(repo)


if __name__ == "__main__":
    main()
