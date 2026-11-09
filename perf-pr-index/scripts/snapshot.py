#!/usr/bin/env python3
"""Record the data snapshot (upstream HEAD commit ids + fetch window + counts)
into data/snapshot.json. build_index.py embeds this into every generated md so
future incremental updates have an unambiguous baseline.

Usage:
  export GITHUB_TOKEN=<token>
  python scripts/snapshot.py            # query HEAD SHAs via GitHub API
  python scripts/snapshot.py --local    # offline: use local clones' git HEAD
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot.json"

REPOS = {
    "vllm": {"full": "vllm-project/vllm", "clone": Path(r"C:\Users\Norma\Desktop\ai_infra\vllm")},
    "sglang": {"full": "sgl-project/sglang", "clone": Path(r"C:\Users\Norma\Desktop\ai_infra\sglang")},
}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:  # iterate; splitlines() U+2028 trap
        n = sum(1 for line in f if line.strip())
    return n


def fetch_window(repo: str) -> tuple[str, str]:
    """min/max fetched_at across the raw jsonl (true API fetch window)."""
    lo, hi = "", ""
    raw = ROOT / "data" / "raw" / f"{repo}-prs.jsonl"
    if raw.exists():
        with raw.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line).get("fetched_at", "")
                lo = t if not lo or t < lo else lo
                hi = t if t > hi else hi
    return lo, hi


def head_via_api(full: str, token: str) -> dict:
    req = urllib.request.Request(f"https://api.github.com/repos/{full}/commits/HEAD")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "perf-pr-index-snapshot")
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        c = json.loads(resp.read().decode("utf-8"))
    return {
        "head_sha": c.get("sha", ""),
        "head_date": (c.get("commit") or {}).get("committer", {}).get("date", ""),
        "head_message": ((c.get("commit") or {}).get("message") or "").splitlines()[0][:100],
    }


def head_via_git(clone: Path) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone,
                         capture_output=True, text=True, check=True).stdout.strip()
    date = subprocess.run(["git", "log", "-1", "--format=%cI"], cwd=clone,
                          capture_output=True, text=True, check=True).stdout.strip()
    msg = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=clone,
                         capture_output=True, text=True, check=True).stdout.strip()[:100]
    return {"head_sha": sha, "head_date": date, "head_message": msg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", action="store_true", help="use local clones' git HEAD instead of the API")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    snap = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repos": {}}
    for key, info in REPOS.items():
        lo, hi = fetch_window(key)
        entry = {
            "repo_full": info["full"],
            "fetch_window_utc": [lo, hi],
            "raw_prs": count_lines(ROOT / "data" / "raw" / f"{key}-prs.jsonl"),
            "classified_prs": count_lines(ROOT / "data" / "classified" / f"{key}-classified.jsonl"),
            "major_prs": count_lines(ROOT / "data" / "major" / f"{key}-major.jsonl"),
            "evidence_cards": sum(1 for _ in (ROOT / "index" / key / "cards").glob("pr-*.md")),
        }
        try:
            entry.update(head_via_git(info["clone"]) if args.local else head_via_api(info["full"], token))
        except Exception as e:  # noqa: BLE001 - record the failure, keep counts
            entry["head_error"] = str(e)
        snap["repos"][key] = entry
        print(f"[{key}] head={entry.get('head_sha', '?')[:12]} raw={entry['raw_prs']} "
              f"major={entry['major_prs']} cards={entry['evidence_cards']}")

    SNAP.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAP.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(SNAP)
    print(f"snapshot -> {SNAP.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
