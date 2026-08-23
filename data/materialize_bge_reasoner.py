"""Materialize hanhainebula/bge-reasoner-data to {query, positive, negative}.

Upstream dataset: https://huggingface.co/datasets/hanhainebula/bge-reasoner-data
Files: 12 per-BRIGHT-domain JSONL files under ``bge-reasoner-data-0904/``.
Schema per row: ``{prompt, query, pos:list[str], neg:list[str]}``.

We build BRIGHT-aligned training triples by:
  - Prepending the ``prompt`` to the query in the format used by ReasonEmbed/D:
        ``f"{prompt}\nQuery: {query}"``
    so the model sees the same instruction-style prefix it would see in the
    vm2825 polish stage.
  - Exploding each row into up to ``max_pos × max_neg`` (query, pos, neg) triples.
  - Concatenating all 12 domains (or a single named domain) and shuffling.

Usage
-----
  python LaRLI/data/materialize_bge_reasoner.py \
      --output-dir data/bge_reasoner_processed \
      --max-pos-per-query 2 --max-neg-per-query 4

  # single domain (e.g. aops only) — useful to target a weak split:
  python LaRLI/data/materialize_bge_reasoner.py \
      --split aops --output-dir data/bge_reasoner_aops_processed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "hanhainebula/bge-reasoner-data"
DATA_SUBDIR = "bge-reasoner-data-0904"
BRIGHT_DOMAINS = [
    "biology", "earth_science", "economics", "psychology",
    "robotics", "stackoverflow", "sustainable_living",
    "leetcode", "pony",
    "aops", "theoremqa_questions", "theoremqa_theorems",
]


def _load_domain_jsonl(domain: str, cache_dir: str | None) -> list[dict]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{DATA_SUBDIR}/{domain}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_query(prompt: str, query: str) -> str:
    prompt = (prompt or "").strip()
    query = (query or "").strip()
    if prompt and query:
        return f"{prompt}\nQuery: {query}"
    return prompt or query


def _explode_row(row: dict, max_pos: int, max_neg: int) -> list[dict]:
    q = _format_query(row.get("prompt", ""), row.get("query", ""))
    pos_list = row.get("pos") or []
    neg_list = row.get("neg") or []
    if not q or not pos_list or not neg_list:
        return []
    out: list[dict] = []
    for pos in pos_list[:max_pos]:
        pos = (pos or "").strip()
        if len(pos) < 10:
            continue
        for neg in neg_list[:max_neg]:
            neg = (neg or "").strip()
            if len(neg) < 10:
                continue
            out.append({"query": q, "positive": pos, "negative": neg})
    return out


def _save_atomically(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(tmp))
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp.replace(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize BGE-Reasoner data to triples.")
    p.add_argument("--split", default="all",
                   help=f"'all' or one of {BRIGHT_DOMAINS}. Default: all.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-pos-per-query", type=int, default=2)
    p.add_argument("--max-neg-per-query", type=int, default=4)
    p.add_argument("--cache-dir", default=None,
                   help="HF cache dir. Defaults to HF_HOME.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    domains = BRIGHT_DOMAINS if args.split == "all" else [args.split]
    if args.split != "all" and args.split not in BRIGHT_DOMAINS:
        raise ValueError(f"Unknown split {args.split!r}. Must be 'all' or one of {BRIGHT_DOMAINS}.")

    per_domain: list[Dataset] = []
    grand_raw = grand_trip = 0
    for dom in domains:
        logger.info("Downloading %s/%s ...", DATA_SUBDIR, dom)
        raw = _load_domain_jsonl(dom, args.cache_dir)
        rows: list[dict] = []
        for r in raw:
            rows.extend(_explode_row(r, args.max_pos_per_query, args.max_neg_per_query))
        logger.info("  %s: %d raw rows → %d triples", dom, len(raw), len(rows))
        grand_raw += len(raw)
        grand_trip += len(rows)
        if rows:
            per_domain.append(Dataset.from_list(rows))

    if not per_domain:
        raise RuntimeError("No rows materialized.")
    combined = concatenate_datasets(per_domain).shuffle(seed=42)
    logger.info("Combined bge-reasoner (%d domains): %d raw rows → %d triples",
                len(domains), grand_raw, grand_trip)

    _save_atomically(combined, out)
    logger.info("Saved to %s (%d triples)", out, len(combined))


if __name__ == "__main__":
    main()
