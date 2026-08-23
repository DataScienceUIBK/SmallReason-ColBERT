"""Plain MaxSim BRIGHT eval for baseline ColBERT models (no importance head).

Used for §2 of the paper: comparing the base and the full model against:
  - mxbai-edge-colbert-v0-32m (untouched, 64-d)
  - answerai-colbert-small-v1
  - GTE-ModernColBERT-v1
  - lightonai/Reason-ModernColBERT (150M)

Score: standard ColBERT MaxSim, no per-token weights.
    s(q, d) = sum_t max_d (q_t · d_t)

Output JSON shape matches eval_bright.py for consistent aggregation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, load_dataset

from pylate import models, evaluation


def load_arrow(p: Path) -> Dataset:
    """Read a BRIGHT .arrow shard into a Dataset.

    The stored schema metadata can reference `datasets` feature types that the
    installed version does not know about (a cache written by a newer release
    raises "Feature type 'List' not found"). We drop that metadata and let
    Dataset infer features from the arrow schema itself.
    """
    with pa.memory_map(str(p), "r") as mm:
        try:
            table = ipc.open_file(mm).read_all()
        except Exception:
            mm.seek(0)
            table = ipc.open_stream(mm).read_all()
    try:
        return Dataset(table)
    except ValueError:
        return Dataset(table.replace_schema_metadata(None))


def load_bright_split(bright_root: Path | None, split: str) -> tuple[Dataset, Dataset]:
    if bright_root is not None and Path(bright_root).is_dir():
        # Walk recursively under examples/ and documents/ to find bright-<split>.arrow.
        # The HF cache layout is examples/<version>/<hash>/bright-<split>.arrow
        root = Path(bright_root)
        ex_matches = sorted(root.glob(f"examples/**/bright-{split}.arrow"))
        do_matches = sorted(root.glob(f"documents/**/bright-{split}.arrow"))
        if ex_matches and do_matches:
            return load_arrow(ex_matches[-1]), load_arrow(do_matches[-1])
    examples = load_dataset("xlangai/BRIGHT", "examples", split=split)
    documents = load_dataset("xlangai/BRIGHT", "documents", split=split)
    return examples, documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model dir or HF id (any pylate-compatible ColBERT).")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--query_length", type=int, default=256)
    ap.add_argument("--pony_query_length", type=int, default=32)
    ap.add_argument("--document_length", type=int, default=2048)
    ap.add_argument("--bright_root",
                    default=None)
    ap.add_argument("--doc_chunk", type=int, default=4000)
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. cuda:0 or cpu. Default: cuda:0 if available, else cpu.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"[eval] device={device} dtype={dtype}")

    print(f"[plain-bright] model={args.model}")

    for split in args.splits:
        qlen = args.pony_query_length if "pony" in split else args.query_length
        out_file = args.output_dir / f"BrightRetrieval_{split}_evaluation_scores_qlen{qlen}.json"
        if out_file.exists():
            print(f"[plain-bright] skip {split} (already done)"); continue

        print(f"\n=== {split}  qlen={qlen} ===")
        m = models.ColBERT(
            model_name_or_path=args.model,
            query_length=qlen,
            document_length=args.document_length,
            device=device,
        )

        q_ds, d_ds = load_bright_split(args.bright_root, split)
        queries = [str(q_ds[i]["query"]) for i in range(len(q_ds))]
        qids = [str(q_ds[i]["id"]) for i in range(len(q_ds))]
        golds_per_q = [list(q_ds[i]["gold_ids"]) for i in range(len(q_ds))]
        excluded = [list(q_ds[i].get("excluded_ids", [])) for i in range(len(q_ds))]
        docs = [str(d_ds[i].get("content", d_ds[i].get("text", ""))) for i in range(len(d_ds))]
        doc_ids = [str(d_ds[i]["id"]) for i in range(len(d_ds))]
        print(f"  queries={len(queries)}  docs={len(docs)}")

        t0 = time.time()
        d_embs = m.encode(docs, is_query=False, batch_size=64,
                          show_progress_bar=False, convert_to_numpy=True)
        q_embs = m.encode(queries, is_query=True, batch_size=64,
                          show_progress_bar=False, convert_to_numpy=True)
        print(f"  encode took {time.time() - t0:.1f}s")

        dim = d_embs[0].shape[1]
        N = len(d_embs)
        all_scores = np.zeros((len(queries), N), dtype=np.float32)
        for cs in range(0, N, args.doc_chunk):
            ce = min(cs + args.doc_chunk, N)
            chunk = d_embs[cs:ce]
            ML = max(d.shape[0] for d in chunk)
            dt = torch.zeros(len(chunk), ML, dim, dtype=dtype, device=device)
            dm = torch.zeros(len(chunk), ML, dtype=torch.bool, device=device)
            for i, d in enumerate(chunk):
                L = d.shape[0]
                dt[i, :L] = torch.from_numpy(d).to(dtype); dm[i, :L] = True
            for qi, qe in enumerate(q_embs):
                q = torch.from_numpy(qe).to(device, dtype=dtype)
                sim = torch.einsum("qd,nld->qnl", q, dt)
                sim.masked_fill_(~dm.unsqueeze(0), -1e4)
                max_per_t = sim.max(dim=-1).values   # [T_q, N_chunk]
                # Plain MaxSim — just sum (uniform weights)
                all_scores[qi, cs:ce] = max_per_t.sum(dim=0).float().cpu().numpy()
            del dt, dm
            if cs % (args.doc_chunk * 5) == 0:
                print(f"    scored {ce}/{N}")

        scores_lol = []
        for qi in range(len(queries)):
            order = np.argsort(-all_scores[qi])[:100]
            ex = set(excluded[qi]) if isinstance(excluded[qi], list) else set()
            scores_lol.append([
                {"id": doc_ids[idx], "score": float(all_scores[qi, idx])}
                for idx in order if doc_ids[idx] not in ex
            ])
        qrels = {qids[i]: {g: 1 for g in golds_per_q[i]} for i in range(len(queries))}
        ev = evaluation.evaluate(
            scores=scores_lol, qrels=qrels, queries=qids,
            metrics=["map", "ndcg@1", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )
        print(f"  {split} → nDCG@10 = {ev['ndcg@10']:.4f}")
        with open(out_file, "w") as f:
            json.dump(ev, f, indent=2)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
