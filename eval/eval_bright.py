"""BRIGHT evaluation for SmallReason-ColBERT (weighted MaxSim with the importance head).

Score (per query, per doc):
    s(q, d) = sum_t  w_t · max_d(q_t · d_t)  /  sum_t w_t

Pony uses query_length=32; all other splits use query_length=256.

Usage
-----
    python eval_bright.py \\
        --base_model ../weights/smallreason-base \\
        --head_dir   ../weights/SmallReason-ColBERT-32M \\
        --output_dir ../BRIGHT_scores_v02 \\
        --splits biology aops theoremqa_theorems leetcode psychology \\
                stackoverflow earth_science economics sustainable_living \\
                robotics theoremqa_questions pony

The repo's ``scripts/eval_bright.sh`` wraps this for 4-GPU parallel eval.
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

# Import WeightedColBERT from sibling src/ folder
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from weighted_colbert import WeightedColBERT, IMPORTANCE_DIRNAME  # noqa: E402

from pylate import evaluation  # noqa: E402


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
    """Load (queries, documents) for a BRIGHT split.

    If bright_root is given, mmap from a local snapshot dir layout matching the
    HF datasets cache; otherwise fall back to load_dataset (downloads on first
    run, then cached automatically).
    """
    if bright_root is not None and Path(bright_root).is_dir():
        # Walk recursively under examples/ and documents/ to find bright-<split>.arrow.
        # The HF cache layout is examples/<version>/<hash>/bright-<split>.arrow
        root = Path(bright_root)
        ex_matches = sorted(root.glob(f"examples/**/bright-{split}.arrow"))
        do_matches = sorted(root.glob(f"documents/**/bright-{split}.arrow"))
        if ex_matches and do_matches:
            return load_arrow(ex_matches[-1]), load_arrow(do_matches[-1])

    # HF Hub — works out of the box, just slower on first call
    examples = load_dataset("xlangai/BRIGHT", "examples", split=split)
    documents = load_dataset("xlangai/BRIGHT", "documents", split=split)
    return examples, documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Path to base model dir")
    ap.add_argument("--head_dir", required=True,
                    help="Dir containing importance_head/ (the model dir containing importance_head/)")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--query_length", type=int, default=256)
    ap.add_argument("--pony_query_length", type=int, default=32)
    ap.add_argument("--document_length", type=int, default=2048)
    ap.add_argument("--bright_root", default=None,
                    help="Optional path to a local BRIGHT snapshot "
                         "(HF datasets cache layout). If unset, downloads via load_dataset.")
    ap.add_argument("--doc_chunk", type=int, default=4000,
                    help="Docs per scoring chunk; lower if you hit OOM.")
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. cuda:0 or cpu. Default: cuda:0 if available, else cpu.")
    ap.add_argument("--dtype", default=None, choices=["float16", "float32"],
                    help="Scoring precision. Default: float16 on GPU, float32 on CPU. "
                         "float32 is slower but removes fp16 accumulation noise.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = ({"float16": torch.float16, "float32": torch.float32}[args.dtype]
             if args.dtype else
             (torch.float16 if device.startswith("cuda") else torch.float32))
    print(f"[eval] device={device} dtype={dtype}")

    # --head_dir may be a local dir or a Hugging Face repo id. When it is a repo
    # id there is no local file to point at, so let from_base resolve the head
    # (it downloads from the Hub and raises if there is none).
    head_path = Path(args.head_dir) / IMPORTANCE_DIRNAME / "model.safetensors"
    if not head_path.exists():
        head_path = None

    for split in args.splits:
        qlen = args.pony_query_length if "pony" in split else args.query_length
        out_file = args.output_dir / f"BrightRetrieval_{split}_evaluation_scores_qlen{qlen}.json"
        if out_file.exists():
            print(f"[eval] skip {split} (already done)"); continue

        print(f"\n=== {split}  qlen={qlen} ===")
        m = WeightedColBERT.from_base(
            args.base_model, query_length=qlen,
            document_length=args.document_length,
            head_path=head_path, device=device,
        )
        m.eval()

        q_ds, d_ds = load_bright_split(args.bright_root, split)
        queries = [str(q_ds[i]["query"]) for i in range(len(q_ds))]
        qids = [str(q_ds[i]["id"]) for i in range(len(q_ds))]
        golds_per_q = [list(q_ds[i]["gold_ids"]) for i in range(len(q_ds))]
        excluded = [list(q_ds[i].get("excluded_ids", [])) for i in range(len(q_ds))]
        docs = [str(d_ds[i].get("content", d_ds[i].get("text", ""))) for i in range(len(d_ds))]
        doc_ids = [str(d_ds[i]["id"]) for i in range(len(d_ds))]
        print(f"  queries={len(queries)}  docs={len(docs)}")

        t0 = time.time()
        d_embs = m.base.encode(docs, is_query=False, batch_size=64,
                               show_progress_bar=False, convert_to_numpy=True)
        q_embs, q_weights = m.encode(queries, is_query=True, batch_size=64,
                                     show_progress_bar=False, return_weights=True)
        print(f"  encode took {time.time() - t0:.1f}s")
        all_w = np.concatenate(q_weights)
        print(f"  per-token weights: mean={all_w.mean():.3f} std={all_w.std():.3f} "
              f"min={all_w.min():.3f} max={all_w.max():.3f}")

        # Brute-force weighted MaxSim, doc-chunked
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
                w = torch.from_numpy(q_weights[qi]).to(device, dtype=dtype)
                w = w / (w.sum() + 1e-6)            # normalise at eval (per-query mean)
                sim = torch.einsum("qd,nld->qnl", q, dt)
                sim.masked_fill_(~dm.unsqueeze(0), -1e4)
                max_per_t = sim.max(dim=-1).values  # [T_q, N_chunk]
                all_scores[qi, cs:ce] = (max_per_t * w.unsqueeze(-1)).sum(dim=0).float().cpu().numpy()
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
