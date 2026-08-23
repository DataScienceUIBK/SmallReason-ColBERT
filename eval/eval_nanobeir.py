"""NanoBEIR sanity eval for SmallReason-ColBERT (with and without the head).

Why NanoBEIR (not full BEIR): each split has ~50 queries × 1-5k docs, runs in
~1-2 min on a single GPU. Full BEIR (NQ 2.7M, MSMARCO 8.8M docs) would take
hours per model. NanoBEIR is the standard sanity-check used by mxbai/sentence-
transformers folks.

Modes
-----
    --weighted  → use WeightedColBERT (importance head)
    (default)   → plain MaxSim (un-headed base)

Usage
-----
    # Plain (no head):
    python eval_nanobeir.py --model <pylate-id-or-path> --output_dir <out> \\
        --splits scifact nfcorpus fiqa2018 touche2020 msmarco nq dbpedia ...

    # Weighted (with head):
    python eval_nanobeir.py --weighted \\
        --base_model <base dir> --head_dir <model dir> \\
        --output_dir <out> --splits ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from pylate import models, evaluation  # noqa: E402

NANOBEIR_HF = {
    "climatefever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia":      "zeta-alpha-ai/NanoDBPedia",
    "fever":        "zeta-alpha-ai/NanoFEVER",
    "fiqa2018":     "zeta-alpha-ai/NanoFiQA2018",
    "hotpotqa":     "zeta-alpha-ai/NanoHotpotQA",
    "msmarco":      "zeta-alpha-ai/NanoMSMARCO",
    "nfcorpus":     "zeta-alpha-ai/NanoNFCorpus",
    "nq":           "zeta-alpha-ai/NanoNQ",
    "quoraretrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs":      "zeta-alpha-ai/NanoSCIDOCS",
    "arguana":      "zeta-alpha-ai/NanoArguAna",
    "scifact":      "zeta-alpha-ai/NanoSciFact",
    "touche2020":   "zeta-alpha-ai/NanoTouche2020",
}


def load_split(split: str):
    hf_id = NANOBEIR_HF[split]
    corpus = load_dataset(hf_id, "corpus", split="train")
    queries = load_dataset(hf_id, "queries", split="train")
    qrels = load_dataset(hf_id, "qrels", split="train")
    return corpus, queries, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Plain ColBERT model dir or HF id")
    ap.add_argument("--weighted", action="store_true",
                    help="Use WeightedColBERT (importance head)")
    ap.add_argument("--base_model", help="base model dir (for --weighted)")
    ap.add_argument("--head_dir", help="model dir containing importance_head/ (for --weighted)")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--query_length", type=int, default=64,
                    help="NanoBEIR queries are short (~10 tokens)")
    ap.add_argument("--document_length", type=int, default=512,
                    help="NanoBEIR docs are short paragraphs")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.weighted:
        from weighted_colbert import WeightedColBERT, IMPORTANCE_DIRNAME
        head_path = Path(args.head_dir) / IMPORTANCE_DIRNAME / "model.safetensors"
        if not head_path.exists():
            raise FileNotFoundError(f"No head at {head_path}")
        model_label = f"weighted({args.base_model}, head={args.head_dir})"
    else:
        if not args.model:
            raise ValueError("--model required when not --weighted")
        model_label = args.model

    print(f"[nanobeir] model={model_label}")

    for split in args.splits:
        out_file = args.output_dir / f"NanoBEIR_{split}_evaluation_scores.json"
        if out_file.exists():
            print(f"[nanobeir] skip {split} (done)"); continue

        print(f"\n=== {split} ===")
        if args.weighted:
            m = WeightedColBERT.from_base(
                args.base_model,
                query_length=args.query_length,
                document_length=args.document_length,
                head_path=head_path,
                device="cuda:0",
            )
            m.eval()
        else:
            m = models.ColBERT(
                model_name_or_path=args.model,
                query_length=args.query_length,
                document_length=args.document_length,
                device="cuda:0",
            )

        corpus, queries, qrels = load_split(split)
        docs = [str(c["text"]) for c in corpus if len(c["text"]) > 0]
        doc_ids = [str(c["_id"]) for c in corpus if len(c["text"]) > 0]
        qs = [str(q["text"]) for q in queries if len(q["text"]) > 0]
        qids = [str(q["_id"]) for q in queries if len(q["text"]) > 0]
        qrels_map = {}
        for r in qrels:
            qid = str(r["query-id"]); cid = str(r["corpus-id"])
            qrels_map.setdefault(qid, set()).add(cid)
        print(f"  queries={len(qs)} docs={len(docs)}")

        d_embs = (m.base if args.weighted else m).encode(
            docs, is_query=False, batch_size=64,
            show_progress_bar=False, convert_to_numpy=True,
        )
        if args.weighted:
            q_embs, q_weights = m.encode(
                qs, is_query=True, batch_size=64,
                show_progress_bar=False, return_weights=True,
            )
        else:
            q_embs = m.encode(
                qs, is_query=True, batch_size=64,
                show_progress_bar=False, convert_to_numpy=True,
            )
            q_weights = None

        N = len(d_embs)
        all_scores = np.zeros((len(qs), N), dtype=np.float32)
        device = "cuda:0"

        # Whole corpus in one chunk (NanoBEIR is small)
        ML = max(d.shape[0] for d in d_embs)
        dim = d_embs[0].shape[1]
        dt = torch.zeros(N, ML, dim, dtype=torch.float16, device=device)
        dm = torch.zeros(N, ML, dtype=torch.bool, device=device)
        for i, d in enumerate(d_embs):
            L = d.shape[0]
            dt[i, :L] = torch.from_numpy(d).to(torch.float16); dm[i, :L] = True
        for qi, qe in enumerate(q_embs):
            q = torch.from_numpy(qe).to(device, dtype=torch.float16)
            sim = torch.einsum("qd,nld->qnl", q, dt)
            sim.masked_fill_(~dm.unsqueeze(0), -1e4)
            max_per_t = sim.max(dim=-1).values
            if q_weights is not None:
                w = torch.from_numpy(q_weights[qi]).to(device, dtype=torch.float16)
                w = w / (w.sum() + 1e-6)
                all_scores[qi, :] = (max_per_t * w.unsqueeze(-1)).sum(dim=0).float().cpu().numpy()
            else:
                all_scores[qi, :] = max_per_t.sum(dim=0).float().cpu().numpy()

        scores_lol = []
        for qi in range(len(qs)):
            order = np.argsort(-all_scores[qi])[:100]
            scores_lol.append([
                {"id": doc_ids[idx], "score": float(all_scores[qi, idx])}
                for idx in order
            ])
        qrels_for_eval = {qid: {cid: 1 for cid in qrels_map.get(qid, set())} for qid in qids}
        ev = evaluation.evaluate(
            scores=scores_lol, qrels=qrels_for_eval, queries=qids,
            metrics=["map", "ndcg@1", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )
        print(f"  {split} → nDCG@10 = {ev['ndcg@10']:.4f}")
        with open(out_file, "w") as f:
            json.dump(ev, f, indent=2)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
