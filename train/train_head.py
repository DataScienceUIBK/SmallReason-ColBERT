"""Train the importance head on top of a frozen reasoning-tuned base.

Single-GPU. ~5 minutes on H100. The base ColBERT is FROZEN; only the
1-layer (~129 params) ImportanceHead trains.

    Score during TRAINING (un-normalised, kept at MaxSim scale so CE has gradient):
        sum_t  w_t · max_d (q_t · d_t)

    Score at EVAL time (normalised by sum(w), see eval.py):
        sum_t  w_t · max_d (q_t · d_t)  /  sum_t w_t

Usage
-----
    python train.py \\
        --base_model        ../weights/smallreason-base \\
        --train_data        ../data_processed/bge_hn_merged_processed \\
        --output_dir        ../weights/SmallReason-ColBERT-32M \\
        --max_steps 3000 --batch_size 16 --lr 5e-4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Import WeightedColBERT from sibling src/ folder
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from weighted_colbert import WeightedColBERT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="Path to base model dir")
    p.add_argument("--train_data", required=True,
                   help="Path to materialized HF dataset with columns "
                        "{query, positive, negative}")
    p.add_argument("--output_dir", required=True,
                   help="Where to save importance_head/ + intermediate checkpoints")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--query_length", type=int, default=256)
    p.add_argument("--document_length", type=int, default=2048)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=1000,
                   help="Intermediate checkpoint stride (0 disables).")
    p.add_argument("--seed", type=int, default=42)
    # Ablation knobs
    p.add_argument("--score_mode", choices=["unnorm", "norm"], default="unnorm",
                   help="Training-time score formula. 'unnorm' = paper recipe "
                        "(Σ w·max). 'norm' = the broken variant (Σ w·max / Σ w).")
    p.add_argument("--init_bias", type=float, default=5.0,
                   help="Sigmoid bias init for the head. b=5 → ~0.99 at step 0.")
    p.add_argument("--init_random", action="store_true",
                   help="Random N(0,1) init for the head weight (paper uses zeros).")
    p.add_argument("--loss_csv", type=str, default=None,
                   help="If set, log per-step (step, loss, head_w_mean, head_w_std) to this CSV.")
    p.add_argument("--device", default=None,
                   help="torch device, e.g. cuda:0 or cpu. "
                        "Default: cuda:0 if available, else cpu.")
    p.add_argument("--head_hidden", type=int, default=0,
                   help="If >0, use 2-layer MLP head Linear(D,H)→ReLU→Linear(H,1). "
                        "0 = paper default (1-layer, 129 params).")
    return p.parse_args()


def encode_with_weights(model, texts, is_query):
    embs = model.base.encode(
        texts, is_query=is_query, batch_size=len(texts),
        show_progress_bar=False, convert_to_numpy=False,
    )
    if is_query:
        weights, embs_grad = [], []
        for e in embs:
            e = e.detach().to(model.device(), dtype=torch.float32)
            weights.append(model.importance(e))   # grad-on (head trainable)
            embs_grad.append(e)
        return embs_grad, weights
    return [e.detach().to(model.device(), dtype=torch.float32) for e in embs]


def weighted_score_train(q_emb, q_w, d_emb, mode="unnorm"):
    """Weighted MaxSim score during training.

    'unnorm' (paper):  Σ_t w_t · max_d (q_t · d_t)              ← real CE gradient
    'norm'   (broken): Σ_t w_t · max_d (q_t · d_t)  /  Σ_t w_t  ← gradient ≈ 0.005, loss stalls at ln(2)
    """
    sim = q_emb @ d_emb.T                # [T_q, T_d]
    max_per_q = sim.max(dim=-1).values   # [T_q]
    s = (max_per_q * q_w).sum()
    if mode == "norm":
        s = s / (q_w.sum().clamp(min=1e-6))
    return s


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[train-head] base:        {args.base_model}")
    print(f"[train-head] train_data:  {args.train_data}")
    print(f"[train-head] output_dir:  {args.output_dir}")
    print(f"[train-head] hparams:     lr={args.lr}  bs={args.batch_size}  "
          f"max_steps={args.max_steps}")

    model = WeightedColBERT.from_base(
        args.base_model, query_length=args.query_length,
        document_length=args.document_length,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        head_hidden=args.head_hidden,
        # We are training the head from scratch, so the base is expected NOT to
        # have one yet. Only inference paths should demand a trained head.
        require_head=False,
    )
    # Re-init the head per ablation flags (overrides the WeightedColBERT default of W=0, b=5)
    with torch.no_grad():
        if args.head_hidden > 0:
            # 2-layer head: re-init only the OUTPUT layer to honour ablation flags;
            # leave the hidden Xavier init alone so the head has signal to learn from.
            if args.init_random:
                model.importance.fc2.weight.normal_(mean=0.0, std=1.0)
            else:
                model.importance.fc2.weight.zero_()
            model.importance.fc2.bias.fill_(args.init_bias)
        else:
            if args.init_random:
                model.importance.linear.weight.normal_(mean=0.0, std=1.0)
            else:
                model.importance.linear.weight.zero_()
            model.importance.linear.bias.fill_(args.init_bias)
    print(f"[train-head] score_mode={args.score_mode}  "
          f"init_bias={args.init_bias}  init_random={args.init_random}")
    print(f"[train-head] head params: "
          f"{sum(p.numel() for p in model.importance.parameters())}")
    print(f"[train-head] base frozen — trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    csv_f = None
    if args.loss_csv:
        csv_f = open(args.loss_csv, "w")
        csv_f.write("step,loss,head_w_mean,head_w_std,head_w_min,head_w_max\n")

    ds = load_from_disk(args.train_data).select_columns(["query", "positive", "negative"])
    print(f"[train-head] dataset: {len(ds):,} rows")

    opt = torch.optim.AdamW(model.importance.parameters(), lr=args.lr, weight_decay=0.0)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: (
            [r["query"] for r in b],
            [r["positive"] for r in b],
            [r["negative"] for r in b],
        ),
    )

    model.train()
    step = 0
    losses = []
    t0 = time.time()
    save_root = Path(args.output_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for q_texts, p_texts, n_texts in dl:
            opt.zero_grad()
            q_embs, q_weights = encode_with_weights(model, q_texts, is_query=True)
            p_embs = encode_with_weights(model, p_texts, is_query=False)
            n_embs = encode_with_weights(model, n_texts, is_query=False)

            pos_scores, neg_scores = [], []
            for qe, qw, pe, ne in zip(q_embs, q_weights, p_embs, n_embs):
                pos_scores.append(weighted_score_train(qe, qw, pe, mode=args.score_mode))
                neg_scores.append(weighted_score_train(qe, qw, ne, mode=args.score_mode))
            pos = torch.stack(pos_scores)
            neg = torch.stack(neg_scores)
            logits = torch.stack([pos, neg], dim=-1)
            labels = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.importance.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1

            # Per-step CSV log (for paper figures)
            if csv_f is not None:
                with torch.no_grad():
                    s = q_weights[0]
                    csv_f.write(f"{step},{loss.item():.6f},{s.mean():.6f},{s.std():.6f},"
                                f"{s.min():.6f},{s.max():.6f}\n")
                    csv_f.flush()

            if step % args.log_steps == 0:
                lm = sum(losses[-args.log_steps:]) / min(args.log_steps, len(losses))
                with torch.no_grad():
                    sample = q_weights[0]
                    print(
                        f"  step {step}  loss={lm:.4f}  "
                        f"head_w[mean,std,min,max]="
                        f"[{sample.mean():.3f},{sample.std():.3f},"
                        f"{sample.min():.3f},{sample.max():.3f}]  "
                        f"({step/(time.time()-t0):.1f} step/s)",
                        flush=True,
                    )

            if args.save_steps and step % args.save_steps == 0:
                model.save(str(save_root / f"checkpoint-{step}"))

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    print(f"[train-head] done — {step} steps, {(time.time()-t0)/60:.1f} min total")
    model.save(str(save_root))
    print(f"[train-head] head saved to {save_root}/importance_head")


if __name__ == "__main__":
    main()
