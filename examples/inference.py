"""Minimal inference example for SmallReason-ColBERT.

Scores a handful of documents against one reasoning-style query using the
weighted MaxSim of the importance head.

    python examples/inference.py
    python examples/inference.py --model /path/to/local/dir --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from weighted_colbert import WeightedColBERT  # noqa: E402

DEFAULT_MODEL = "DataScience-UIBK/SmallReason-ColBERT-32M"

QUERY = (
    "We know Earth has three Hadley cells per hemisphere, but gas giants such as "
    "Jupiter appear to have many more. What factors determine how many circulation "
    "cells a planet's atmosphere has?"
)

DOCUMENTS = [
    "The number of meridional circulation cells scales with a planet's rotation rate "
    "and the depth of its atmosphere: faster rotation narrows the Rossby deformation "
    "radius, so the same pole-to-equator temperature gradient is resolved by more, "
    "narrower cells.",
    "Hadley circulation transports heat from the equator poleward. On Earth the cell "
    "terminates near 30 degrees latitude, where descending air produces the "
    "subtropical deserts.",
    "To cook dried pasta well, use plenty of salted boiling water and stir during the "
    "first minute so the pieces do not stick together.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF repo id or local dir containing importance_head/")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: CPU)")
    ap.add_argument("--query_length", type=int, default=256)
    ap.add_argument("--document_length", type=int, default=2048)
    args = ap.parse_args()

    # Raises if the importance head is missing, rather than silently degrading
    # to the un-headed base.
    model = WeightedColBERT.from_base(
        args.model,
        query_length=args.query_length,
        document_length=args.document_length,
        device=args.device,
    )
    model.eval()

    q_embs, q_weights = model.encode([QUERY], is_query=True, return_weights=True)
    d_embs = model.encode(DOCUMENTS, is_query=False)

    scored = [
        (float(WeightedColBERT.weighted_maxsim(q_embs[0], q_weights[0], d)), i)
        for i, d in enumerate(d_embs)
    ]
    scored.sort(reverse=True)

    print(f"\nQuery: {QUERY}\n")
    print(f"{'rank':<6}{'score':>9}  document")
    print("-" * 78)
    for rank, (score, i) in enumerate(scored, 1):
        print(f"{rank:<6}{score:>9.4f}  {DOCUMENTS[i][:60]}...")

    w = q_weights[0]
    print(f"\nquery tokens: {len(w)}   "
          f"gate mean={w.mean():.3f} std={w.std():.3f} "
          f"min={w.min():.3f} max={w.max():.3f}")


if __name__ == "__main__":
    main()
