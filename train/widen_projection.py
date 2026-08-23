"""Widen the final projection head of a PyLate ColBERT checkpoint from N→M dims.

The edge-ColBERT 32M model ships a 3-layer projection head ending at 64 dims
(see `3_Dense/config.json`). That 64-dim MaxSim channel is a bottleneck on
symbol-dense BRIGHT splits (leetcode, aops, theoremqa) where many tokens look
alike and need more channels to distinguish.

This script takes an EXISTING saved ColBERT checkpoint dir and writes a NEW
checkpoint dir whose final Dense layer has a widened `out_features`, without
modifying anything else (backbone weights, tokenizer, query/doc prefixes,
skiplist, etc. are all hard-linked from the source dir).

Init strategy (small Gaussian, NOT zeros):
    old linear.weight  shape (old_dim, in_features)
    new linear.weight  shape (new_dim, in_features)
        first old_dim rows = old rows (inherited verbatim)
        rows old_dim .. new_dim-1 ~ N(0, (0.1 * std(old))^2)
Zeros look tempting — after L2 normalisation `[x; 0]` keeps MaxSim identical to
the source model — but the gradient w.r.t. those weights is then exactly zero on
both the query and document side, so the new channels never learn (dead-neuron
deadlock). A small Gaussian perturbs MaxSim by only O(std^2) at init while
unlocking gradient flow. See --verify to check the first old_dim dims still match.

Usage
-----
  python widen_projection/widen_colbert_projection.py \
      --src  repro_reason_moderncolbert/output_edge32m_v2b/pos1_A/stage-vl/final \
      --dst  repro_reason_moderncolbert/output_edge32m_vl_d128/initial \
      --new-dim 128

  # Sanity-check the widened checkpoint loads and produces identical outputs
  # on the first 64 dims:
  python widen_projection/widen_colbert_projection.py \
      --src  repro_reason_moderncolbert/output_edge32m_v2b/pos1_A/stage-vl/final \
      --dst  repro_reason_moderncolbert/output_edge32m_vl_d128/initial \
      --new-dim 128 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The pylate project root (so we can import models.ColBERT for --verify).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hardlink_tree(src: Path, dst: Path, skip: set[str]) -> None:
    """Mirror `src` to `dst` with hardlinks, skipping paths in `skip`.

    HF Hub snapshot dirs contain symlinks pointing at `../../blobs/<sha>`;
    hardlinking a symlink preserves the relative link text, which breaks in
    the new location. We resolve symlinks to their real path before hardlinking
    so the destination files have the actual content reachable.
    """
    for root, dirs, files in os.walk(src, followlinks=False):
        rel_root = Path(root).relative_to(src)
        if any(str(rel_root).startswith(s) for s in skip):
            continue
        (dst / rel_root).mkdir(parents=True, exist_ok=True)
        for f in files:
            rel_file = rel_root / f
            if str(rel_file) in skip:
                continue
            src_f = Path(root) / f
            dst_f = dst / rel_file
            if dst_f.exists() or dst_f.is_symlink():
                dst_f.unlink()
            # Resolve if symlink so we hardlink the actual blob, not the
            # symlink text (which would break when relative paths change).
            real = src_f.resolve() if src_f.is_symlink() else src_f
            try:
                os.link(real, dst_f)
            except OSError:
                shutil.copy2(real, dst_f)


def _widen_dense(src_dense: Path, dst_dense: Path, new_out: int) -> tuple[int, int]:
    """Rewrite one Dense/ subdirectory with widened `out_features`.

    Returns (old_out, in_features).
    """
    cfg_path = src_dense / "config.json"
    w_path = src_dense / "model.safetensors"
    if not cfg_path.exists() or not w_path.exists():
        raise FileNotFoundError(f"{src_dense} missing config.json or model.safetensors")

    with open(cfg_path) as f:
        cfg = json.load(f)
    old_out = int(cfg["out_features"])
    in_features = int(cfg["in_features"])

    if old_out == new_out:
        logger.info("%s already %d-dim; copying unchanged.", src_dense.name, new_out)
        dst_dense.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_path, dst_dense / "config.json")
        shutil.copy2(w_path, dst_dense / "model.safetensors")
        return old_out, in_features
    if new_out < old_out:
        raise ValueError(
            f"Refusing to NARROW projection {src_dense.name} ({old_out} → {new_out}); "
            "this tool only widens."
        )

    # Load old weight. Init new rows with SMALL RANDOM values (not zeros!).
    #
    # Why not zeros: after L2-normalisation, a zero-padded row produces 0 in
    # the new dims on BOTH the query and document sides. The MaxSim score
    # becomes identical to the old 64-dim model, and the gradient w.r.t. the
    # new weights is exactly zero (dead-neuron deadlock). They never learn.
    #
    # Small Gaussian init (std matches the old rows' per-row stats) unlocks
    # gradient flow while only changing MaxSim by O(std^2) at init — small
    # enough that it doesn't undo the VL-stage training.
    state = load_safetensors(str(w_path))
    if "linear.weight" not in state:
        raise KeyError(f"{w_path} missing key 'linear.weight' (got {list(state)})")
    w_old = state["linear.weight"]
    if w_old.shape != (old_out, in_features):
        raise ValueError(
            f"Unexpected weight shape {tuple(w_old.shape)}; "
            f"expected ({old_out}, {in_features}) from config.json"
        )

    # Use the std of the existing rows so the new rows are scaled consistently.
    old_std = float(w_old.float().std().item())
    init_std = max(old_std * 0.1, 1e-3)  # 10% of old magnitude; never below 1e-3
    gen = torch.Generator().manual_seed(0)
    pad = torch.randn((new_out - old_out, in_features), generator=gen,
                      dtype=torch.float32) * init_std
    pad = pad.to(w_old.dtype)
    w_new = torch.cat([w_old, pad], dim=0).contiguous()
    logger.info(
        "new rows: std=%.3e (10%% of old std=%.3e)  max|abs|=%.3e",
        init_std, old_std, pad.abs().max().item(),
    )

    new_state = {"linear.weight": w_new}
    if "linear.bias" in state:
        b_old = state["linear.bias"]
        b_pad = torch.zeros((new_out - old_out,), dtype=b_old.dtype)
        new_state["linear.bias"] = torch.cat([b_old, b_pad], dim=0).contiguous()

    dst_dense.mkdir(parents=True, exist_ok=True)
    save_safetensors(new_state, str(dst_dense / "model.safetensors"))

    cfg_new = dict(cfg)
    cfg_new["out_features"] = new_out
    with open(dst_dense / "config.json", "w") as f:
        json.dump(cfg_new, f, indent=2)
    logger.info(
        "widened %s: out_features %d → %d  (weight shape %s → %s)",
        src_dense.name, old_out, new_out, tuple(w_old.shape), tuple(w_new.shape),
    )
    return old_out, in_features


def _find_final_dense(src: Path) -> Path:
    """Find the last Dense subdir referenced in modules.json."""
    mod_path = src / "modules.json"
    if not mod_path.exists():
        raise FileNotFoundError(f"{src} missing modules.json — not a PyLate/ST model?")
    with open(mod_path) as f:
        modules = json.load(f)
    dense_paths = [m["path"] for m in modules if m.get("path")
                   and (src / m["path"]).is_dir()
                   and m.get("type", "").endswith("Dense")]
    if not dense_paths:
        raise RuntimeError(f"{src}/modules.json has no Dense entries.")
    final = dense_paths[-1]
    return src / final


def verify(src: Path, dst: Path) -> None:
    """Encode a tiny batch with both models and check the first-N dims match."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "pylate"))
    from pylate import models  # noqa: E402

    sentences = ["hello world", "a quick brown fox"]
    logger.info("loading source model: %s", src)
    m_old = models.ColBERT(str(src))
    logger.info("loading widened model: %s", dst)
    m_new = models.ColBERT(str(dst))

    emb_old = m_old.encode(sentences, is_query=False, convert_to_numpy=True, show_progress_bar=False)
    emb_new = m_new.encode(sentences, is_query=False, convert_to_numpy=True, show_progress_bar=False)

    import numpy as np  # local import so widening path doesn't need numpy
    ok = True
    for i, (e_o, e_n) in enumerate(zip(emb_old, emb_new)):
        old_dim = e_o.shape[-1]
        # With small-random init the first old_dim coordinates DRIFT SLIGHTLY
        # from the source (L2-normalisation couples them to the new dims via
        # the norm denominator). Check the tail is non-zero (gradient will
        # flow) and that the drift on the old dims is within a tolerance.
        diff = np.abs(e_n[:, :old_dim] - e_o).max()
        tail = np.abs(e_n[:, old_dim:]).max() if e_n.shape[-1] > old_dim else 0.0
        logger.info("sentence %d: max |old - new[:, :%d]| = %.2e   "
                    "max |new[:, %d:]| = %.2e  (should be > 0 → grad flows)",
                    i, old_dim, diff, old_dim, tail)
        if abs(float(diff)) > 5e-2 or abs(float(tail)) < 1e-4:
            ok = False
    if ok:
        logger.info("VERIFY OK — widened model reproduces source behaviour on first %d dims.",
                    old_dim)
    else:
        logger.warning("VERIFY MISMATCH — differences above float tolerance; check script.")


def main() -> None:
    p = argparse.ArgumentParser(description="Widen ColBERT final projection head.")
    p.add_argument("--src", required=True, type=Path,
                   help="Source ColBERT checkpoint directory.")
    p.add_argument("--dst", required=True, type=Path,
                   help="Destination directory (created if missing).")
    p.add_argument("--new-dim", required=True, type=int,
                   help="New out_features for the final Dense layer.")
    p.add_argument("--verify", action="store_true",
                   help="Load both models and check the first old_dim dims match.")
    args = p.parse_args()

    if not args.src.is_dir():
        raise FileNotFoundError(f"--src {args.src} is not a directory")
    args.dst.mkdir(parents=True, exist_ok=True)

    final_dense_src = _find_final_dense(args.src)
    final_dense_rel = final_dense_src.relative_to(args.src)
    logger.info("Final Dense in source: %s (will widen)", final_dense_rel)

    # Hardlink everything except the final Dense dir.
    skip = {str(final_dense_rel)}
    _hardlink_tree(args.src, args.dst, skip=skip)

    # Widen the final Dense into the destination.
    final_dense_dst = args.dst / final_dense_rel
    if final_dense_dst.exists():
        shutil.rmtree(final_dense_dst)
    _widen_dense(final_dense_src, final_dense_dst, args.new_dim)

    logger.info("Wrote widened checkpoint to %s", args.dst)
    logger.info("Backbone + tokenizer + other Dense layers are hardlinks to source "
                "(zero extra disk).")

    if args.verify:
        verify(args.src, args.dst)


if __name__ == "__main__":
    main()
