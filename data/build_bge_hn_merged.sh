#!/bin/bash
# Build the BGE-HN merged training dataset that base stage-2 and head training consume.
#
# Materializes two sources, then concatenates + shuffles them:
#   1. ReasonIR-HQ (from reasonir/reasonir-data on HF) → ~2M (q, pos, neg) triples
#   2. BGE-Reasoner (from hanhainebula/bge-reasoner-data on HF) → ~720K triples
# Merged total: ~2.7M rows. ~13 GB on disk.
#
# Usage (from the repo root):
#   bash data/build_bge_hn_merged.sh
#
# Output:
#   data_processed/bge_hn_merged_processed/   (HF dataset dir, columns: query/positive/negative)
#
# Time: ~30 min on a fast NFS (most of it is HF download + materialization). Safe to re-run; idempotent.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT_ROOT="${OUT_ROOT:-$ROOT/data_processed}"

mkdir -p "$OUT_ROOT"

echo "[build] Materializing ReasonIR-HQ → $OUT_ROOT/reasonir_hq_processed"
python "$HERE/materialize_reasonir.py" \
    --config hq \
    --output-dir "$OUT_ROOT/reasonir_hq_processed"

echo "[build] Materializing BGE-Reasoner → $OUT_ROOT/bge_reasoner_processed"
python "$HERE/materialize_bge_reasoner.py" \
    --output-dir "$OUT_ROOT/bge_reasoner_processed" \
    --max-pos-per-query 2 --max-neg-per-query 4

echo "[build] Merging → $OUT_ROOT/bge_hn_merged_processed"
python "$HERE/merge_datasets.py" \
    --inputs "$OUT_ROOT/bge_reasoner_processed" "$OUT_ROOT/reasonir_hq_processed" \
    --output-dir "$OUT_ROOT/bge_hn_merged_processed"

echo "[build] Done."
echo "[build] Merged dataset: $OUT_ROOT/bge_hn_merged_processed"
echo "[build] Use as --train_data for train/train_head.py and as DATA_PATH for stage 2."
