#!/bin/bash
# 4-GPU parallel BRIGHT eval for SmallReason-ColBERT (run from inside an interactive job).
#
# Usage:
#   bash launch_eval.sh                          # default paths
#   BASE_MODEL=... HEAD_DIR=... bash launch_eval.sh
#   GPUS=0,1,2,3 bash launch_eval.sh             # pick which 4 GPUs to use
#
# Wall time: ~10 min on 4×H100. Aggregates per-split results at the end.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

BASE_MODEL="${BASE_MODEL:-$ROOT/weights/SmallReason-ColBERT-32M}"
HEAD_DIR="${HEAD_DIR:-$ROOT/weights/SmallReason-ColBERT-32M}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/bright_scores}"
DOC_LENGTH="${DOC_LENGTH:-2048}"
DOC_CHUNK="${DOC_CHUNK:-2000}"
GPUS="${GPUS:-0,1,2,3}"
# Optional: a local BRIGHT arrow cache. Empty = download from the Hub.
BRIGHT_ROOT="${BRIGHT_ROOT:-}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -ne 4 ]; then
    echo "[launch_eval] ERROR: GPUS must list exactly 4 ids (got '$GPUS')." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$HERE/logs"

# HEAD_DIR may be a local directory or a Hugging Face repo id. Only validate the
# local case; for a repo id the loader resolves (and validates) the head itself.
if [ -d "$HEAD_DIR" ] && [ ! -f "$HEAD_DIR/importance_head/model.safetensors" ]; then
    echo "[launch_eval] ERROR: $HEAD_DIR/importance_head/model.safetensors not found."
    echo "             Run: python train/train_head.py  (then train/assemble_release.sh)"
    exit 1
fi

echo "[launch_eval] base_model=$BASE_MODEL"
echo "[launch_eval] head_dir=$HEAD_DIR"
echo "[launch_eval] output_dir=$OUTPUT_DIR"
echo "[launch_eval] gpus=$GPUS"

declare -a SPLITS_G0=("aops" "earth_science")
declare -a SPLITS_G1=("leetcode" "theoremqa_questions" "theoremqa_theorems")
declare -a SPLITS_G2=("biology" "psychology" "robotics" "pony")
declare -a SPLITS_G3=("stackoverflow" "economics" "sustainable_living")

cd "$HERE"
for gid in 0 1 2 3; do
    arr_name="SPLITS_G${gid}[@]"
    splits="${!arr_name}"
    cuda_id="${GPU_ARR[$gid]}"
    log="$HERE/logs/eval_g${gid}_$(date +%Y%m%d_%H%M%S).log"
    echo "[launch_eval] CUDA_VISIBLE_DEVICES=$cuda_id → $splits   (log: $log)"
    bright_arg=""
    [ -n "$BRIGHT_ROOT" ] && bright_arg="--bright_root $BRIGHT_ROOT"
    CUDA_VISIBLE_DEVICES=$cuda_id python -u eval_bright.py \
        --base_model "$BASE_MODEL" \
        --head_dir   "$HEAD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --document_length "$DOC_LENGTH" \
        --doc_chunk "$DOC_CHUNK" \
        $bright_arg \
        --splits $splits \
        > "$log" 2>&1 &
done

wait
echo "[launch_eval] all 4 workers done. Aggregating:"
python3 - <<EOF
import json, os
SPLITS = ['biology','aops','theoremqa_theorems','leetcode','psychology','stackoverflow',
          'earth_science','economics','sustainable_living','robotics','theoremqa_questions','pony']
QLENS = {'pony': 32}
total, n = 0.0, 0
print(f'{"split":<25} {"nDCG@10":>10}')
print('-'*38)
for s in SPLITS:
    qlen = QLENS.get(s, 256)
    p = os.path.join('$OUTPUT_DIR', f'BrightRetrieval_{s}_evaluation_scores_qlen{qlen}.json')
    if not os.path.exists(p):
        print(f'{s:<25}   missing'); continue
    d = json.load(open(p))
    print(f'{s:<25} {d["ndcg@10"]*100:>10.2f}')
    total += d["ndcg@10"]; n += 1
print('-'*38)
if n:
    print(f'{"MEAN ({}/12)".format(n):<25} {total/n*100:>10.2f}')
EOF
