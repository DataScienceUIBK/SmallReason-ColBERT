#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ucb-s2-bgehn
#SBATCH --time=18:00:00
#SBATCH --nodes=2
#
# Stage 2 (final): BGE-HN polish on top of stage-vl.
# Cluster auto-applies "full gpu node(s)" → 4 tasks/node, 1 GPU each.

set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT"

# Activate your environment. Either export CONDA_INIT+ENV_NAME, or have the
# environment already active before sbatch.
if [ -n "${CONDA_INIT:-}" ] && [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
    conda activate "${ENV_NAME:-smallreason}"
fi

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NCCL_IFNAME="${NCCL_IFNAME:-eno2np0}"
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD" \
    bash -c "ip -4 -o addr show $NCCL_IFNAME | awk '{print \$4}' | cut -d/ -f1")
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

export NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600

export PYTHONNOUSERSITE=1
# pylate is a pip dependency (see requirements.txt). Set PYLATE_SRC only if you
# want to shadow it with a local checkout.
export PYTHONPATH="$PROJECT${PYLATE_SRC:+:$PYLATE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

BASE_MODEL="${BASE_MODEL:-$PROJECT/weights/_intermediate/stage-vl/final}"
DATA_PATH="${DATA_PATH:-$PROJECT/data_processed/bge_hn_merged_processed}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/weights/_intermediate}"
RUN_NAME="${RUN_NAME:-stage-bge-hn}"

EPOCHS=1
LR=5e-6
BATCH_SIZE=32
MINI_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2
export MAX_GRAD_NORM=10000
QUERY_LENGTH=256
DOCUMENT_LENGTH=2048
ATTN=flash_attention_2
DTYPE=bfloat16

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: stage-vl final missing at $BASE_MODEL" >&2; exit 1
fi
if [ ! -f "$DATA_PATH/dataset_info.json" ]; then
    echo "ERROR: BGE-HN dataset missing at $DATA_PATH" >&2; exit 1
fi

echo "============================================================"
echo "Stage 2 (BGE-HN polish)"
echo "  base   : $BASE_MODEL"
echo "  data   : $DATA_PATH"
echo "  output : $OUTPUT_DIR/$RUN_NAME/final"
echo "  master : $MASTER_ADDR:$MASTER_PORT  ($NCCL_IFNAME)"
echo "  ntasks : $SLURM_NTASKS  (4 per node × $SLURM_JOB_NUM_NODES nodes)"
echo "============================================================"

srun --gpu-bind=none bash -c '
    export RANK="$SLURM_PROCID"
    export LOCAL_RANK="$SLURM_LOCALID"
    export WORLD_SIZE="$SLURM_NTASKS"
    echo "[task] RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE host=$(hostname -s)  CVD=${CUDA_VISIBLE_DEVICES:-?}"
    python -u '"$PROJECT"'/train/launch_multinode.py \
        --base_model '"$BASE_MODEL"' \
        --train_data_path '"$DATA_PATH"' \
        --epochs '"$EPOCHS"' --lr '"$LR"' \
        --batch_size '"$BATCH_SIZE"' \
        --mini_batch_size '"$MINI_BATCH_SIZE"' \
        --gradient_accumulation_steps '"$GRADIENT_ACCUMULATION_STEPS"' \
        --query_length '"$QUERY_LENGTH"' \
        --document_length '"$DOCUMENT_LENGTH"' \
        --attn_implementation '"$ATTN"' \
        --torch_dtype '"$DTYPE"' \
        --output_dir '"$OUTPUT_DIR"' \
        --run_name '"$RUN_NAME"
echo "[stage2] Done. Output: $OUTPUT_DIR/$RUN_NAME/final"
