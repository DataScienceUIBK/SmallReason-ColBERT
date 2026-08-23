#!/bin/bash
# Stage 2: assemble the model dir containing importance_head/.
#
# After train.py runs, you have <output_dir>/importance_head/{model.safetensors,config.json}.
# To make a fully-loadable release dir we hardlink (or copy) the base model
# files alongside the importance head.
#
# Usage:
#   bash assemble_release.sh <base-dir> <out-dir>
#
# Example:
#   bash assemble_release.sh \
#        ../SmallReason-ColBERT-32M \
#        ../SmallReason-ColBERT-32M
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <base-dir> <out-dir>"
    exit 1
fi

V01="$1"
V02="$2"

if [ ! -d "$V01" ]; then
    echo "[assemble] ERROR: base dir not found: $V01"
    exit 1
fi
if [ ! -f "$V02/importance_head/model.safetensors" ]; then
    echo "[assemble] ERROR: importance head not found at $V02/importance_head/."
    echo "          Run train.py first."
    exit 1
fi

echo "[assemble] base dir: $V01"
echo "[assemble] target:   $V02"

# Top-level files: link everything from the base dir except results/training/eval dirs
# and the importance_head we already have.
for entry in "$V01"/*; do
    name=$(basename "$entry")
    case "$name" in
        importance_head|results|training|evaluation|.git*|README.md)
            continue
            ;;
    esac
    target="$V02/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        # Already exists as real file/dir — skip to be safe
        echo "[assemble] skip $name (already exists)"
        continue
    fi
    rm -f "$target"
    if [ -d "$entry" ]; then
        mkdir -p "$target"
        # Recursively hardlink subdir contents (keeps small Dense weight files
        # de-duplicated on disk)
        for sub in "$entry"/*; do
            sname=$(basename "$sub")
            ln -f "$sub" "$target/$sname" 2>/dev/null || cp -r "$sub" "$target/$sname"
        done
    else
        ln -f "$entry" "$target" 2>/dev/null || cp "$entry" "$target"
    fi
    echo "[assemble] linked $name"
done

echo
echo "[assemble] done. model dir containing importance_head/ contents:"
ls -la "$V02"
echo
echo "[assemble] importance head:"
ls -la "$V02/importance_head"
