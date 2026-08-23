# `data/` — training data preparation

The training data for the base (stage 2) and the importance head is a **merged
HF dataset of `(query, positive, negative)` triples** with two upstream sources:

| Source | HF dataset | What we use |
|---|---|---|
| ReasonIR-HQ | [`reasonir/reasonir-data`](https://huggingface.co/datasets/reasonir/reasonir-data) `hq` split | ~2 M long-reasoning triples |
| BGE-Reasoner | [`hanhainebula/bge-reasoner-data`](https://huggingface.co/datasets/hanhainebula/bge-reasoner-data) all 12 BRIGHT domains | ~720 K BRIGHT-aligned triples |

Stage 1 (VL warmup) of base training also needs a separate ReasonIR-VL dataset:

| Source | HF dataset | What we use |
|---|---|---|
| ReasonIR-VL | [`reasonir/reasonir-data`](https://huggingface.co/datasets/reasonir/reasonir-data) `vl` split | ~245 K varied-length triples |

## Build the merged BGE-HN dataset

```bash
# From the repo root:
bash data/build_bge_hn_merged.sh
```

That single script:
1. materializes ReasonIR-HQ → `data_processed/reasonir_hq_processed/`
2. materializes BGE-Reasoner → `data_processed/bge_reasoner_processed/`
3. concatenates + shuffles them → `data_processed/bge_hn_merged_processed/`

End size: **~2.7 M rows, ~13 GB on disk.** Idempotent — re-running skips
already-materialized parts.

## Build the VL dataset (only needed for the full full base-training pipeline)

```bash
python data/materialize_reasonir.py \
    --config vl \
    --output-dir data_processed/reasonir_vl_processed
```

Size: ~245 K rows.

## File reference

| File | Purpose |
|---|---|
| `prepare_data.py` | Loaders for ReasonIR-HQ, ReasonIR-VL, and BRIGHT-eval data |
| `materialize_reasonir.py` | CLI to dump HQ or VL to disk as a flat `{query,positive,negative}` dataset |
| `materialize_bge_reasoner.py` | CLI to dump BGE-Reasoner data (12 BRIGHT domains, ~720 K triples) |
| `merge_datasets.py` | CLI to concatenate + shuffle multiple processed datasets |
| `build_bge_hn_merged.sh` | One-command wrapper that runs the three steps above |
| `__init__.py` | Re-exports the loaders so `from data import load_reasonir_hq_dataset` works |

## Caching

All HF downloads land in `${HF_HOME:-~/.cache/huggingface}`. If you want to
share the cache between users on a cluster filesystem, point `HF_HOME` at
a shared path before running.
