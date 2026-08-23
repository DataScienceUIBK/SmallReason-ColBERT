<div align="center">

# 🔍 SmallReason-ColBERT

**A 32M late-interaction retriever for reasoning-intensive retrieval**

*A reasoning-tuned ColBERT base plus a query-side importance head*

[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SmallReason--ColBERT--32M-FFD21E)](https://huggingface.co/DataScience-UIBK/SmallReason-ColBERT-32M)
[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026-B31B1B)](https://2026.emnlp.org/)
[![Code License](https://img.shields.io/badge/Code-Apache%202.0-3DA639)](LICENSE)
[![Weights License](https://img.shields.io/badge/Weights-CC--BY--NC--4.0-EF9421)](https://creativecommons.org/licenses/by-nc/4.0/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyLate](https://img.shields.io/badge/PyLate-1.5.0-4B8BBE)](https://github.com/lightonai/pylate)
[![Params](https://img.shields.io/badge/Params-32M-6E7681)](#)

[Install](#-install) &nbsp;·&nbsp; [Inference](#-inference) &nbsp;·&nbsp; [How it works](#-how-it-works) &nbsp;·&nbsp; [Reproduce](#-reproducing-the-model) &nbsp;·&nbsp; [Cite](#-citation)

</div>

---

## 📦 Install

```bash
pip install -r requirements.txt
```

The released model was built and verified with **pylate 1.5.0**. No patched or
vendored pylate is needed.

## 🚀 Inference

```bash
python examples/inference.py
```

Or directly:

```python
import sys; sys.path.insert(0, "src")
from weighted_colbert import WeightedColBERT

model = WeightedColBERT.from_base(
    "DataScience-UIBK/SmallReason-ColBERT-32M",   # HF id or local dir
    query_length=256, document_length=2048, device="cuda:0",
)

q_embs, q_weights = model.encode(["your reasoning query"], is_query=True,
                                 return_weights=True)
d_embs = model.encode(["doc a", "doc b"], is_query=False)

for d in d_embs:
    print(float(WeightedColBERT.weighted_maxsim(q_embs[0], q_weights[0], d)))
```

### ⚠️ One thing to know about loading

The importance head lives in `importance_head/` and is **not** part of the model's
`modules.json`. Loading the repo with plain PyLate gives you the un-headed base — no
error, just 1.8 nDCG@10 worse:

| Load path | Result | BRIGHT mean |
|---|---|---:|
| `pylate.models.ColBERT(...)` | base only, head ignored | 19.61 |
| `WeightedColBERT.from_base(...)` | full model | **21.41** |

`from_base` **raises** if it cannot find a head, so you cannot hit this silently
through this repo. Pass `require_head=False` if you deliberately want the base.

---

## 🧠 How it works

```
mxbai-edge-colbert-v0-32m  (public 32M ColBERT, 64-d projection)
   │
   ├─ 1. widen projection 64 → 128            train/widen_projection.py
   ├─ 2. warmup on ReasonIR-VL                train/stage1_warmup.sbatch
   ├─ 3. polish on ReasonIR-HQ + BGE-Reasoner train/stage2_polish.sbatch
   │        └─► reasoning-tuned base                          19.61
   └─ 4. freeze base, train 129-param head    train/train_head.py
            └─► SmallReason-ColBERT                           21.41
```

**The trick that makes the head work.** It is trained against the **un-normalised**
weighted score `Σ wₜ · maxⱼ(Qₜ·Dⱼ)` but evaluated against the **length-normalised**
one `Σ wₜ·maxⱼ(...) / Σ wₜ`. Train against the normalised score instead and the
cross-entropy gradient collapses, the loss stalls at `ln 2`, the gates never leave
their initialisation, and BRIGHT drops **3.59** nDCG@10. The `--score_mode norm` flag
on `train_head.py` reproduces that failure.

The head is initialised `W = 0`, `b = 5`, so every gate starts at `σ(5) ≈ 0.993` and
the head is a no-op against the frozen base at step zero.

---

## 🔬 Reproducing the model

### 0️⃣ Data

```bash
bash data/build_bge_hn_merged.sh          # ReasonIR-HQ + BGE-Reasoner, ~2.7M rows, ~13 GB
python data/materialize_reasonir.py --config vl \
       --output-dir data_processed/reasonir_vl_processed   # ~245K rows
```

### 1️⃣–3️⃣ Base (8× H100, ~24 h)

```bash
python train/widen_projection.py \
    --src weights/mxbai-edge-colbert-v0-32m \
    --dst weights/base-32m-widened --new-dim 128 --verify

sbatch train/stage1_warmup.sbatch      # ReasonIR-VL warmup,  lr 1e-5, ~8 h
sbatch train/stage2_polish.sbatch      # hard-negative polish, lr 5e-6, ~16 h
```

Both stages use PyLate's `CachedContrastive` loss over in-batch negatives. Override
`BASE_MODEL`, `DATA_PATH`, `OUTPUT_DIR`, `CONDA_INIT`, `ENV_NAME` by environment
variable; the sbatch files assume 2 nodes × 4 GPUs.

### 4️⃣ Head (1 GPU, ~12 min)

```bash
python train/train_head.py \
    --base_model weights/smallreason-base \
    --train_data data_processed/bge_hn_merged_processed \
    --output_dir weights/SmallReason-ColBERT-32M \
    --max_steps 3000 --batch_size 16 --lr 5e-4

bash train/assemble_release.sh weights/smallreason-base weights/SmallReason-ColBERT-32M
```

`assemble_release.sh` places the base files alongside `importance_head/` so the
result is a single loadable directory.

### 📊 Evaluate

```bash
# BRIGHT, with the head (4 GPUs, ~10 min)
BASE_MODEL=weights/SmallReason-ColBERT-32M HEAD_DIR=weights/SmallReason-ColBERT-32M \
  bash eval/launch_eval.sh

# BRIGHT, plain MaxSim (the un-headed base row)
python eval/eval_bright_plain.py --model weights/smallreason-base \
  --output_dir bright_scores_base --splits biology earth_science ...

# NanoBEIR sanity
python eval/eval_nanobeir.py --weighted \
  --base_model weights/SmallReason-ColBERT-32M \
  --head_dir   weights/SmallReason-ColBERT-32M
```

Protocol: brute-force MaxSim, `query_length=256` (Pony: 32), `document_length=2048`.

---

## 🗂️ Layout

```
src/weighted_colbert.py     model + importance head + weighted MaxSim (the loader)
train/widen_projection.py   64 → 128-d projection widening
train/train_base.py         base training entrypoint (CachedContrastive)
train/launch_multinode.py   Slurm/NCCL launcher shim around train_base.py
train/stage{1,2}_*.sbatch   warmup / polish job scripts
train/train_head.py         importance-head training (single GPU)
train/assemble_release.sh   base + head → one loadable dir
data/                       ReasonIR-VL / ReasonIR-HQ / BGE-Reasoner preparation
eval/                       BRIGHT (headed + plain) and NanoBEIR evaluation
examples/inference.py       minimal scoring demo
```

## 📄 License

Code: **Apache-2.0** (see `LICENSE`).
Model weights: **CC-BY-NC-4.0**, inherited from the ReasonIR and BGE-Reasoner
training data. The upstream base model is Apache-2.0.

## 📝 Citation

```bibtex
@inproceedings{smallreason-colbert,
  title     = {SmallReason-ColBERT: An Ultra-Small Late-Interaction Retriever
               for Reasoning Intensive Retrieval},
  author    = {Abdallah, Abdelrahman and Ali, Mohammed and Jatowt, Adam},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

Thanks to Antoine Chaffin (LightOn, Reason-ModernColBERT) for flagging the upstream
`2_Dense/use_residual` config bug in `mxbai-edge-colbert-v0-32m`.
