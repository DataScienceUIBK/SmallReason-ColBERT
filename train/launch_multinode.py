"""Launcher that forwards to LaRLI/train_baseline.py, with a multi-node fix.

Background
----------
`sentence_transformers.util.get_device_name()` contains a latent bug that
affects multi-node torchrun runs:

    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()  # actually the GLOBAL rank
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return f"cuda:{local_rank}"

`torch.distributed.get_rank()` returns the **global** rank (0..world_size-1),
not the local rank within the node. On node 1 of a 2-node 4-GPU run, ranks
4-7 end up trying to bind to `cuda:4..cuda:7`, but only `cuda:0..cuda:3` are
visible per node → `RuntimeError: CUDA error: invalid device ordinal`.

This launcher:
  1. Pins the current process to `cuda:LOCAL_RANK` from torchrun.
  2. Monkey-patches `sentence_transformers.util.get_device_name` (and the
     local alias in `SentenceTransformer`) to return the correct local rank
     instead of the global one.
  3. Hands off to `LaRLI.train_baseline.main()` unchanged.

Nothing in LaRLI/ or pylate/ or sentence_transformers/ is modified.

Usage (from your sbatch, inside srun + bash -c):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$SLURM_PROCID ... \
        widen_projection/launch_train_multinode.py \
        --base_model ...  --train_data_path ...  (same args as train_baseline.py)
"""

from __future__ import annotations

import os
import sys

# ── 1. Resolve local rank from torchrun env ────────────────────────────
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
_GLOBAL_RANK = int(os.environ.get("RANK", 0))
_WORLD = int(os.environ.get("WORLD_SIZE", 1))
print(f"[launch] RANK={_GLOBAL_RANK} LOCAL_RANK={_LOCAL_RANK} WORLD={_WORLD}", flush=True)

import torch  # noqa: E402

if torch.cuda.is_available():
    print(f"[launch] cuda.device_count={torch.cuda.device_count()} (before pin)", flush=True)
    if _LOCAL_RANK < torch.cuda.device_count():
        torch.cuda.set_device(_LOCAL_RANK)
        print(f"[launch] torch.cuda.set_device({_LOCAL_RANK}); current={torch.cuda.current_device()}", flush=True)
    else:
        print(f"[launch] WARNING: LOCAL_RANK={_LOCAL_RANK} >= device_count={torch.cuda.device_count()}", flush=True)

# ── 2. Monkey-patch ST's get_device_name before anything imports ST ───
import sentence_transformers.util as _st_util  # noqa: E402


def _patched_get_device_name() -> str:
    """Return `cuda:LOCAL_RANK` under torchrun, else fall back to ST's own logic."""
    if torch.cuda.is_available():
        dev = f"cuda:{_LOCAL_RANK}"
        print(f"[launch] get_device_name() -> {dev} (rank {_GLOBAL_RANK})", flush=True)
        return dev
    return _ORIG_GET_DEVICE_NAME()


_ORIG_GET_DEVICE_NAME = _st_util.get_device_name
_st_util.get_device_name = _patched_get_device_name
print(f"[launch] patched sentence_transformers.util.get_device_name", flush=True)

# SentenceTransformer imports `get_device_name` by name at module load time,
# so we also need to patch the already-bound reference inside that module.
# IMPORTANT: `sentence_transformers/__init__.py` does
#   `from .SentenceTransformer import SentenceTransformer`
# which shadows the submodule name with the class. `import
# sentence_transformers.SentenceTransformer` therefore returns the CLASS, not
# the module. The real module is in `sys.modules`.
_ST_MODULE = sys.modules.get("sentence_transformers.SentenceTransformer")
if _ST_MODULE is None:
    # Force-import the submodule so it lands in sys.modules.
    import importlib
    _ST_MODULE = importlib.import_module("sentence_transformers.SentenceTransformer")

if hasattr(_ST_MODULE, "get_device_name"):
    _ST_MODULE.get_device_name = _patched_get_device_name
    print(f"[launch] patched sentence_transformers.SentenceTransformer (module).get_device_name",
          flush=True)
else:
    print(f"[launch] WARNING: module has no get_device_name; attrs with 'device': "
          f"{[a for a in dir(_ST_MODULE) if 'device' in a.lower()]}", flush=True)

# ── 3. Override max_grad_norm if requested via env var ────────────────
# Widened projection heads start with zero-init weights in the new dims.
# Those zeros produce pathological gradients at bootstrap (norm ~30-50k),
# which HF Trainer's default max_grad_norm=1.0 clips away to near-nothing.
# Set MAX_GRAD_NORM=100 (or higher / "inf") in the sbatch to let the new
# dims actually learn; the original 64-dim weights are already trained so
# they won't blow up.
_MAX_GRAD_NORM_OVERRIDE = os.environ.get("MAX_GRAD_NORM")
if _MAX_GRAD_NORM_OVERRIDE:
    import sentence_transformers.training_args as _st_args

    _orig_ta_init = _st_args.SentenceTransformerTrainingArguments.__post_init__

    def _patched_post_init(self):
        try:
            val = float(_MAX_GRAD_NORM_OVERRIDE)
        except ValueError:
            val = float("inf")
        self.max_grad_norm = val
        print(f"[launch] forced max_grad_norm = {val} (was default 1.0)", flush=True)
        _orig_ta_init(self)

    _st_args.SentenceTransformerTrainingArguments.__post_init__ = _patched_post_init

# ── 4. Make the project importable and hand off ───────────────────────
# Repo root is the parent of train/. pylate is a pip dependency.
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_PROJECT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_base import main  # noqa: E402

if __name__ == "__main__":
    main()
