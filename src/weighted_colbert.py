"""WeightedColBERT — adds a 1-layer per-query-token importance head on top of the reasoning-tuned base.

Architecture diff vs the un-headed base:

  base:     score(q, d) = sum_t  max_d (q_t · d_t)              (uniform sum)
  full:     score(q, d) = sum_t  w_t · max_d (q_t · d_t) / sum_t w_t
            where  w_t = sigmoid(W · q_t + b)                   (1-layer, ~129 params)

The base ColBERT (backbone + 3-layer Dense projection) is FROZEN; only the
importance head trains. Initialised so all weights ≈ 1 — at init, weighted
MaxSim is identical (up to a 0.99 constant) to plain MaxSim, so training can
only improve from the un-headed base.

Implemented as a wrapper around `pylate.models.ColBERT` (no pylate code is
modified). Encode + score paths are reimplemented to expose token weights.

Usage:
  from weighted_colbert import WeightedColBERT
  m = WeightedColBERT.from_base("SmallReason-ColBERT-32M").to("cuda:0")
  q_embs, q_weights = m.encode(["..."], is_query=True, return_weights=True)
  d_embs = m.encode(["..."], is_query=False)
  score = WeightedColBERT.weighted_maxsim(q_embs[0], q_weights[0], d_embs[0])
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors

# pylate is a pip dependency (see requirements.txt)
from pylate import models  # noqa: E402

IMPORTANCE_DIRNAME = "importance_head"


class ImportanceHead(nn.Module):
    """1-layer importance scorer applied to L2-normalised query token embeddings.

    Output is in (0, 1) via sigmoid. We initialise so the head outputs ~1
    everywhere at init: weight=0, bias=5  →  sigmoid(5) ≈ 0.9933. So the
    weighted MaxSim is initially equivalent (up to a 0.99 constant) to the
    plain MaxSim — no risk of degrading the un-headed base at start.
    """

    def __init__(self, in_features: int = 128, init_bias: float = 5.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        # token_embeddings: [..., in_features] L2-normalised
        # output: [...] (last dim squeezed) — scalar weight per token
        return torch.sigmoid(self.linear(token_embeddings)).squeeze(-1)


class MLPImportanceHead(nn.Module):
    """2-layer MLP importance scorer (capacity ablation).

    Architecture: Linear(D, H) → ReLU → Linear(H, 1) → sigmoid.
    Initialised so the head output ≈ sigmoid(init_bias) at step 0 — same
    invariant as ImportanceHead. We zero the *output* layer's weights so
    that whatever the hidden layer produces, the pre-sigmoid is just
    `out_bias`. `init_bias` is loaded into the output bias.
    """

    def __init__(self, in_features: int = 128, hidden: int = 128,
                 init_bias: float = 5.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=True)
        # Hidden layer: small Xavier-ish init so it has signal to learn from
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Output layer: zero weights, init_bias on bias → sigmoid(init_bias) at start
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(token_embeddings))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


class WeightedColBERT(nn.Module):
    """Wrapper around pylate.models.ColBERT that adds an importance head.

    The base ColBERT is frozen by default; only the importance head is
    trainable. Encode methods return (token_embeddings, importance_weights).
    """

    def __init__(self, base: models.ColBERT, freeze_base: bool = True,
                 head_hidden: int = 0, init_bias: float = 5.0) -> None:
        super().__init__()
        self.base = base
        # Detect projection output dim from the last Dense in modules.json
        self._dim = self._infer_dim(base)
        self._head_hidden = int(head_hidden)
        if self._head_hidden > 0:
            self.importance = MLPImportanceHead(
                in_features=self._dim, hidden=self._head_hidden, init_bias=init_bias,
            )
        else:
            self.importance = ImportanceHead(
                in_features=self._dim, init_bias=init_bias,
            )
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    @staticmethod
    def _infer_dim(base: models.ColBERT) -> int:
        # Walk modules: the last Dense's out_features is the per-token vector dim.
        last_out = None
        for _, m in base._modules.items():
            if hasattr(m, "out_features"):
                last_out = m.out_features
        if last_out is None:
            raise RuntimeError("Could not infer per-token dim from base model.")
        return int(last_out)

    # ── Loading / saving ─────────────────────────────────────────────────
    @classmethod
    def from_base(
        cls,
        base_path: str | Path,
        query_length: int = 256,
        document_length: int = 2048,
        head_path: str | Path | None = None,
        device: str | None = None,
        head_hidden: int = 0,
        require_head: bool = True,
    ) -> "WeightedColBERT":
        """Load the base ColBERT and its importance head.

        `base_path` may be a local directory or a Hugging Face repo id. In both
        cases the head is looked up at `<base>/importance_head/`.

        If no head can be found, this raises by default. That is deliberate: a
        missing head silently degrades the model to the un-headed base (19.61 vs
        21.41 on BRIGHT) with no other symptom. Pass `require_head=False` if you
        genuinely want the base with an untrained (no-op) head.
        """
        # Resolve head_path first so we can read the architecture config.
        # Try a local directory, then fall back to the Hugging Face Hub.
        if head_path is None:
            cand = Path(base_path) / IMPORTANCE_DIRNAME / "model.safetensors"
            if cand.exists():
                head_path = cand
            else:
                try:
                    from huggingface_hub import hf_hub_download

                    head_path = hf_hub_download(
                        repo_id=str(base_path),
                        filename=f"{IMPORTANCE_DIRNAME}/model.safetensors",
                    )
                except Exception as exc:  # not a hub repo, offline, or no head
                    if require_head:
                        raise RuntimeError(
                            f"No importance head found for {base_path!r}: looked for "
                            f"{cand} locally and for "
                            f"'{IMPORTANCE_DIRNAME}/model.safetensors' on the Hub "
                            f"({type(exc).__name__}: {exc}). Without the head this "
                            "model scores 19.61 on BRIGHT instead of 21.41. Pass "
                            "require_head=False if that is what you want."
                        ) from exc
        if head_path is None and require_head:
            raise RuntimeError(
                f"No importance head found for {base_path!r}. Without the head this "
                "model scores 19.61 on BRIGHT instead of 21.41. Pass "
                "require_head=False if that is what you want."
            )
        # If head_hidden not explicitly set and a head config exists, read it
        if head_hidden == 0 and head_path is not None:
            cfg = Path(head_path).parent / "config.json"
            if not cfg.exists():
                try:
                    from huggingface_hub import hf_hub_download

                    cfg = Path(hf_hub_download(
                        repo_id=str(base_path),
                        filename=f"{IMPORTANCE_DIRNAME}/config.json",
                    ))
                except Exception:
                    pass
            if cfg.exists():
                try:
                    head_hidden = int(json.loads(cfg.read_text()).get("head_hidden", 0))
                except Exception:
                    pass

        base = models.ColBERT(
            model_name_or_path=str(base_path),
            query_length=query_length,
            document_length=document_length,
        )
        m = cls(base, head_hidden=head_hidden)
        if head_path is not None:
            sd = load_safetensors(str(head_path))
            m.importance.load_state_dict(sd)
            print(f"[WeightedColBERT] loaded importance head ({'MLP h=' + str(head_hidden) if head_hidden else '1L'}) from {head_path}")
        if device:
            m = m.to(device)
        return m

    def save(self, output_dir: str | Path) -> None:
        """Save the importance head into <output_dir>/importance_head/.

        The base ColBERT is *not* re-saved — copy/symlink the base dir
        externally (see assemble_release.sh), then drop the head into
        the `importance_head/` subdir.
        """
        out = Path(output_dir) / IMPORTANCE_DIRNAME
        out.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in self.importance.state_dict().items()}
        save_safetensors(sd, str(out / "model.safetensors"))
        cfg = {
            "in_features": self._dim,
            "head_hidden": self._head_hidden,
            "init_kind": "sigmoid_bias5",
        }
        with open(out / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[WeightedColBERT] saved importance head to {out}")

    # ── Forward / encoding ───────────────────────────────────────────────
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        is_query: bool,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        return_weights: bool = False,
    ):
        """Encode and optionally return importance weights for query tokens.

        Returns:
          - if is_query and return_weights: (list[np.ndarray], list[np.ndarray])
          - else: list[np.ndarray]
        """
        embs = self.base.encode(
            sentences,
            is_query=is_query,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        if is_query and return_weights:
            ws = []
            for e in embs:
                t = torch.from_numpy(e).to(self.device(), dtype=torch.float32)
                w = self.importance(t).cpu().numpy()
                ws.append(w)
            return embs, ws
        return embs

    @staticmethod
    def weighted_maxsim(
        q_emb: np.ndarray | torch.Tensor,
        q_weights: np.ndarray | torch.Tensor,
        d_emb: np.ndarray | torch.Tensor,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted-MaxSim score for a single (q, d) pair.

        Used at EVAL time. Normalised by sum(w) so scores are comparable
        across queries with different lengths. Rank order within a query
        is invariant to the per-query rescaling.

        - q_emb     [T_q, D]
        - q_weights [T_q]   in (0, 1)
        - d_emb     [T_d, D]
        - d_mask    [T_d]   (optional — defaults to all-True)
        """
        if isinstance(q_emb, np.ndarray):
            q_emb = torch.from_numpy(q_emb)
        if isinstance(q_weights, np.ndarray):
            q_weights = torch.from_numpy(q_weights)
        if isinstance(d_emb, np.ndarray):
            d_emb = torch.from_numpy(d_emb)
        sim = q_emb @ d_emb.T  # [T_q, T_d]
        if d_mask is not None:
            sim = sim.masked_fill(~d_mask.unsqueeze(0), -1e4)
        max_per_q = sim.max(dim=-1).values  # [T_q]
        w = q_weights / q_weights.sum().clamp(min=1e-6)  # normalise → sums to 1
        return (max_per_q * w).sum()
