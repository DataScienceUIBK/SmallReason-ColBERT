from __future__ import annotations

import inspect
import logging


logger = logging.getLogger(__name__)


def apply_flash_attn_modernbert_compat() -> None:
    """Patch flash-attn RotaryEmbedding for Transformers 4.48 ModernBERT.

    `transformers==4.48.x` ModernBERT passes `pos_idx_in_fp32=` into
    `flash_attn.layers.rotary.RotaryEmbedding`. Some newer flash-attn releases
    removed that public constructor kwarg while preserving the internal fp32
    position-index behavior. This shim accepts the old kwarg so ModernBERT can
    still initialize under the reproduction stack.
    """

    try:
        from flash_attn.layers.rotary import RotaryEmbedding
    except ModuleNotFoundError:
        return

    init_signature = inspect.signature(RotaryEmbedding.__init__)
    if "pos_idx_in_fp32" in init_signature.parameters:
        return
    if getattr(RotaryEmbedding.__init__, "_modernbert_compat", False):
        return

    original_init = RotaryEmbedding.__init__

    def compat_init(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        device=None,
        pos_idx_in_fp32=None,
        **kwargs,
    ):
        return original_init(
            self,
            dim=dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            device=device,
        )

    compat_init._modernbert_compat = True  # type: ignore[attr-defined]
    RotaryEmbedding.__init__ = compat_init
    logger.warning(
        "Applied flash-attn RotaryEmbedding compatibility shim for ModernBERT "
        "(missing `pos_idx_in_fp32` kwarg in installed flash-attn)."
    )
