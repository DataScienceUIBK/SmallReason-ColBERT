"""Baseline Training Script: Standard ColBERT on ReasonIR-HQ.

Reproduces Reason-ModernColBERT training exactly:
- Standard PyLate ColBERT (no architectural modifications)
- Standard CachedContrastive loss (no coverage scoring)
- GTE-ModernColBERT-v1 starting checkpoint
- ReasonIR-HQ data (100K examples)

This is the sanity check: if we can't match Reason-ModernColBERT's 0.337
nDCG@10 on BRIGHT biology with this, the problem is in our training pipeline,
not in any architectural modification.

Usage:
    torchrun --nproc_per_node=4 train_baseline.py
"""

import argparse
import logging
import os
import sys

import torch

# Make repo-root + this-dir importable. pylate is a pip dependency.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_project_root, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

from data import load_reasonir_hq_dataset, load_training_dataset_from_disk
from modernbert_flash_attn_compat import apply_flash_attn_modernbert_compat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline ColBERT on ReasonIR-HQ")
    parser.add_argument("--base_model", type=str, default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--query_length", type=int, default=128)
    parser.add_argument("--document_length", type=int, default=8192)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output/baseline")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="Path to a pre-materialized {query, positive, negative} dataset on disk. "
             "If unset, falls back to on-the-fly load_reasonir_hq_dataset().",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override run_name (also controls output subdir). Defaults to "
             "'baseline-ColBERT-ReasonIR'.",
    )
    parser.add_argument(
        "--inherit_skiplist",
        action="store_true",
        help="Inherit skiplist_words from the loaded checkpoint's "
             "config_sentence_transformers.json (instead of forcing an empty "
             "skiplist). Use when fine-tuning a checkpoint that was trained "
             "with a non-empty skiplist (e.g. Reason-ModernColBERT).",
    )
    return parser.parse_args()


def _get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _load_train_dataset(args):
    if args.train_data_path:
        logger.info(f"Loading training data from disk: {args.train_data_path}")
        return load_training_dataset_from_disk(args.train_data_path)
    logger.info("Loading ReasonIR-HQ on the fly (no --train_data_path given)...")
    return load_reasonir_hq_dataset(cache_dir=args.cache_dir)


def main():
    args = parse_args()
    run_name = args.run_name or "baseline-ColBERT-ReasonIR"
    output_dir = os.path.join(args.output_dir, run_name)
    global_rank = _get_global_rank()
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if global_rank == 0:
        logger.info(f"Run: {run_name}")
        logger.info(f"Output: {output_dir}")

    # ── 1. Load data ──────────────────────────────────────────────────
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        if global_rank == 0:
            logger.info("Rank 0: Downloading/loading training data...")
            train_dataset = _load_train_dataset(args)
            logger.info(f"Training examples: {len(train_dataset)}")
        torch.distributed.barrier()
        if global_rank != 0:
            logger.info(f"Rank {global_rank}: Loading data from cache...")
            train_dataset = _load_train_dataset(args)
    else:
        logger.info("Loading training data...")
        train_dataset = _load_train_dataset(args)
        logger.info(f"Training examples: {len(train_dataset)}")

    # ── 2. Standard ColBERT model (NO modifications) ──────────────────
    logger.info(f"Loading standard ColBERT: {args.base_model}")
    if args.attn_implementation == "flash_attention_2":
        apply_flash_attn_modernbert_compat()
    model_kwargs = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.torch_dtype:
        if args.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)

    colbert_kwargs = dict(
        model_name_or_path=args.base_model,
        document_length=args.document_length,
        query_length=args.query_length,
        model_kwargs=model_kwargs or None,
    )
    if not args.inherit_skiplist:
        # Default behaviour: force empty skiplist (what the edge-ColBERT curriculum uses).
        colbert_kwargs["skiplist_words"] = []
    model = models.ColBERT(**colbert_kwargs)

    if args.gradient_checkpointing:
        auto_model = getattr(model[0], "auto_model", None)
        if auto_model is not None:
            auto_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on backbone.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── 3. Standard CachedContrastive loss (NO coverage) ──────────────
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=True,
        temperature=1.0,
    )

    # ── 4. Training args ──────────────────────────────────────────────
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=500,
        logging_steps=1,
        fp16=False,
        bf16=True,
        run_name=run_name,
        learning_rate=args.lr,
        dataloader_num_workers=8,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )

    # ── 5. Train ──────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    logger.info("Starting baseline training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    logger.info(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
