"""Concatenate several saved-to-disk {query, positive, negative} datasets into one.

Useful for building a single Stage-B data dir from the three RaDeR splits, or a
single Stage-D data dir from multiple reason-embed variants.

Usage
-----
  python LaRLI/data/merge_datasets.py \
      --inputs data/rader_numina_all_processed \
               data/rader_math_qas_lex_processed \
               data/rader_math_cot_lex_processed \
      --output-dir data/rader_merged_processed

Preserves only the three columns {query, positive, negative}, shuffles the
result with a fixed seed so the training loop sees interleaved sources.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def _free_bytes(path: Path) -> int:
    """Available bytes on the filesystem hosting `path`."""
    p = path if path.exists() else path.parent
    while not p.exists():
        p = p.parent
    stats = shutil.disk_usage(str(p))
    return stats.free


def _safe_rmtree(path: Path) -> None:
    """Remove `path` even if partial writes left stray files behind."""
    if not path.exists():
        return
    shutil.rmtree(path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


REQUIRED_COLS = ("query", "positive", "negative")


def _load_one(path: Path):
    if not (path / "dataset_info.json").exists():
        raise FileNotFoundError(f"{path} is not a saved HF dataset (missing dataset_info.json)")
    ds = load_from_disk(str(path))
    missing = set(REQUIRED_COLS) - set(ds.column_names)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return ds.select_columns(list(REQUIRED_COLS))


def main() -> None:
    p = argparse.ArgumentParser(description="Merge saved-to-disk triplet datasets.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Paths to input dataset directories.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-shards", type=int, default=None,
                   help="Force shard count (default: datasets auto). Use 1 for a "
                        "single .arrow file — slower but safer on flaky FS.")
    p.add_argument("--num-proc", type=int, default=1,
                   help="Parallel shard writers. Default=1 (serial) because HF "
                        "save_to_disk has been observed to race on NFS/lustre. "
                        "Bump to 4 only on a healthy local FS.")
    p.add_argument("--min-free-gb", type=int, default=20,
                   help="Abort early if the filesystem has less than this many "
                        "GB free (guards against mid-write disk-full failures).")
    args = p.parse_args()

    parts = []
    total = 0
    for raw in args.inputs:
        path = Path(raw)
        ds = _load_one(path)
        logger.info("  + %s (%d rows)", path, len(ds))
        parts.append(ds)
        total += len(ds)

    combined = concatenate_datasets(parts).shuffle(seed=args.seed)
    logger.info("Combined: %d rows across %d inputs", len(combined), len(parts))
    assert len(combined) == total

    out = Path(args.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    free_gb = _free_bytes(out.parent) / (1 << 30)
    logger.info("Free space on target FS: %.1f GB", free_gb)
    if free_gb < args.min_free_gb:
        raise SystemExit(
            f"Only {free_gb:.1f} GB free on {out.parent}, below --min-free-gb "
            f"{args.min_free_gb}. Aborting to avoid a mid-write failure."
        )

    tmp = out.parent / f".{out.name}.tmp"
    # Defensive: blow away ANY leftover from a prior crash before starting.
    _safe_rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if args.num_shards is not None:
        save_kwargs["num_shards"] = args.num_shards
    if args.num_proc and args.num_proc > 1:
        save_kwargs["num_proc"] = args.num_proc
    # If num_proc is 1 (default), DO NOT pass it — HF's default is serial and
    # we avoid the multiprocessing path entirely. Passing num_proc=1 explicitly
    # still triggers the multiproc code path in some datasets versions.

    try:
        combined.save_to_disk(str(tmp), **save_kwargs)
    except Exception:
        logger.exception("save_to_disk failed; leaving %s for inspection.", tmp)
        raise

    # Verify every shard claimed in state.json actually exists AND is
    # internally readable. On networked filesystems we've seen shards that
    # have the expected byte length but are structurally malformed — the
    # training process hits `OSError: Expected to be able to read N bytes for
    # message body, got M` only at load time. Fail loudly HERE instead.
    state_file = tmp / "state.json"
    if not state_file.exists():
        raise SystemExit(f"state.json missing from {tmp}; save_to_disk incomplete.")
    import json
    state = json.loads(state_file.read_text())
    expected_shards = state.get("_data_files") or []
    import pyarrow as pa
    import pyarrow.ipc as ipc
    for shard in expected_shards:
        path = tmp / shard["filename"]
        if not path.exists():
            raise SystemExit(
                f"Shard missing from {tmp}: {shard['filename']}. "
                "Re-run the merge — the filesystem dropped shards mid-write."
            )
        try:
            with pa.memory_map(str(path), "r") as src:
                reader = ipc.open_stream(src)
                _ = reader.read_all()
        except Exception as e:
            raise SystemExit(
                f"Shard {path} exists ({path.stat().st_size} bytes) but is "
                f"structurally malformed: {type(e).__name__}: {e}. "
                "Re-run the merge (preferably with fewer --num-shards)."
            )
    logger.info("Verified %d shards in %s (existence + readability).",
                len(expected_shards), tmp)

    # Atomic-ish swap.
    if out.exists():
        backup = out.parent / f".{out.name}.old"
        _safe_rmtree(backup)
        out.replace(backup)
    tmp.replace(out)
    if (out.parent / f".{out.name}.old").exists():
        _safe_rmtree(out.parent / f".{out.name}.old")
    logger.info("Saved to %s", out)
    print(out)


if __name__ == "__main__":
    main()
