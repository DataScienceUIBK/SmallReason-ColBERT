"""Materialize ReasonIR HQ and/or VL to disk as flat {query, positive, negative} datasets.

Mirrors the existing `data/reasonir_hq_processed/` layout so both splits are loaded
the same way at train time.

Usage
-----
    python LaRLI/data/materialize_reasonir.py --config hq --output-dir data/reasonir_hq_processed
    python LaRLI/data/materialize_reasonir.py --config vl --output-dir data/reasonir_vl_processed
    python LaRLI/data/materialize_reasonir.py --config both   # materializes both

The script is idempotent: if the target directory already contains a saved dataset,
it exits early.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# prepare_data.py is in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_data import (
    _is_saved_dataset,
    load_reasonir_hq_dataset,
    load_reasonir_vl_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _save_atomically(dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(tmp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_path.replace(output_path)


def materialize(config: str, output_dir: Path, cache_dir: str | None) -> Path:
    if _is_saved_dataset(output_dir):
        logger.info("%s already materialized at %s, skipping.", config, output_dir)
        return output_dir

    if config == "hq":
        ds = load_reasonir_hq_dataset(cache_dir=cache_dir)
    elif config == "vl":
        ds = load_reasonir_vl_dataset(cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown config: {config}")

    ds = ds.select_columns(["query", "positive", "negative"])
    logger.info("Saving %s (%d rows) to %s", config, len(ds), output_dir)
    _save_atomically(ds, output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize ReasonIR splits to disk.")
    parser.add_argument("--config", choices=["hq", "vl", "both"], required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory (required when --config is hq or vl).",
    )
    parser.add_argument(
        "--hq-output-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "reasonir_hq_processed"),
        help="Used only when --config=both.",
    )
    parser.add_argument(
        "--vl-output-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "reasonir_vl_processed"),
        help="Used only when --config=both.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config in ("hq", "vl"):
        if not args.output_dir:
            raise SystemExit("--output-dir is required when --config is hq or vl")
        out = materialize(args.config, Path(args.output_dir), args.cache_dir)
        print(out)
    else:
        hq_out = materialize("hq", Path(args.hq_output_dir), args.cache_dir)
        vl_out = materialize("vl", Path(args.vl_output_dir), args.cache_dir)
        print(hq_out)
        print(vl_out)


if __name__ == "__main__":
    main()
