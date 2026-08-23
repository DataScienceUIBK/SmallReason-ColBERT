"""Data preparation for LaRLI training and evaluation.

Loads and joins data from HuggingFace:
- Training: reasonir/reasonir-data (config="hq") + xlangai/BRIGHT (config="documents")
- Evaluation: xlangai/BRIGHT (config="examples" + per-domain documents)

No data generation required — all data is publicly available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def _is_saved_dataset(path: str | Path) -> bool:
    p = Path(path)
    return (p / "dataset_info.json").exists() and (p / "state.json").exists()


def _build_id2doc(bright_documents: DatasetDict) -> dict[str, str]:
    """Build a mapping from document ID to document text from BRIGHT documents."""
    id2doc = {}
    for task_name in bright_documents.keys():
        task_data = bright_documents[task_name]
        for entry in task_data:
            doc_id = str(entry["id"])
            content = entry["content"]
            id2doc[doc_id] = content
    return id2doc


def _process_hq_entry(entry: dict, id2doc: dict[str, str]) -> dict:
    """Process a single ReasonIR-HQ entry by resolving document IDs to text.

    The HQ split has:
    - query: [instruction_str, actual_query_str]
    - pos: [[reasoning_prefix, doc_id_or_text], ...]
    - neg: [[reasoning_prefix, neg_text], ...]
    """
    pos_docs = entry["pos"]
    neg_docs = entry["neg"]

    # Resolve the positive document
    pos_prefix = pos_docs[0][0] if len(pos_docs) > 0 and len(pos_docs[0]) > 1 else ""
    pos_id = pos_docs[0][1] if len(pos_docs) > 0 and len(pos_docs[0]) > 1 else ""

    # Try to resolve from BRIGHT documents, fall back to using as-is
    pos_text = id2doc.get(pos_id, pos_id)
    if pos_prefix:
        pos_text = pos_prefix + " " + pos_text

    # Build query text (join instruction + query)
    query_text = " ".join(entry["query"]) if isinstance(entry["query"], list) else entry["query"]

    # Build negative text
    neg_text = ""
    if neg_docs and len(neg_docs[0]) > 1:
        neg_text = neg_docs[0][0] + " " + neg_docs[0][1]
    elif neg_docs and len(neg_docs[0]) == 1:
        neg_text = neg_docs[0][0]

    return {
        "query": query_text,
        "positive": pos_text,
        "negative": neg_text,
    }


def load_reasonir_hq_dataset(
    reasonir_path: str = "reasonir/reasonir-data",
    bright_path: str = "xlangai/BRIGHT",
    cache_dir: str | None = None,
    test_size: float = 0.0,
) -> Dataset | tuple[Dataset, Dataset]:
    """Load ReasonIR-HQ training data with resolved document texts.

    Parameters
    ----------
    reasonir_path
        HuggingFace path for ReasonIR data.
    bright_path
        HuggingFace path for BRIGHT (to resolve document IDs).
    cache_dir
        Directory to cache downloaded datasets.
    test_size
        If > 0, split into train and validation sets.

    Returns
    -------
    Dataset or (Dataset, Dataset)
        The processed training dataset, or (train, val) if test_size > 0.
    """
    logger.info("Loading ReasonIR-HQ dataset...")
    hq_dataset = load_dataset(
        reasonir_path, "hq", split="train", cache_dir=cache_dir
    )

    logger.info("Loading BRIGHT documents for ID resolution...")
    bright_docs = load_dataset(bright_path, "documents", cache_dir=cache_dir)

    logger.info("Building document ID -> text mapping...")
    id2doc = _build_id2doc(bright_docs)
    logger.info(f"Built mapping with {len(id2doc)} documents.")

    logger.info("Processing HQ entries...")
    processed = hq_dataset.map(
        lambda x: _process_hq_entry(x, id2doc),
        remove_columns=hq_dataset.column_names,
        desc="Resolving document IDs",
    )

    # Filter out entries with empty positive or negative texts
    processed = processed.filter(
        lambda x: len(x["positive"]) > 10 and len(x["negative"]) > 10,
        desc="Filtering empty entries",
    )

    logger.info(f"Processed dataset: {len(processed)} examples.")

    if test_size > 0:
        splits = processed.train_test_split(test_size=test_size, seed=42)
        return splits["train"], splits["test"]

    return processed


def _process_vl_entry(entry: dict) -> dict:
    """Process a single ReasonIR-VL entry.

    The VL split embeds full positive/negative document text directly (no BRIGHT
    document-ID resolution needed). Schema matches HQ otherwise.
    """
    pos_docs = entry["pos"]
    neg_docs = entry["neg"]

    if pos_docs and len(pos_docs[0]) > 1:
        pos_text = (pos_docs[0][0] + " " + pos_docs[0][1]).strip()
    elif pos_docs and len(pos_docs[0]) == 1:
        pos_text = pos_docs[0][0]
    else:
        pos_text = ""

    query_text = " ".join(entry["query"]) if isinstance(entry["query"], list) else entry["query"]

    if neg_docs and len(neg_docs[0]) > 1:
        neg_text = (neg_docs[0][0] + " " + neg_docs[0][1]).strip()
    elif neg_docs and len(neg_docs[0]) == 1:
        neg_text = neg_docs[0][0]
    else:
        neg_text = ""

    return {"query": query_text, "positive": pos_text, "negative": neg_text}


def load_reasonir_vl_dataset(
    reasonir_path: str = "reasonir/reasonir-data",
    cache_dir: str | None = None,
    test_size: float = 0.0,
) -> Dataset | tuple[Dataset, Dataset]:
    """Load ReasonIR-VL (Varied-Length, ~245K) with text already embedded.

    Same output schema as `load_reasonir_hq_dataset`: columns {query, positive, negative}.
    """
    logger.info("Loading ReasonIR-VL dataset...")
    vl_dataset = load_dataset(reasonir_path, "vl", split="train", cache_dir=cache_dir)

    logger.info("Processing VL entries...")
    processed = vl_dataset.map(
        _process_vl_entry,
        remove_columns=vl_dataset.column_names,
        desc="Flattening VL",
    )
    processed = processed.filter(
        lambda x: len(x["positive"]) > 10 and len(x["negative"]) > 10,
        desc="Filtering empty VL entries",
    )
    logger.info(f"Processed VL dataset: {len(processed)} examples.")

    if test_size > 0:
        splits = processed.train_test_split(test_size=test_size, seed=42)
        return splits["train"], splits["test"]
    return processed


def load_training_dataset_from_disk(path: str | Path) -> Dataset:
    """Load a pre-materialized {query, positive, negative} dataset from disk."""
    p = Path(path)
    if not _is_saved_dataset(p):
        raise FileNotFoundError(
            f"Not a saved HF dataset at {p} (missing dataset_info.json/state.json)."
        )
    ds = load_from_disk(str(p))
    if isinstance(ds, DatasetDict):
        ds = ds[next(iter(ds.keys()))]
    required = {"query", "positive", "negative"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset at {p} is missing columns: {missing}")
    return ds.select_columns(["query", "positive", "negative"])


def load_bright_evaluation(
    bright_path: str = "xlangai/BRIGHT",
    splits: list[str] | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Load BRIGHT evaluation data.

    Parameters
    ----------
    bright_path
        HuggingFace path for BRIGHT.
    splits
        Which BRIGHT splits to load. If None, loads all 12.
    cache_dir
        Directory to cache downloaded datasets.

    Returns
    -------
    dict
        Dictionary with keys:
        - "queries": {split_name: {query_id: query_text}}
        - "corpus": {split_name: {doc_id: doc_text}}
        - "qrels": {split_name: {query_id: {doc_id: relevance}}}
    """
    all_splits = [
        "biology", "earth_science", "economics", "psychology",
        "robotics", "stackoverflow", "sustainable_living",
        "leetcode", "pony", "aops",
        "theoremqa_questions", "theoremqa_theorems",
    ]
    if splits is None:
        splits = all_splits

    logger.info(f"Loading BRIGHT evaluation for splits: {splits}")

    # Load examples (queries + gold IDs)
    examples = load_dataset(bright_path, "examples", cache_dir=cache_dir)

    # Load documents
    documents = load_dataset(bright_path, "documents", cache_dir=cache_dir)

    result = {"queries": {}, "corpus": {}, "qrels": {}}

    for split_name in splits:
        if split_name not in examples:
            logger.warning(f"Split '{split_name}' not found in BRIGHT examples.")
            continue

        split_examples = examples[split_name]
        split_queries = {}
        split_qrels = {}

        for entry in split_examples:
            qid = entry["id"]
            query = entry["query"]
            split_queries[qid] = query

            # Build qrels from gold_ids
            gold_ids = entry.get("gold_ids", [])
            if gold_ids:
                split_qrels[qid] = {doc_id: 1 for doc_id in gold_ids}

        result["queries"][split_name] = split_queries
        result["qrels"][split_name] = split_qrels

        # Build corpus from documents
        if split_name in documents:
            split_corpus = {}
            for entry in documents[split_name]:
                split_corpus[str(entry["id"])] = entry["content"]
            result["corpus"][split_name] = split_corpus

    return result
