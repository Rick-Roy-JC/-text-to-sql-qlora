"""
Phase 2: data prep.

Expects data/raw/spider.zip to already be downloaded manually (see
data/README.md for why — no stable scriptable direct-download URL exists).

Steps:
  1. Unzip into data/spider/
  2. Carve train/val split from train_spider.json only (val_fraction in
     configs/train_config.yaml), stratified by db_id.
  3. Audit token lengths of the fully-rendered with-schema prompt (+ gold SQL)
     for every train, val, and dev example using the actual base model
     tokenizer. Report counts exceeding the safe threshold (default 3500).
  4. (Run again after the threshold-handling strategy is confirmed) emit
     processed JSONL files for train/val/dev x {with_schema, no_schema}.

This script intentionally stops after step 3 on first run — it prints the
audit report and exits without writing processed files, so the threshold
strategy can be confirmed before anything is generated.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prompts import build_schema_block, build_training_example  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_ZIP = DATA_DIR / "raw" / "spider_data.zip"
SPIDER_DIR = DATA_DIR / "spider"

SAFE_TOKEN_THRESHOLD = 3500  # leaves ~600 tokens for SQL + special tokens within 4096 ctx


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "train_config.yaml") as f:
        return yaml.safe_load(f)


def ensure_unpacked() -> None:
    if SPIDER_DIR.exists() and (SPIDER_DIR / "tables.json").exists():
        print(f"[prepare_data] Spider already unpacked at {SPIDER_DIR}")
        return
    if not RAW_ZIP.exists():
        raise FileNotFoundError(
            f"Expected Spider zip at {RAW_ZIP}. Download manually from "
            "https://yale-lily.github.io/spider and place it there."
        )
    print(f"[prepare_data] Unpacking {RAW_ZIP} -> {SPIDER_DIR}")
    if SPIDER_DIR.exists():
        # Empty placeholder dir from a prior partial run, or wrong contents — clear it
        # so the extracted "spider_data" folder can be renamed onto this path cleanly.
        import shutil
        shutil.rmtree(SPIDER_DIR)
    with zipfile.ZipFile(RAW_ZIP) as zf:
        zf.extractall(DATA_DIR)
    # Spider's zip typically extracts to a "spider" or "spider_data" subfolder;
    # normalize so tables.json etc. always end up directly under data/spider/.
    if not (SPIDER_DIR / "tables.json").exists():
        candidates = [p for p in DATA_DIR.iterdir() if p.is_dir() and (p / "tables.json").exists()]
        if not candidates:
            raise FileNotFoundError(
                f"Unpacked zip but couldn't find tables.json under {DATA_DIR}. "
                "Inspect the zip's internal folder structure and adjust this script."
            )
        candidates[0].rename(SPIDER_DIR)
    print("[prepare_data] Unpack complete.")


def split_train_val(train_examples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratify by db_id so val isn't dominated by a handful of schemas."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for ex in train_examples:
        by_db[ex["db_id"]].append(ex)

    rng = random.Random(seed)
    train_split, val_split = [], []
    for db_id, examples in by_db.items():
        examples = examples[:]
        rng.shuffle(examples)
        n_val = max(1, round(len(examples) * val_fraction)) if len(examples) > 1 else 0
        val_split.extend(examples[:n_val])
        train_split.extend(examples[n_val:])
    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def audit_token_lengths(
    examples: list[dict],
    tables_by_db: dict[str, dict],
    tokenizer,
    split_name: str,
    threshold: int,
) -> list[dict]:
    over_threshold = []
    for ex in examples:
        schema_block = build_schema_block(ex["db_id"], tables_by_db[ex["db_id"]])
        rendered = build_training_example(
            question=ex["question"],
            schema_block=schema_block,
            sql=ex["query"],
            variant="with_schema",
            eos_token=tokenizer.eos_token,
        )
        n_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if n_tokens > threshold:
            over_threshold.append({
                "db_id": ex["db_id"],
                "question": ex["question"],
                "n_tokens": n_tokens,
            })

    pct = 100 * len(over_threshold) / max(1, len(examples))
    print(f"[prepare_data] {split_name}: {len(over_threshold)}/{len(examples)} ({pct:.2f}%) exceed {threshold} tokens (with-schema)")
    return over_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=SAFE_TOKEN_THRESHOLD)
    parser.add_argument("--audit-only", action="store_true", default=True,
                         help="Stop after the token audit (default). Pass --no-audit-only once threshold handling is confirmed.")
    parser.add_argument("--no-audit-only", dest="audit_only", action="store_false")
    args = parser.parse_args()

    config = load_config()
    ensure_unpacked()

    with open(SPIDER_DIR / "tables.json") as f:
        tables_list = json.load(f)
    tables_by_db = {t["db_id"]: t for t in tables_list}

    with open(SPIDER_DIR / "train_spider.json") as f:
        train_all = json.load(f)
    with open(SPIDER_DIR / "dev.json") as f:
        dev = json.load(f)

    train_split, val_split = split_train_val(
        train_all, val_fraction=config["data"]["val_fraction"], seed=config["training"]["seed"]
    )
    print(f"[prepare_data] train_spider.json: {len(train_all)} total -> "
          f"{len(train_split)} train / {len(val_split)} val")
    print(f"[prepare_data] dev.json (held-out eval, untouched by split logic): {len(dev)} examples")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    print(f"\n[prepare_data] Token-length audit (with-schema prompt + gold SQL + EOS, threshold={args.threshold}):")
    over_train = audit_token_lengths(train_split, tables_by_db, tokenizer, "train", args.threshold)
    over_val = audit_token_lengths(val_split, tables_by_db, tokenizer, "val", args.threshold)
    over_dev = audit_token_lengths(dev, tables_by_db, tokenizer, "dev (held-out eval)", args.threshold)

    audit_report = {
        "threshold": args.threshold,
        "train_over": over_train,
        "val_over": over_val,
        "dev_over": over_dev,
    }
    report_path = REPO_ROOT / "results" / "token_audit.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n[prepare_data] Full audit written to {report_path}")

    if args.audit_only:
        print(
            "\n[prepare_data] Stopping here (audit-only mode). "
            "Confirm the over-threshold handling strategy, then re-run with --no-audit-only "
            "to generate processed train/val/dev x {with_schema, no_schema} JSONL files."
        )
        return

    # Audit came back at 0 over-threshold examples across all splits, so no
    # truncation/exclusion logic is needed here — every example is emitted as-is.
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    splits = {"train": train_split, "val": val_split, "dev": dev}
    for split_name, examples in splits.items():
        for variant in ("with_schema", "no_schema"):
            out_path = processed_dir / f"{split_name}_{variant}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    schema_block = (
                        build_schema_block(ex["db_id"], tables_by_db[ex["db_id"]])
                        if variant == "with_schema" else None
                    )
                    text = build_training_example(
                        question=ex["question"],
                        schema_block=schema_block,
                        sql=ex["query"],
                        variant=variant,
                        eos_token=tokenizer.eos_token,
                    )
                    record = {
                        "db_id": ex["db_id"],
                        "question": ex["question"],
                        "query": ex["query"],
                        "text": text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[prepare_data] Wrote {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
