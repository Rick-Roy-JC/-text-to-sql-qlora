"""
Phase 3: QLoRA fine-tuning entrypoint. Runs on Colab/Kaggle (CUDA + bitsandbytes
required) — not locally on Windows.

Usage:
    python src/train.py --regime ablation --lora-rank 8
    python src/train.py --regime ablation --lora-rank 16
    python src/train.py --regime ablation --lora-rank 32
    python src/train.py --regime final --lora-rank <winning_rank>

Regime and lora-rank are CLI flags (override configs/train_config.yaml's
regime.name / lora.r at runtime) rather than separate scripts or config edits —
keeps one code path and a visible command-history record of each of the 4 runs.

Checkpointing: output_dir should point at a mounted Google Drive path (see
notebooks/train_colab.ipynb) so Colab's ~90-min idle disconnect doesn't lose
more than `save_steps` worth of progress. Re-running the same command resumes
from the latest checkpoint automatically if one exists.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(config: dict, lora_rank: int):
    base_model_name = config["model"]["base_model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Phi-3's pad_token is identical to eos_token (same id) — fine for generation,
    # but the data collator below must mask padding out of the loss explicitly,
    # since a padded position is otherwise indistinguishable from a real EOS target.

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        # trust_remote_code intentionally omitted: it pulls Phi-3's own cached
        # modeling_phi3.py from the HF Hub, which expects the old
        # rope_scaling["type"] key. Current transformers (>=4.44) ships a
        # native, version-matched Phi-3 implementation using rope_scaling["rope_type"] —
        # using that built-in implementation avoids the KeyError entirely.
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=config["lora"]["alpha"],  # fixed across r=8/16/32 per the locked ablation design
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_datasets(config: dict, tokenizer, regime: str):
    variant = config["data"]["prompt_variant"]
    processed_dir = REPO_ROOT / "data" / "processed"

    train_path = processed_dir / f"train_{variant}.jsonl"
    val_path = processed_dir / f"val_{variant}.jsonl"

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")

    regime_cfg = config["regime"][regime]
    subset_size = regime_cfg["train_subset_size"]
    if subset_size is not None:
        rng = random.Random(config["training"]["seed"])
        indices = rng.sample(range(len(train_ds)), min(subset_size, len(train_ds)))
        train_ds = train_ds.select(indices)

    max_seq_length = config["model"]["max_seq_length"]

    def tokenize_with_masked_prompt(example):
        full_ids = tokenizer(
            example["text"], truncation=True, max_length=max_seq_length, add_special_tokens=False
        )["input_ids"]

        # Recover the prompt-only length so we can mask prompt+schema tokens out
        # of the loss; only the gold SQL + EOS span should contribute to loss.
        prompt_text = example["text"][: example["text"].rfind("### SQL:\n") + len("### SQL:\n")]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    train_ds = train_ds.map(tokenize_with_masked_prompt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_with_masked_prompt, remove_columns=val_ds.column_names)

    return train_ds, val_ds


class PaddingCollator:
    """Pads input_ids/attention_mask/labels to the longest sequence in the batch.

    Padding positions get label=-100 (not just attention_mask=0) — required
    because pad_token_id == eos_token_id for Phi-3's tokenizer, so an unmasked
    padded label would silently teach the model extra, unintended EOS targets.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["ablation", "final"], required=True)
    parser.add_argument("--lora-rank", type=int, required=True, choices=[8, 16, 32])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_config.yaml"))
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "checkpoints"),
        help="Override for Colab: point this at a mounted Google Drive path.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_name = f"{args.regime}_r{args.lora_rank}"
    output_dir = str(Path(args.output_root) / run_name)

    model, tokenizer = build_model_and_tokenizer(config, args.lora_rank)
    train_ds, val_ds = build_datasets(config, tokenizer, args.regime)

    epochs = config["regime"][args.regime]["epochs"]
    t = config["training"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        optim=t["optim"],
        num_train_epochs=epochs,
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=t["save_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=t["bf16"],
        seed=t["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PaddingCollator(tokenizer),
    )

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        print(f"[train] Resuming from checkpoint: {last_checkpoint}")
    else:
        print(f"[train] No existing checkpoint at {output_dir} — starting fresh.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    adapter_dir = REPO_ROOT / config["output"]["adapter_dir"] / run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] Final adapter saved to {adapter_dir}")

    metrics = trainer.evaluate()
    metrics_path = REPO_ROOT / "results" / f"train_metrics_{run_name}.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Final val metrics written to {metrics_path}: {metrics}")


if __name__ == "__main__":
    main()
