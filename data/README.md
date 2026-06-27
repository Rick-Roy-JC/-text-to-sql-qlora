# Data

Spider is not committed to this repo (license + size). `prepare_data.py` downloads
and unpacks it into `data/spider/` (gitignored).

## Splits — how we use them

- **`train_spider.json`** — public training set (~7000 examples across multiple DBs).
  We carve our own train/val split from this file only (default 90/10, see
  `configs/train_config.yaml: data.val_fraction`). This is the *only* file used
  to choose hyperparameters, pick the winning LoRA rank, and select checkpoints.
- **`dev.json`** — public dev set (~1034 examples). Spider's official hidden test
  set is not publicly released. Per Spider's standard evaluation protocol, `dev`
  is the conventional substitute reported in public comparisons. **We never train
  on, validate on, or tune against `dev.json`** — it is touched exactly once, to
  produce the final numbers in `results/eval_report.md`.
- **`tables.json` / `database/`** — schema metadata and SQLite databases, used for
  (a) building schema-aware prompts and (b) executing generated SQL for execution
  accuracy.

## Known property: train/dev schema-density gap

Token-length audits (see `results/token_audit.json`) show train/val with-schema
prompts running notably longer (p99 ~3,150-3,220 tokens) than dev (p99 ~1,000).
This is a property of **Spider's official train/dev split** — dev databases are
generally simpler/smaller schemas by design — not an artifact of how we carved
our own train/val split from `train_spider.json`. Documented here so it doesn't
get misread as a splitting bug later.

## Source

https://yale-lily.github.io/spider — download link is gated behind a short form
on the Spider page; `prepare_data.py` expects the zip to already be downloaded
manually into `data/raw/` (auto-download isn't reliable since the host changes
the link periodically), then it unpacks and preprocesses from there.
