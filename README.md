# Text-to-SQL QLoRA Fine-Tuning

Fine-tunes [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
on [Spider](https://yale-lily.github.io/spider) using QLoRA, evaluated with exact-match
and execution accuracy, base vs fine-tuned, plus a LoRA-rank ablation (r=8/16/32).

Project 2 of a 4-project AI/ML portfolio. Project 1: Clinical NLP RAG pipeline
(FAISS + Flan-T5, retrieval metrics + LLM-judge faithfulness scoring).

## Why Phi-3-mini over Llama-3.2-3B

MIT-licensed (no gated-repo approval wait) and stronger baseline code/SQL
performance — both matter for a clean base-vs-fine-tuned comparison story.
Config is model-agnostic (`configs/train_config.yaml: model.base_model`), so
swapping in Llama-3.2-3B-Instruct is a one-line change.

## Why Spider over WikiSQL

WikiSQL is single-table only, so a "schema in prompt" ablation barely moves
accuracy. Spider's multi-table JOINs force genuine schema-awareness, making
both the headline numbers and the ablation meaningful.

## Evaluation splits — read before trusting numbers in `results/eval_report.md`

Spider's hidden test set is **not publicly released**. We train/val-split
`train_spider.json` ourselves (90/10) and report all final metrics on
`dev.json`, which is never used for training or model selection. See
[`data/README.md`](data/README.md) for full detail. This follows Spider's
standard public-comparison convention — `dev` is not a stand-in we mixed up
with `test`.

## Repo structure

```
configs/train_config.yaml   # model, LoRA, data, training hyperparams
data/prepare_data.py        # build train/val/test split files from Spider
src/prompts.py              # with-schema / no-schema prompt templates
src/train.py                # QLoRA fine-tuning entrypoint (run on Colab/Kaggle)
src/infer.py                # generate SQL from a question (+ schema)
src/eval.py                 # exact-match + execution accuracy harness
notebooks/train_colab.ipynb # thin Colab wrapper around src/train.py
results/eval_report.md      # base vs fine-tuned + ablation results
demo/app.py                 # Gradio: question -> SQL -> (optional) result
```

## Training environment

Training requires a CUDA GPU (QLoRA 4-bit via `bitsandbytes`) and runs on free
**Colab** (T4, ~90-min idle disconnect, informal daily GPU-hour cap) or free
**Kaggle** (T4/P100, explicit 30 GPU-hrs/week quota) — not locally on Windows.
Local Python 3.11 / Git Bash handles data prep, the eval harness, and the
Gradio demo.

Two distinct training budgets, to keep ablation and headline numbers from
being confused (see `configs/train_config.yaml: regime`):

- **Ablation runs** (r=8, r=16, r=32): 2,500-example subset, 2 epochs each
  (~25-30 min/run on T4) — same subset across all three ranks for a fair
  comparison.
- **Final run**: winning rank, full ~6,300-example train split, 3 epochs —
  this produces the model reported as "fine-tuned" in `eval_report.md`.

## Status

- [x] Phase 1: repo setup
- [ ] Phase 2: data prep
- [ ] Phase 3: training
- [ ] Phase 4: evaluation
- [ ] Phase 5: Gradio demo

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
```
