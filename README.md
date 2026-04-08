# Alloy-Bench Eval

Standalone workspace for evaluating LLMs on [Alloy-Bench](https://huggingface.co/datasets/nn-tech/Alloy-Bench) — Russian multiple-choice benchmark for metallurgy and mining.

This repo uses a dedicated `uv` environment. It is intentionally separate from `turbo-allignment`, but the direct benchmark dependencies are pinned to the same versions used by the `turbo-allignment` benchmark runner so the setup can be reproduced on another machine.

## Quick start

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment and install dependencies
cd alloy-bench-eval
uv sync

# 3. Run both models
bash run_all.sh
```

For a reproducible setup on another machine, prefer:

```bash
cd alloy-bench-eval
uv sync --frozen
```

`--frozen` installs strictly from `uv.lock`.

## Environment compatibility

The project keeps a separate `.venv`, but pins the core benchmark runtime packages to match the environment used by the benchmark runner in `turbo-allignment`.

Benchmark runtime compatibility target:

- Python `>=3.10,<3.13`
- `vllm==0.8.5.post1`
- `transformers==4.51.3`
- `peft==0.8.2`
- `torch==2.6.0`
- `outlines==0.1.11`
- `openpyxl==3.1.5`
- `pandas==2.3.3`

Dataset conversion dependencies are pinned separately for reproducibility in this repo:

- `datasets==4.8.4`
- `pyarrow==23.0.1`

If the benchmark environment in `turbo-allignment` changes, update these pins and regenerate `uv.lock`.

The script will:
1. Convert the dataset from HuggingFace to XLSX (first run only)
2. Evaluate `nn-tech/MetalGPT-1`
3. Evaluate `Qwen/Qwen3-32B`
4. Save results to `results/`

## Manual usage

### Convert dataset

From HuggingFace:
```bash
uv run python convert_dataset.py --output data/Alloy-Bench.xlsx
```

From local parquet (e.g. after `huggingface-cli download`):
```bash
uv run python convert_dataset.py --input /path/to/parquet --output data/Alloy-Bench.xlsx
```

### Run evaluation

With model name directly:
```bash
uv run python run_benchmark.py \
  --model nn-tech/MetalGPT-1 \
  --data-path data/Alloy-Bench.xlsx \
  --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 2048
```

With config file:
```bash
uv run python run_benchmark.py \
  --config configs/pure_metalgpt.json \
  --data-path data/Alloy-Bench.xlsx
```

With LoRA adapter:
```bash
uv run python run_benchmark.py \
  --model Qwen/Qwen3-32B \
  --lora /path/to/lora/adapter \
  --data-path data/Alloy-Bench.xlsx
```

You can verify the interpreter and package versions with:

```bash
uv run python -c "import sys, vllm, transformers, peft, torch; print(sys.executable); print(sys.version); print(vllm.__version__, transformers.__version__, peft.__version__, torch.__version__)"
```

### CLI parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | — | HF model name or local path |
| `--config` | — | JSON config (alternative to `--model`) |
| `--lora` | — | LoRA adapter path |
| `--data-path` | — | BENCH3-style XLSX dataset |
| `--output` | auto | Output XLSX path |
| `--batch-size` | 50 | Batch size for vLLM |
| `--temperature` | 0.6 | Generation temperature |
| `--top-p` | 0.95 | Top-p sampling |
| `--top-k` | 20 | Top-k sampling |
| `--max-new-tokens` | 2048 | Max generation length |

## Output format

Each results XLSX contains three sheets:

- **Results** — accuracy per knowledge area + macro average
- **Samples** — per-question correctness with model/correct answers
- **Metadata** — model path, dataset path, timestamp

## Structure

```
alloy-bench-eval/
├── pyproject.toml        # pinned runtime dependencies
├── uv.lock               # reproducible uv lockfile
├── run_all.sh            # one-command launcher
├── run_benchmark.py      # evaluation runner
├── bench_eval.py         # core: prompts, vLLM inference, scoring
├── convert_dataset.py    # HF/parquet → XLSX converter
├── configs/
│   ├── pure_metalgpt.json
│   └── pure_qwen3.json
├── data/                 # dataset goes here
└── results/              # output goes here
```
