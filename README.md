# gmm-guided synthetic transfer data

## Environment (uv)

- Install [uv](https://github.com/astral-sh/uv) if you do not already have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Sync the project dependencies and create `.venv`: `uv sync`
- Either activate the venv (`source .venv/bin/activate`) or prefix commands with `uv run` to use the locked toolchain.
- Python 3.13+ is required; uv will install it automatically when syncing.

## Train the GMM

- GPU training: `uv run python src/gmm.py datasets/filtered.csv --output output/big-k.csv --use-gpu --gpu-device cuda:1 --gpu-batch-size 65536 --gpu-max-cap`
