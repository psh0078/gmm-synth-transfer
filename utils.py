from typing import Iterable
from argparse import ArgumentParser

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def iter_progress(iterable: Iterable, desc: str | None = None, leave: bool = True):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)

def log_stage(message: str):
    print(f"[stage] {message}")

def parse_args():
    parser = ArgumentParser(description="Generate a synthetic dataset using a fitted GMM.")
    parser.add_argument(
        "input",
        nargs="?",
        default="datasets/filtered.csv",
        help="Path to the CSV containing the real transfer records. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output",
        default="output/output.csv",
        help="Path to write the synthetic CSV. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Train the GMM on a GPU using PyTorch.",
    )
    parser.add_argument(
        "--gpu-device",
        default="cuda",
        help="Torch device string to use when --use-gpu is set. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-max-iter",
        type=int,
        default=200,
        help="Max EM iterations for the GPU trainer. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-n-init",
        type=int,
        default=1,
        help="Number of random initializations for the GPU trainer. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance (in log-likelihood delta) for the GPU trainer. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=16384,
        help="Batch size (in samples) for GPU EM responsibilities. Use 0 to process the full dataset. Defaults to %(default)s.",
    )
    return parser.parse_args()