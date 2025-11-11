from typing import Iterable

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
