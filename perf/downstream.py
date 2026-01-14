from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cupy as cp
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

TARGET = "st_xfer_time_ms"

FEATURE_COLS = [
    "st_files",
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_bytes_xfered",
    "st_faults",
    "st_files_skipped",
    "st_skipped_errors",
    "encrypt_data",
]

DROP_COLS = ["grp_uuid", "user_id", "request_time", "complete_time"]

K_SIGMA = 3.0
SEED = 0
CSV_ENGINE = "pyarrow"
XGB_DEVICE = "cuda"
XGB_TREE_METHOD = "hist"
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream XGBoost metrics on real vs synthetic data."
    )
    parser.add_argument(
        "--real",
        type=Path,
        default=Path("../datasets/filtered.csv"),
        help="Path to the real CSV. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("../output/before.csv"),
        help="Path to the synthetic CSV. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../output/before.json"),
        help="Path to write the JSON report. Defaults to %(default)s.",
    )
    return parser.parse_args()


def load_dataframe(path: Path, engine: str | None = None) -> pd.DataFrame:
    kwargs = {"engine": engine} if engine else {}
    df = pd.read_csv(path, **kwargs)
    if "encrypt_data" in df.columns:
        df["encrypt_data"] = df["encrypt_data"].astype(np.int8)  # True->1, False->0
    return df


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr = spearmanr(y_true, y_pred).correlation
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def evaluate_model(model, X_tr, y_tr, X_te, y_te, y_te_raw) -> dict:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if hasattr(pred, "get"):
        pred = pred.get()
    if hasattr(y_te, "get"):
        y_te = y_te.get()
    if hasattr(y_te_raw, "get"):
        y_te_raw = y_te_raw.get()

    return {
        "mae_log": mean_absolute_error(y_te, pred),
        "spearman": safe_spearman(y_te, pred),
        "mae_log_top90": mean_absolute_error(
            y_te[y_te_raw >= np.percentile(y_te_raw, 90)],
            pred[y_te_raw >= np.percentile(y_te_raw, 90)],
        ),
        "mae_log_top99": mean_absolute_error(
            y_te[y_te_raw >= np.percentile(y_te_raw, 99)],
            pred[y_te_raw >= np.percentile(y_te_raw, 99)],
        ),
        "pinball_90": pinball_loss(y_te, pred, 0.9),
        "pinball_95": pinball_loss(y_te, pred, 0.95),
    }


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_COLS),
        ]
    )


def make_xgb(
    device: str,
    tree_method: str,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        tree_method=tree_method,
        device=device,
        random_state=seed,
        n_jobs=0,
    )


def downstream_eval_xgb(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    seed: int,
    device: str,
    tree_method: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> dict:
    real_train, real_test = train_test_split(
        real_df, test_size=0.2, random_state=seed
    )

    y_tr = np.log1p(real_train[TARGET].values)
    y_te = np.log1p(real_test[TARGET].values)
    y_te_raw = real_test[TARGET].values
    y_syn = np.log1p(synth_df[TARGET].values)

    real_train = real_train.drop(columns=DROP_COLS, errors="ignore")
    real_test = real_test.drop(columns=DROP_COLS, errors="ignore")
    synth_df = synth_df.drop(columns=DROP_COLS, errors="ignore")

    pre = make_preprocessor()
    X_tr = pre.fit_transform(real_train)
    X_te = pre.transform(real_test)
    X_syn = pre.transform(synth_df)

    X_tr = cp.asarray(X_tr)
    X_te = cp.asarray(X_te)
    X_syn = cp.asarray(X_syn)
    y_tr = cp.asarray(y_tr)
    y_te = cp.asarray(y_te)
    y_syn = cp.asarray(y_syn)

    model = make_xgb(
        device=device,
        tree_method=tree_method,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )

    rr = evaluate_model(model, X_tr, y_tr, X_te, y_te, y_te_raw)
    sr = evaluate_model(model, X_syn, y_syn, X_te, y_te, y_te_raw)

    return {
        "RR": rr,
        "SR": sr,
        "Delta": {k: sr[k] - rr[k] for k in rr},
    }


def filter_by_sigma(df: pd.DataFrame, mu: float, sigma: float, k: float) -> pd.DataFrame:
    lower = mu - k * sigma
    upper = mu + k * sigma
    return df[(df[TARGET] >= lower) & (df[TARGET] <= upper)]


def main() -> None:
    args = parse_args()
    real_df = load_dataframe(args.real, engine=CSV_ENGINE)
    synth_df = load_dataframe(args.synthetic, engine=CSV_ENGINE)

    mu = real_df[TARGET].mean()
    sigma = real_df[TARGET].std()

    real_filtered = filter_by_sigma(real_df, mu, sigma, K_SIGMA)
    synth_filtered = filter_by_sigma(synth_df, mu, sigma, K_SIGMA)

    if real_filtered.empty or synth_filtered.empty:
        raise ValueError("Filtered data is empty; adjust K_SIGMA or inputs.")

    results = downstream_eval_xgb(
        real_filtered,
        synth_filtered,
        seed=SEED,
        device=XGB_DEVICE,
        tree_method=XGB_TREE_METHOD,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, default=float)
    args.output.write_text(payload)
    print(payload)
    print(f"Saved downstream metrics to {args.output}")


if __name__ == "__main__":
    main()
