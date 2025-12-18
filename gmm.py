from pathlib import Path
from utils import iter_progress, log_stage, parse_args

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer
from gmm_gpu import fit_best_gmm_gpu, invert_latent_samples_gpu, prepare_boxcox_features_gpu

FEATURE_COLS = [
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_bytes_xfered",
    "st_faults",
    "st_skipped_errors",
    "st_xfer_time_ms"
]

COUNT_LIKE_COLS = [
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_faults",
]

STATUS_COLS = ["st_successful", "st_failed", "st_expired", "st_canceled"]

# shrinking the range to only the cluster counts I "realistically" need
DEFAULT_COMPONENT_GRID = [8, 12, 16, 24, 32]
DEFAULT_QUANTILE_LEVELS = np.array([0.0, 0.5, 0.9, 0.95, 0.99, 0.995, ], dtype=float)
BOXCOX_SHIFT = 1.0
CALIBRATION_COLS = [
    "st_bytes_xfered",
    "st_xfer_time_ms",
]
DEFAULT_REG_COVAR = 5e-3

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    cache_path = csv_path.with_suffix(".parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.rename(columns={df_raw.columns[0]: "record_id"})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(cache_path)
    return df_raw.copy()

def prepare_boxcox_features(df: pd.DataFrame, feature_cols, boxcox_shift: float):
    X = df[feature_cols].copy()
    X_boxcox_input = X + boxcox_shift

    constant_cols = X_boxcox_input.columns[X_boxcox_input.nunique() <= 1].tolist()
    if constant_cols:
        print(f"Skipping constant columns for Box-Cox: {constant_cols}")
    constant_values = {col: X[col].iloc[0] for col in constant_cols}

    train_cols = [col for col in feature_cols if col not in constant_cols]
    if not train_cols:
        raise ValueError("No features left after removing constant columns.")

    transformer = PowerTransformer(method="box-cox", standardize=True)
    X_boxcox = transformer.fit_transform(X_boxcox_input[train_cols])
    X_boxcox = pd.DataFrame(X_boxcox, columns=train_cols, index=df.index)
    return X_boxcox, transformer, constant_cols, constant_values


def fit_best_gmm(
    X_boxcox: pd.DataFrame,
    component_grid=DEFAULT_COMPONENT_GRID,
    random_state=42,
    reg_covar: float = DEFAULT_REG_COVAR,
):
    bic_records = []
    best_model = None
    best_bic = np.inf
    component_iter = iter_progress(component_grid, desc="CPU GMM components")
    for n in component_iter:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=random_state,
            reg_covar=reg_covar,
            max_iter=500,
        )
        gmm.fit(X_boxcox)
        bic = gmm.bic(X_boxcox)
        bic_records.append({"n_components": n, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    if best_model is None:
        raise RuntimeError("GMM fitting failed to produce a model.")
    bic_df = pd.DataFrame(bic_records)
    return best_model, bic_df

def quantile_calibrate(source_values, target_values, quantile_levels=None):
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if source.size == 0:
        return source
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS
    else:
        quantile_levels = np.asarray(quantile_levels, dtype=float)
    quantile_levels = np.unique(np.clip(quantile_levels, 0.0, 1.0))
    quantile_levels.sort()
    source_q = np.maximum.accumulate(np.quantile(source, quantile_levels))
    target_q = np.maximum.accumulate(np.quantile(target, quantile_levels))
    eps = 1e-6
    for i in range(1, len(source_q)):
        if source_q[i] <= source_q[i - 1]:
            source_q[i] = source_q[i - 1] + eps
        if target_q[i] <= target_q[i - 1]:
            target_q[i] = target_q[i - 1] + eps
    calibrated = np.interp(source, source_q, target_q, left=target_q[0], right=target_q[-1])
    tail_ratio = (target_q[-1] + eps) / (source_q[-1] + eps)
    higher_mask = source > source_q[-1]
    if higher_mask.any():
        calibrated[higher_mask] = target_q[-1] + (source[higher_mask] - source_q[-1]) * tail_ratio
    lower_mask = source < source_q[0]
    if lower_mask.any():
        if len(source_q) > 1:
            slope = (target_q[1] - target_q[0]) / max(source_q[1] - source_q[0], eps)
        else:
            slope = tail_ratio
        calibrated[lower_mask] = target_q[0] + (source[lower_mask] - source_q[0]) * slope
    return calibrated


def invert_latent_samples(latent_samples, transformer, feature_names, boxcox_shift: float):
    latent_df = pd.DataFrame(latent_samples, columns=feature_names)
    positive = transformer.inverse_transform(latent_df.values)
    positive = pd.DataFrame(positive, columns=feature_names, index=latent_df.index)
    synthetic = positive - boxcox_shift
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan)
    synthetic = synthetic.fillna(0)
    synthetic = synthetic.clip(lower=0)
    return synthetic


def apply_count_constraints(synthetic_features: pd.DataFrame):
    present_counts = [col for col in COUNT_LIKE_COLS if col in synthetic_features.columns]
    if present_counts:
        rounded = synthetic_features[present_counts].round().clip(lower=0)
        synthetic_features[present_counts] = rounded.astype(int)

    status_cols = [col for col in STATUS_COLS if col in synthetic_features.columns]
    totals = None
    if status_cols:
        totals = synthetic_features[status_cols].sum(axis=1)
    if "st_dirs" in synthetic_features.columns:
        dirs_series = synthetic_features["st_dirs"]
        totals = dirs_series if totals is None else totals + dirs_series
    if totals is None:
        totals = pd.Series(0, index=synthetic_features.index)
    synthetic_features["st_files"] = totals.clip(lower=0).round().astype(int)

    if {"st_dirs", "st_files"}.issubset(synthetic_features.columns):
        dirs = synthetic_features["st_dirs"].astype(int)
        files = synthetic_features["st_files"].astype(int)
        synthetic_features["st_dirs"] = np.minimum(dirs, files).astype(int)


def apply_constant_columns(synthetic_features: pd.DataFrame, constant_cols, constant_values):
    for col in constant_cols:
        synthetic_features[col] = constant_values[col]


def assemble_dataset(df: pd.DataFrame, synthetic_features: pd.DataFrame):
    synthetic_dataset = df.copy()
    for col in synthetic_features.columns:
        if col in synthetic_dataset.columns:
            synthetic_dataset[col] = synthetic_features[col]
    synthetic_dataset = synthetic_dataset[df.columns]
    return synthetic_dataset

def calibrate_feature_column(synthetic_features: pd.DataFrame, df: pd.DataFrame, col: str, quantile_levels):
    if col in synthetic_features.columns and col in df.columns:
        target_values = df[col].astype(float).values
        source_values = synthetic_features[col].astype(float).values
        calibrated = quantile_calibrate(source_values, target_values, quantile_levels=quantile_levels)
        synthetic_features[col] = np.clip(calibrated, 0, None)

def generate_synthetic_dataset(
    data_path: Path,
    output_path: Path,
    component_grid=DEFAULT_COMPONENT_GRID,
    quantile_levels=DEFAULT_QUANTILE_LEVELS,
    use_gpu: bool = False,
    gpu_kwargs: dict | None = None,
):
    log_stage(f"Loading data from {data_path}")
    df = load_dataframe(data_path)
    if use_gpu:
        gpu_kwargs = gpu_kwargs or {}
        gpu_device = gpu_kwargs.get("device", "cuda")
        log_stage("Preparing Box-Cox features on GPU")
        (
            X_boxcox_gpu,
            transformer_gpu,
            constant_cols,
            constant_values,
            feature_names,
        ) = prepare_boxcox_features_gpu(df, FEATURE_COLS, BOXCOX_SHIFT, device=gpu_device)
        log_stage("Training GMM on GPU")
        gpu_args = gpu_kwargs | {"dtype": transformer_gpu.dtype}
        best_gmm, bic_df = fit_best_gmm_gpu(
            X_boxcox_gpu,
            component_grid=component_grid,
            **gpu_args,
        )
    else:
        log_stage("Preparing Box-Cox features")
        X_boxcox, transformer, constant_cols, constant_values = prepare_boxcox_features(df, FEATURE_COLS, BOXCOX_SHIFT)
        feature_names = X_boxcox.columns
        log_stage("Training GMM on CPU")
        best_gmm, bic_df = fit_best_gmm(X_boxcox, component_grid=component_grid, reg_covar=DEFAULT_REG_COVAR)
    print("BIC scores:\n", bic_df)
    print("Selected components:", best_gmm.n_components)

    log_stage("Sampling from fitted GMM")
    latent_samples, labels = best_gmm.sample(len(df))
    if use_gpu:
        log_stage("Inverting Box-Cox transform on GPU")
        synthetic_features = invert_latent_samples_gpu(
            latent_samples,
            transformer_gpu,
            feature_names,
            BOXCOX_SHIFT,
            device=gpu_device,
            index=df.index,
        )
    else:
        log_stage("Inverting Box-Cox transform")
        synthetic_features = invert_latent_samples(latent_samples, transformer, feature_names, BOXCOX_SHIFT)

    log_stage("Calibrating feature quantiles")
    for col in CALIBRATION_COLS:
        calibrate_feature_column(synthetic_features, df, col, quantile_levels)
    log_stage("Applying count constraints")
    apply_count_constraints(synthetic_features)
    log_stage("Restoring constant columns")
    apply_constant_columns(synthetic_features, constant_cols, constant_values)

    synthetic_features["cluster"] = labels
    log_stage("Assembling final dataset")
    synthetic_dataset = assemble_dataset(df, synthetic_features)

    log_stage(f"Writing synthetic data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_dataset.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")

def main():
    args = parse_args()
    data_path = Path(args.input)
    output_path = Path(args.output)
    gpu_kwargs = None
    if args.use_gpu:
        gpu_kwargs = {
            "device": args.gpu_device,
            "max_iter": args.gpu_max_iter,
            "n_init": args.gpu_n_init,
            "tol": args.gpu_tol,
            "batch_size": args.gpu_batch_size,
            "reg_covar": args.gpu_reg_covar,
            "use_kmeans_init": not args.gpu_disable_kmeans_init,
            "kmeans_iters": args.gpu_kmeans_iters,
        }
    generate_synthetic_dataset(
        data_path,
        output_path,
        use_gpu=args.use_gpu,
        gpu_kwargs=gpu_kwargs,
    )

if __name__ == "__main__":
    main()
