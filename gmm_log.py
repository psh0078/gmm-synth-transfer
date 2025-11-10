from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_bytes_xfered",
    "st_faults",
    "st_skipped_errors",
]

COUNT_LIKE_COLS = [
    "st_dirs",
    "st_successful",
    "st_failed",
    "st_expired",
    "st_canceled",
    "st_bytes_xfered",
    "st_faults",
]


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.rename(columns={df_raw.columns[0]: "record_id"})
    return df_raw.copy()


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    log_shift = 1.0  # ensures inputs stay positive before log1p
    X_log = np.log1p(X)

    constant_cols = X_log.columns[X_log.nunique() <= 1].tolist()
    constant_values = {col: X[col].iloc[0] for col in constant_cols}

    train_cols = [col for col in FEATURE_COLS if col not in constant_cols]
    if not train_cols:
        raise ValueError("No non-constant features available for modelling.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log[train_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=train_cols, index=df.index)

    return X_scaled, scaler, train_cols, constant_cols, constant_values


def select_gmm_components(X_scaled: pd.DataFrame, component_grid=range(1, 9)):
    bic_records = []
    best_model = None
    best_bic = np.inf
    for n in component_grid:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=42,
        )
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        bic_records.append({"n_components": n, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    bic_df = pd.DataFrame(bic_records)
    return best_model, bic_df


def invert_samples(
    latent_samples: np.ndarray,
    scaler: StandardScaler,
    train_cols,
    constant_cols,
    constant_values,
    index,
):
    synthetic_scaled = pd.DataFrame(latent_samples, columns=train_cols, index=index)
    synthetic_log = scaler.inverse_transform(synthetic_scaled)
    synthetic_log = pd.DataFrame(synthetic_log, columns=train_cols, index=index)
    synthetic_features = pd.DataFrame(np.expm1(synthetic_log), columns=train_cols, index=index)

    for col in constant_cols:
        synthetic_features[col] = constant_values[col]

    synthetic_features = synthetic_features.replace([np.inf, -np.inf], np.nan)
    synthetic_features = synthetic_features.fillna(0)
    synthetic_features = synthetic_features.clip(lower=0)

    present_counts = [col for col in COUNT_LIKE_COLS if col in synthetic_features.columns]
    if present_counts:
        rounded = synthetic_features[present_counts].round().clip(lower=0)
        synthetic_features[present_counts] = rounded.astype(int)

    status_cols = [col for col in ["st_successful", "st_failed", "st_expired", "st_canceled"] if col in synthetic_features.columns]
    totals = None
    if status_cols:
        totals = synthetic_features[status_cols].sum(axis=1)
    if "st_dirs" in synthetic_features.columns:
        dirs_series = synthetic_features["st_dirs"]
        totals = dirs_series if totals is None else totals + dirs_series
    if totals is None:
        totals = pd.Series(0, index=index)
    synthetic_features["st_files"] = totals.clip(lower=0).round().astype(int)

    if {"st_dirs", "st_files"}.issubset(synthetic_features.columns):
        dirs = synthetic_features["st_dirs"].astype(int)
        files = synthetic_features["st_files"].astype(int)
        synthetic_features["st_dirs"] = np.minimum(dirs, files)
        synthetic_features["st_dirs"] = synthetic_features["st_dirs"].astype(int)

    return synthetic_features


def main():
    data_path = Path("og-transfer.csv")
    output_path = Path("output/output_log.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_path)
    X_scaled, scaler, train_cols, constant_cols, constant_values = prepare_features(df)

    best_gmm, bic_df = select_gmm_components(X_scaled)
    print("BIC scores:\n", bic_df)
    print("Selected components:", best_gmm.n_components)

    latent_samples, labels = best_gmm.sample(len(df))
    synthetic_features = invert_samples(
        latent_samples,
        scaler,
        train_cols,
        constant_cols,
        constant_values,
        index=df.index,
    )
    synthetic_features["cluster"] = labels

    synthetic_dataset = df.copy()
    for col in synthetic_features.columns:
        if col in synthetic_dataset.columns:
            synthetic_dataset[col] = synthetic_features[col]
    synthetic_dataset = synthetic_dataset[df.columns]

    synthetic_dataset.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")


if __name__ == "__main__":
    main()
