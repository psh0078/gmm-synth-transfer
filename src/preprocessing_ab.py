import pandas as pd

csv_path = "../datasets/original.csv"
KEEP_COLS = [
    "grp_uuid", "user_id", "request_time", "grp_status", "encrypt_data",
    "grp_delete", "st_files", "st_dirs", "st_successful", "st_failed",
    "st_expired", "st_canceled", "st_bytes_xfered", "st_faults",
    "st_skipped_errors", "src_host_ep_id", "dst_host_ep_id",
    "complete_time", "st_xfer_time_ms",
]

df = pd.read_csv(csv_path, engine="pyarrow")
df = df.filter(items=KEEP_COLS)
print("\nColumns with dtypes:")
for col, dtype in df.dtypes.items():
    print(f"{col}: {dtype}")

def keep_transfer_columns(df):
    df['st_xfer_time_ms'] = pd.to_numeric(df['st_xfer_time_ms'], errors='coerce')
    df_clean = df.dropna(subset=["st_xfer_time_ms"])
    df_clean = df_clean[df_clean['grp_delete'] != True]
    df_clean = df_clean[df_clean['st_files'] != 0]
    missing = set(KEEP_COLS) - set(df_clean.columns)
    if missing:
        raise ValueError(f"DataFrame missing expected columns: {', '.join(sorted(missing))}")
    return df_clean.loc[:, KEEP_COLS].copy()

filtered_df = keep_transfer_columns(df)

filtered_df.to_csv("../datasets/filtered_ab.csv", index=False)

print(filtered_df.describe)
