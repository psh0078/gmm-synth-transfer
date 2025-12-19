from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SDV quality metrics between a real CSV and a synthetic CSV."
    )
    parser.add_argument(
        "--real",
        required=True,
        type=Path,
        help="Path to the real CSV (e.g. datasets/small-transfer.csv).",
    )
    parser.add_argument(
        "--synthetic",
        required=True,
        type=Path,
        help="Path to the synthetic CSV (e.g. output/synth.csv).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional path to a saved SingleTableMetadata JSON file. "
        "If omitted, metadata will be inferred from the real CSV.",
    )
    parser.add_argument(
        "--save-metadata",
        type=Path,
        help="If set (and metadata was inferred), write the detected metadata JSON to this path.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path to dump the summary report JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap applied to both real and synthetic DataFrames before evaluation.",
    )
    parser.add_argument(
        "--column-plot",
        type=str,
        help="Optional column name to visualize via SDV's column plot.",
    )
    parser.add_argument(
        "--column-plot-html",
        type=Path,
        help="If set with --column-plot, save the figure as HTML to this path instead of showing it.",
    )
    parser.add_argument(
        "--column-plot-type",
        type=str,
        help="Optional plot type override for the column visualization (e.g., distplot, bar).",
    )
    return parser.parse_args()


def load_dataframe(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if (
            isinstance(first_col, str)
            and first_col.startswith("Unnamed")
            and "record_id" not in df.columns
        ):
            df = df.rename(columns={first_col: "record_id"})
    if limit is not None:
        df = df.head(limit)
    return df


def load_or_detect_metadata(
    df: pd.DataFrame, metadata_path: Path | None, save_path: Path | None
) -> SingleTableMetadata:
    if metadata_path:
        metadata = SingleTableMetadata.load_from_json(str(metadata_path))
    else:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "overwrite" if save_path.exists() else "write"
            metadata.save_to_json(str(save_path), mode=mode)
    return metadata


def report_to_dict(report) -> dict:
    properties_df = report.get_properties()
    details = {
        row["Property"]: report.get_details(row["Property"]).to_dict(orient="records")
        for row in properties_df.to_dict(orient="records")
    }
    return {
        "overall_score": report.get_score(),
        "properties": properties_df.to_dict(orient="records"),
        "details": details,
    }


def main() -> None:
    args = parse_args()

    real_df = load_dataframe(args.real, args.limit)
    synthetic_df = load_dataframe(args.synthetic, args.limit)
    metadata = load_or_detect_metadata(real_df, args.metadata, args.save_metadata)

    quality_report = evaluate_quality(real_df, synthetic_df, metadata, verbose=True)

    overall = quality_report.get_score()
    print(f"\nOverall SDV quality score: {overall:.4f}")

    properties = quality_report.get_properties()
    print("\nProperty scores:")
    print(properties.to_string(index=False))

    if args.report_json:
        summary = report_to_dict(quality_report)
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nSaved report JSON to {args.report_json}")

if __name__ == "__main__":
    main()
