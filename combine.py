"""Merge hard and soft scores into a single CSV with derived columns."""

import pandas as pd


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def main() -> None:
    hard = pd.read_csv("results/hard_scores.csv")
    soft = pd.read_csv("results/soft_scores.csv")

    soft = soft.drop(columns=["persona", "prompt_idx"], errors="ignore")
    combined = hard.merge(soft, on="custom_id", how="inner")

    # Normalize components (min-max across all rows)
    norm_condescension = min_max_normalize(combined["condescension_count"])
    norm_hedge = min_max_normalize(combined["hedge_count"])
    norm_warning = min_max_normalize(combined["warning_count"])
    norm_hand_holding = min_max_normalize(combined["hand_holding"])

    # Condescension index: simple average of 4 normalized components
    combined["condescension_index"] = (
        norm_condescension + norm_hedge + norm_warning + norm_hand_holding
    ) / 4

    combined.to_csv("results/combined_scores.csv", index=False)

    print(f"Rows: {len(combined)}, Columns: {len(combined.columns)}")
    print(f"Wrote results/combined_scores.csv")


if __name__ == "__main__":
    main()
