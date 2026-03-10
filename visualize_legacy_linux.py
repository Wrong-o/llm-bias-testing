"""Compare legacy Linux personas: sysadmin vs beginner."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#ffffff"
GRID_COLOR = "#2a2a4e"

PERSONAS = ["linux_sysadmin", "linux_beginner"]
COLORS = {"linux_sysadmin": "#51cf66", "linux_beginner": "#ffd43b"}
LABELS = {"linux_sysadmin": "Linux Sysadmin", "linux_beginner": "Linux Beginner"}

# Hand-picked metrics that show the biggest behavioral differences
COMPARE_METRICS = [
    "code_lines", "sudo_count", "inline_comment_count",
    "step_count", "apt_mentions", "pacman_mentions",
    "gui_tool_mentions", "exclamation_count",
]

METRIC_LABELS = {
    "code_lines": "Code Lines",
    "sudo_count": "sudo Uses",
    "inline_comment_count": "Inline Comments",
    "step_count": "Step-by-Step\nInstructions",
    "apt_mentions": "apt Mentions",
    "pacman_mentions": "pacman Mentions",
    "gui_tool_mentions": "GUI Tool\nMentions",
    "exclamation_count": "Exclamation\nMarks",
}


def main():
    df = pd.read_csv("results/combined_scores.csv")
    df = df[df["persona"].isin(PERSONAS)]

    means = df.groupby("persona")[COMPARE_METRICS].mean()
    sems = df.groupby("persona")[COMPARE_METRICS].sem()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    x = np.arange(len(COMPARE_METRICS))
    width = 0.35

    for i, persona in enumerate(PERSONAS):
        vals = means.loc[persona, COMPARE_METRICS].values.astype(float)
        errs = sems.loc[persona, COMPARE_METRICS].values.astype(float) * 1.96
        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, vals, width, yerr=errs,
            color=COLORS[persona], label=LABELS[persona],
            alpha=0.9, capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#ffffff50"},
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT_COLOR,
            )

    labels = [METRIC_LABELS.get(m, m.replace("_", " ").title()) for m in COMPARE_METRICS]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Value per Response", fontsize=13)
    ax.set_title(
        "Sysadmin vs Beginner: How Mistral Adapts to Linux Expertise Level",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    legend = ax.legend(fontsize=12, framealpha=0.3, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    out_path = "results/legacy_linux_comparison.png"
    fig.savefig(out_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
