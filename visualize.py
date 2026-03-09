"""Generate 5 publication-quality charts from experiment results."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONAS = ["macos_dev", "windows_dev", "linux_sysadmin", "linux_beginner", "unspecified"]

PERSONA_COLORS = {
    "macos_dev": "#00d4ff",
    "windows_dev": "#ff6b6b",
    "linux_sysadmin": "#51cf66",
    "linux_beginner": "#ffd43b",
    "unspecified": "#868e96",
}

PERSONA_LABELS = {
    "macos_dev": "macOS Dev",
    "windows_dev": "Windows Dev",
    "linux_sysadmin": "Linux Sysadmin",
    "linux_beginner": "Linux Beginner",
    "unspecified": "Unspecified",
}

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#ffffff"
GRID_COLOR = "#2a2a4e"

SOFT_SCORES = [
    "assumed_competence",
    "condescension",
    "hand_holding",
    "confidence",
    "verbosity",
    "professionalism",
]

HARD_METRICS = [
    "response_length", "code_lines", "explanation_lines", "code_to_explanation_ratio",
    "preamble_length", "warning_count", "hedge_count", "condescension_count",
    "inline_comment_count", "emoji_count", "exclamation_count", "docker_mentions",
    "wsl_mention", "sudo_count", "gui_tool_mentions", "brew_mentions",
    "apt_mentions", "pacman_mentions", "choco_mentions", "link_count",
    "step_count", "heading_count", "bullet_count",
]

FIGSIZE = (12, 6.28)
DPI = 150


def _style_ax(ax):
    """Apply dark theme to axes."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=12)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _style_fig(fig):
    """Apply dark theme to figure."""
    fig.patch.set_facecolor(BG_COLOR)


def _save(fig, path):
    """Save and close figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Chart 1: Radar
# ---------------------------------------------------------------------------

def chart_radar(df: pd.DataFrame):
    """Overlaid radar chart of 6 soft scores per persona."""
    means = df.groupby("persona")[SOFT_SCORES].mean()

    # Normalize to 0-1 (scores are on 1-5 scale)
    normed = means / 5.0

    angles = np.linspace(0, 2 * np.pi, len(SOFT_SCORES), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    labels = [s.replace("_", " ").title() for s in SOFT_SCORES]

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw=dict(polar=True))
    _style_fig(fig)
    ax.set_facecolor(BG_COLOR)

    for persona in PERSONAS:
        if persona not in normed.index:
            continue
        values = normed.loc[persona].tolist()
        values += values[:1]
        color = PERSONA_COLORS[persona]
        ax.plot(angles, values, color=color, linewidth=2.2, alpha=0.8,
                label=PERSONA_LABELS[persona])
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=10, color=TEXT_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)

    ax.set_title("LLM Soft-Score Profiles by Persona", fontsize=18, color=TEXT_COLOR,
                 fontweight="bold", pad=20)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=11,
                       framealpha=0.3, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, "results/radar.png")


# ---------------------------------------------------------------------------
# Chart 2: Grouped bars — top 6 most-divergent hard metrics
# ---------------------------------------------------------------------------

def chart_bars_key_metrics(df: pd.DataFrame):
    """Grouped bar chart of the 6 most-divergent hard metrics."""
    persona_means = df.groupby("persona")[HARD_METRICS].mean()
    variances = persona_means.var(axis=0)
    top6 = variances.nlargest(6).index.tolist()

    means = df.groupby("persona")[top6].mean()
    sems = df.groupby("persona")[top6].sem()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _style_fig(fig)
    _style_ax(ax)

    x = np.arange(len(top6))
    n_personas = len(PERSONAS)
    width = 0.15
    offsets = np.linspace(-(n_personas - 1) / 2 * width, (n_personas - 1) / 2 * width, n_personas)

    for i, persona in enumerate(PERSONAS):
        if persona not in means.index:
            continue
        vals = means.loc[persona, top6].values.astype(float)
        errs = (sems.loc[persona, top6].values.astype(float)) * 1.96
        ax.bar(x + offsets[i], vals, width, yerr=errs,
               color=PERSONA_COLORS[persona], label=PERSONA_LABELS[persona],
               alpha=0.9, capsize=2, error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#ffffff50"})

    metric_labels = [m.replace("_", " ").title() for m in top6]
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("Mean Value", fontsize=14)
    ax.set_title("Top 6 Most-Divergent Hard Metrics by Persona", fontsize=18, fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    legend = ax.legend(fontsize=10, framealpha=0.3, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, "results/bars_key_metrics.png")


# ---------------------------------------------------------------------------
# Chart 3: Package manager bias (stacked bars)
# ---------------------------------------------------------------------------

def chart_package_mgr(df: pd.DataFrame):
    """Stacked bar chart of package manager mentions per persona."""
    pkg_cols = ["brew_mentions", "apt_mentions", "pacman_mentions", "choco_mentions"]
    pkg_colors = {"brew_mentions": "#00d4ff", "apt_mentions": "#51cf66",
                  "pacman_mentions": "#ff922b", "choco_mentions": "#be4bdb"}
    pkg_labels = {"brew_mentions": "brew", "apt_mentions": "apt",
                  "pacman_mentions": "pacman", "choco_mentions": "choco"}

    means = df.groupby("persona")[pkg_cols].mean()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _style_fig(fig)
    _style_ax(ax)

    x = np.arange(len(PERSONAS))
    bottom = np.zeros(len(PERSONAS))

    for col in pkg_cols:
        vals = []
        for p in PERSONAS:
            vals.append(means.loc[p, col] if p in means.index else 0)
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, bottom=bottom, color=pkg_colors[col],
               label=pkg_labels[col], alpha=0.9, width=0.55)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=12)
    ax.set_ylabel("Mean Mentions per Response", fontsize=14)
    ax.set_title("Package Manager Bias by Persona", fontsize=18, fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    legend = ax.legend(fontsize=12, framealpha=0.3, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, "results/bars_package_mgr.png")


# ---------------------------------------------------------------------------
# Chart 4: Condescension Index — horizontal bars (hero chart)
# ---------------------------------------------------------------------------

def chart_condescension_index(df: pd.DataFrame):
    """Horizontal bar chart of mean condescension_index per persona, sorted."""
    means = df.groupby("persona")["condescension_index"].mean().reindex(PERSONAS).dropna()
    means_sorted = means.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _style_fig(fig)
    _style_ax(ax)

    y = np.arange(len(means_sorted))
    colors = [PERSONA_COLORS[p] for p in means_sorted.index]

    bars = ax.barh(y, means_sorted.values, color=colors, height=0.6, alpha=0.95,
                   edgecolor="#ffffff20", linewidth=1.5)

    # Value labels
    for bar, val in zip(bars, means_sorted.values):
        ax.text(bar.get_width() + max(means_sorted.values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=14, fontweight="bold", color=TEXT_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels([PERSONA_LABELS[p] for p in means_sorted.index], fontsize=14, fontweight="bold")
    ax.set_xlabel("Condescension Index (mean)", fontsize=14)
    ax.set_title("Condescension Index by Persona", fontsize=20, fontweight="bold")
    ax.set_xlim(0, max(means_sorted.values) * 1.25)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    _save(fig, "results/condescension_index.png")


# ---------------------------------------------------------------------------
# Chart 5: WSL mention rate
# ---------------------------------------------------------------------------

def chart_wsl_rate(df: pd.DataFrame):
    """Vertical bar chart of WSL mention rate per persona."""
    means = df.groupby("persona")["wsl_mention"].mean().reindex(PERSONAS).fillna(0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _style_fig(fig)
    _style_ax(ax)

    x = np.arange(len(PERSONAS))
    colors = [PERSONA_COLORS[p] for p in PERSONAS]
    pct = means.values * 100

    bars = ax.bar(x, pct, color=colors, width=0.55, alpha=0.9,
                  edgecolor="#ffffff20", linewidth=1.5)

    for bar, val in zip(bars, pct):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pct) * 0.02,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONAS], fontsize=12)
    ax.set_ylabel("WSL Mention Rate (%)", fontsize=14)
    ax.set_title("WSL Mention Rate by Persona", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(max(pct) * 1.3, 1))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    _save(fig, "results/wsl_rate.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = "results/combined_scores.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    os.makedirs("results", exist_ok=True)

    print("\nGenerating charts...")
    chart_radar(df)
    chart_bars_key_metrics(df)
    chart_package_mgr(df)
    chart_condescension_index(df)
    chart_wsl_rate(df)
    print("\nDone — 5 charts saved to results/")


if __name__ == "__main__":
    main()
