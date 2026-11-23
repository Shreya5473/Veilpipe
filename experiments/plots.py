"""
experiments/plots.py  –  shared plotting utilities

Helper functions used by multiple experiment scripts.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Colour palette used across all experiments
SCHEME_COLORS = {
    "BTE2024": "#457b9d",
    "OTS2025": "#2a9d8f",
    "BEATMEV": "#e63946",
}
SCHEME_MARKERS = {
    "BTE2024": "o",
    "OTS2025": "s",
    "BEATMEV": "^",
}

BLOCK_COLORS = {250: "#e63946", 500: "#457b9d", 1000: "#2a9d8f", 2000: "#e9c46a"}


def save_fig(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """Save figure to RESULTS_DIR/<filename> and return the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  → {path}")
    return path


def line_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    colors: dict,
    markers: dict,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
) -> None:
    """
    Draw one line per group value onto *ax*.

    Parameters
    ----------
    ax         : matplotlib axes
    df         : data
    x_col      : column name for x-axis
    y_col      : column name for y-axis (mean / central value)
    group_col  : column used to split lines
    colors     : {group_value → colour}
    markers    : {group_value → marker}
    ci_low_col : optional column for lower CI bound (draws fill_between)
    ci_high_col: optional column for upper CI bound
    """
    for grp, sub in df.groupby(group_col):
        sub = sub.sort_values(x_col)
        c = colors.get(grp, "#555555")
        m = markers.get(grp, "o")
        ax.plot(sub[x_col], sub[y_col], color=c, marker=m,
                label=str(grp), linewidth=2, markersize=7)
        if ci_low_col and ci_high_col:
            ax.fill_between(sub[x_col], sub[ci_low_col], sub[ci_high_col],
                            color=c, alpha=0.15)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)


def heatmap_plot(
    ax: plt.Axes,
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cell_text: list[list[str]] | None = None,
) -> None:
    """
    Draw a heatmap on *ax*.

    Parameters
    ----------
    matrix     : 2-D list of float values
    row_labels : labels for each row
    col_labels : labels for each column
    cell_text  : optional 2-D list of strings to overlay on cells
    """
    import numpy as np
    mat = np.array(matrix, dtype=float)
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(c) for c in col_labels])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if title:
        ax.set_title(title, fontweight="bold")
    if cell_text:
        for i, row in enumerate(cell_text):
            for j, txt in enumerate(row):
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color="black")


__all__ = [
    "save_fig",
    "line_plot",
    "heatmap_plot",
    "SCHEME_COLORS",
    "SCHEME_MARKERS",
    "BLOCK_COLORS",
    "RESULTS_DIR",
]
