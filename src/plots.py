"""
Plotting helpers for NILM notebooks.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_power_trace(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "",
    figsize: tuple = (14, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Line plot of power traces."""
    if columns is None:
        columns = list(df.columns)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=0.8)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_comparison(
    true: pd.Series,
    pred: pd.Series,
    appliance: str = "",
    figsize: tuple = (14, 3),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay ground-truth and prediction for one appliance."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(true.index, true.values, label="Ground truth", linewidth=0.8, alpha=0.8)
    ax.plot(pred.index, pred.values, label="Prediction",   linewidth=0.8, alpha=0.8, linestyle="--")
    ax.set_ylabel("Power (W)")
    ax.set_title(appliance)
    ax.legend(fontsize=8)
    return ax


def plot_missing_heatmap(
    df: pd.DataFrame,
    title: str = "Missing data",
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """Heatmap of NaN locations (time × column)."""
    fig, ax = plt.subplots(figsize=figsize)
    missing = df.isna().astype(int).T
    ax.imshow(missing.values, aspect="auto", interpolation="nearest", cmap="Reds")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xlabel("Time index")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_state_histogram(
    series: pd.Series,
    state_defs: Dict,
    appliance: str = "",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Power histogram with state mean lines."""
    fig, ax = plt.subplots(figsize=figsize)
    vals = series.dropna().values
    ax.hist(vals, bins=100, density=True, alpha=0.6, color="steelblue")
    colors = plt.cm.tab10.colors
    for i, (mean, label) in enumerate(zip(state_defs["means"], state_defs["labels"])):
        ax.axvline(mean, color=colors[i % len(colors)], linestyle="--", label=f"{label} ({mean:.0f} W)")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Density")
    ax.set_title(f"{appliance} power distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    metric: str = "NRMSE",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Bar chart of a single metric across appliances / houses."""
    fig, ax = plt.subplots(figsize=figsize)
    if "house" in metrics_df.columns:
        pivot = metrics_df.pivot(index="appliance", columns="house", values=metric)
        pivot.plot(kind="bar", ax=ax)
    else:
        metrics_df.set_index("appliance")[metric].plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per appliance")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
