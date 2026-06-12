"""Plotting utilities for the validation notebooks.

Provides scatter plots with asymmetric error regions or error bars, guide-line
helpers for comparison plots, and a figure-saving helper.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cfspopcon.unit_handling import magnitude, magnitude_in_units
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def plot_with_error_regions(x_ds, x_key, x_units, y_ds, y_key, y_units, mask=None, ax=None, label=None, color="C0", face_alpha=0.2, edge_alpha=0.3):
    """Scatter plot with an asymmetric error ellipse drawn around each point.

    Errors are read from ``<key>_err`` (symmetric) or ``<key>_err_neg``/``<key>_err_pos``
    (asymmetric) variables of the corresponding dataset.
    """
    if ax is None:
        _fig, ax = plt.subplots()
    if mask is None:
        mask = np.ones_like(magnitude(x_ds[x_key]))

    # 1. Extract values and apply mask
    # We convert to numpy arrays immediately to make indexing and iteration cleaner
    x_vals = magnitude_in_units(x_ds[x_key], x_units).where(mask).values
    y_vals = magnitude_in_units(y_ds[y_key], y_units).where(mask).values

    face_rgba = mcolors.to_rgba(color, alpha=face_alpha)
    edge_rgba = mcolors.to_rgba(color, alpha=edge_alpha)

    # 2. Define all 4 error components neatly
    x_err_neg, x_err_pos = get_error_components(x_ds, x_key, x_units, mask)
    y_err_neg, y_err_pos = get_error_components(y_ds, y_key, y_units, mask)

    # 3. Iterate through data points
    for i in range(len(x_vals)):
        # Skip NaNs (produced by the mask or missing data)
        if np.isnan(x_vals[i]) or np.isnan(y_vals[i]):
            continue

        path = get_asymmetric_ellipse_path(
            x_vals[i],
            y_vals[i],
            x_err_pos[i],
            x_err_neg[i],
            y_err_neg[i],
            y_err_pos[i]
        )

        patch = PathPatch(path, facecolor=face_rgba, edgecolor=edge_rgba)
        ax.add_patch(patch)

    ax.scatter(x_vals, y_vals, s=10, color=color, label=label)

    return ax


def plot_with_errorbars(x_ds, x_key, x_units, y_ds, y_key, y_units, mask=None, ax=None, capsize=3, **kwargs):
    """Scatter plot with (possibly asymmetric) error bars via matplotlib errorbar."""
    if ax is None:
        _fig, ax = plt.subplots()
    if mask is None:
        mask = np.ones_like(magnitude(x_ds[x_key]))

    x_vals = magnitude_in_units(x_ds[x_key], x_units).where(mask).values
    y_vals = magnitude_in_units(y_ds[y_key], y_units).where(mask).values

    x_err_neg, x_err_pos = get_error_components(x_ds, x_key, x_units, mask)
    y_err_neg, y_err_pos = get_error_components(y_ds, y_key, y_units, mask)

    return ax.errorbar(
        x = x_vals,
        xerr = np.vstack((x_err_neg, x_err_pos)),
        y = y_vals,
        yerr = np.vstack((y_err_neg, y_err_pos)),
        fmt = "o",
        capsize=capsize,
        **kwargs
    )


def plot_error_regions_by_category(
    ax,
    x_ds,
    x_key,
    x_units,
    y_ds,
    y_key,
    y_units,
    categories: dict,
    category_data: xr.DataArray,
    extra_mask=None,
    plot_func=plot_with_error_regions,
):
    """Draw one error-region series per category, colored C0..Cn and labelled from the dict.

    Args:
        ax: axes to plot into.
        x_ds: dataset holding the x values.
        x_key: variable name for the x values.
        x_units: units for the x values.
        y_ds: dataset holding the y values.
        y_key: variable name for the y values.
        y_units: units for the y values.
        categories: mapping of category value -> legend label, in plot order.
        category_data: per-point category values, compared against the dict keys.
        extra_mask: additional mask applied to every series (e.g. a converged flag).
        plot_func: series plotting function (plot_with_error_regions or plot_with_errorbars).
    """
    for i, (category, label) in enumerate(categories.items()):
        mask = category_data == category
        if extra_mask is not None:
            mask = mask & extra_mask
        plot_func(
            ax=ax,
            x_ds=x_ds,
            x_key=x_key,
            x_units=x_units,
            y_ds=y_ds,
            y_key=y_key,
            y_units=y_units,
            mask=mask,
            label=label,
            color=f"C{i}",
        )
    return ax


def add_identity_guides(ax, slopes=(1.0, 0.5, 2.0), color="k", linewidth=None):
    """Add through-origin guide lines: solid for slope 1, dashed for the other slopes."""
    for slope in slopes:
        linestyle = "-" if slope == 1.0 else "--"
        ax.axline((0, 0), slope=slope, color=color, linestyle=linestyle, linewidth=linewidth)
    return ax


def add_ratio_guides(ax, levels=(1.0, 0.5, 2.0), color="k", linewidth=None):
    """Add horizontal guide lines: solid at 1, dashed at the other levels."""
    for level in levels:
        linestyle = "-" if level == 1.0 else "--"
        ax.axhline(level, color=color, linestyle=linestyle, linewidth=linewidth)
    return ax


def get_error_components(ds, key, units, mask):
    """Helper to return (neg_err, pos_err) arrays regardless of input type."""
    if f"{key}_err" in ds:
        err = np.abs(magnitude_in_units(ds[f"{key}_err"], units).where(mask).values)
        return err, err  # Symmetric: neg and pos are the same

    elif f"{key}_err_neg" in ds and f"{key}_err_pos" in ds:
        err_neg = np.abs(magnitude_in_units(ds[f"{key}_err_neg"], units).where(mask).values)
        err_pos = np.abs(magnitude_in_units(ds[f"{key}_err_pos"], units).where(mask).values)
        return err_neg, err_pos

    else:
        print(f"No error found for {key}")
        # Fallback for no error: array of zeros
        zeros = np.zeros_like(ds[key])
        return zeros, zeros


def get_asymmetric_ellipse_path(x, y, xerr_pos, xerr_neg, yerr_neg, yerr_pos):
    """Create a Path for an asymmetric ellipse.

    Handles independent errors for +x, -x, +y, and -y.
    """
    # Constant for drawing a circle/ellipse with 4 Bezier curves
    MAGIC = 0.552284749831

    path_data = [
        (Path.MOVETO, (x + xerr_pos, y)),
        # Curve to Top Center (+y)
        (Path.CURVE4, (x + xerr_pos, y + yerr_pos * MAGIC)),
        (Path.CURVE4, (x + xerr_pos * MAGIC, y + yerr_pos)),
        (Path.CURVE4, (x, y + yerr_pos)),
        # Curve to Left Center (-x)
        (Path.CURVE4, (x - xerr_neg * MAGIC, y + yerr_pos)),
        (Path.CURVE4, (x - xerr_neg, y + yerr_pos * MAGIC)),
        (Path.CURVE4, (x - xerr_neg, y)),
        # Curve to Bottom Center (-y)
        (Path.CURVE4, (x - xerr_neg, y - yerr_neg * MAGIC)),
        (Path.CURVE4, (x - xerr_neg * MAGIC, y - yerr_neg)),
        (Path.CURVE4, (x, y - yerr_neg)),
        # Curve to Right Center (+x)
        (Path.CURVE4, (x + xerr_pos * MAGIC, y - yerr_neg)),
        (Path.CURVE4, (x + xerr_pos, y - yerr_neg * MAGIC)),
        (Path.CURVE4, (x + xerr_pos, y)),
        (Path.CLOSEPOLY, (x + xerr_pos, y))
    ]

    codes, verts = zip(*path_data, strict=True)
    return Path(verts, codes)


def save_with_transp(output_folder, name, dpi=300):
    """Save the current figure twice: opaque ``<name>.png`` and transparent ``<name>_transp.png``."""
    for suffix, transparent in (("", False), ("_transp", True)):
        plt.savefig(output_folder / f"{name}{suffix}.png", dpi=dpi, transparent=transparent, bbox_inches="tight")
