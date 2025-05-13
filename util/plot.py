import torch
import random
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from textwrap import wrap

color_list = [
    "crimson",
    "orange",
    "cornflowerblue",
    "blueviolet",
    "seagreen",
]
hatch_list = ["o", "x", "/", ".", "*", "-", "\\"]
linestyle_list = [
    "-",
    "--",
    "--",
    "-.",
    ":",
]
markertype_list = ["o", "^", "x", "x", "|"]


def plot_cdf(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label="CDF (%)",
    log_switch=False,
    rotate_xaxis=False,
    ylim_low=0,
    xlim=None,
    xlim_bottom=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    title=None,
    enable_abs=False,
    group_size=1,
    colors=None,
    fig_idx=0,
    fig_size=(5, 2.3),
):
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=fig_size)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if not colors:
        colors = color_list

    # plt.axhline(99, color="k", linewidth=3, linestyle="--", zorder=0)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
        ax.set_xscale("log")

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)
    linelabels = ["\n".join(wrap(l, 30)) for l in linelabels]
    for i in range(len(raw_data)):
        data = raw_data[i]
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue
        if enable_abs:
            data = abs(data)
        # data=random.sample(data,min(1e6,len(data)))
        data_size = len(data)
        # data=list(filter(lambda score: 0<=score < std_val, data))
        # Set bins edges
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)

        # Use the histogram function to bin the data
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        counts = counts.astype(float) / data_size

        # Find the cdf
        cdf = np.cumsum(counts)
        cdf = 100 * cdf / cdf[-1]
        # Plot the cdf
        if i < len(linelabels):
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=colors[(i % group_size) % len(colors)],
                label=linelabels[i],
                linewidth=2,
            )
        else:
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=colors[(i % group_size) % len(colors)],
                linewidth=2,
            )

    if len(linelabels) > 0:
        legend_properties = {"size": legend_font}
        plt.legend(
            prop=legend_properties,
            frameon=False,
            loc=loc,
        )

    plt.ylim((ylim_low, 100))
    if xlim_bottom:
        plt.xlim(left=xlim_bottom)
    if xlim:
        plt.xlim(right=xlim)
    # plt.tight_layout()
    # plt.tight_layout(pad=0.5, w_pad=0.04, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    plt.grid(True)

    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_bars(
    data,
    x_labels,
    file_name,
    bar_labels,
    x_label,
    y_label,
    figsize=(5, 2.3),
    fontsize=15,
    legend_font=15,
    loc=2,
    ylim=None,
    ylim_bottom=None,
    legend_cols=1,
    title=None,
    colors=None,
    patterns=None,  # Added parameter for bar patterns
    edgecolor="black",  # Added parameter for bar edge color
    width=0.25,
    fig_idx=0,
    log_switch=False,
):
    """
    Plots grouped bar charts for the given datasets.

    Parameters:
    - data: List of lists containing bar heights for each group (scenario).
    - x_labels: Labels for the x-axis categories (slowdown ranges).
    - file_name: Name of the file to save the plot.
    - bar_labels: Labels for each bar group (scenarios).
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - figsize: Size of the figure.
    - fontsize: Font size for axis labels and title.
    - legend_font: Font size for the legend.
    - loc: Legend location.
    - ylim: Tuple specifying y-axis limits.
    - title: Title of the plot.
    - colors: List of colors for the bars.
    - patterns: List of patterns for the bars (optional).
    - edgecolor: Color of the edges of the bars (default: black).
    - width: Width of each bar.
    - log_switch: Whether to use logarithmic scaling for the y-axis.
    """
    colors = colors if colors else color_list
    x = np.arange(len(x_labels))

    fig = plt.figure(fig_idx, figsize=figsize)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for idx, scenario_data in enumerate(data):
        ax.bar(
            x + idx * width,
            scenario_data,
            width,
            color=colors[idx % len(colors)],
            edgecolor=edgecolor,
            hatch=patterns[idx % len(patterns)] if patterns else None,  # Apply patterns
            label=bar_labels[idx] if idx < len(bar_labels) else None,
        )

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x + (width * (len(data) - 1)) / 2, x_labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(prop={"size": legend_font}, frameon=False, loc=loc, ncol=legend_cols)
    plt.grid(True)

    if ylim:
        plt.ylim(top=ylim)
    if ylim_bottom:
        plt.ylim(bottom=ylim_bottom)

    if log_switch:
        ax.set_yscale("log")

    if title:
        plt.title(title, fontsize=fontsize)

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_lines(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label,
    log_switch=False,
    log_switch_x=False,
    rotate_xaxis=False,
    ylim=None,
    xlim=None,
    ylim_bottom=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    legend_cols=1,
    title=None,
    format_idx=None,
    fig_idx=0,
    linewidth=2,
    colors=None,
    fig_size=(5, 2.3),
):
    """
    Plots multiple line plots for the given datasets.

    Parameters:
    - raw_data: List of datasets (each dataset is a tuple of x and y values).
    - file_name: Name of the file to save the plot.
    - linelabels: List of labels for the lines.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - log_switch: Whether to use logarithmic scaling on the x-axis.
    - rotate_xaxis: Whether to rotate x-axis tick labels.
    - ylim: Tuple specifying y-axis limits.
    - xlim: Tuple specifying x-axis limits.
    - fontsize: Font size for axis labels and title.
    - legend_font: Font size for the legend.
    - loc: Legend location.
    - title: Title of the plot.
    - fig_idx: Figure index for the plot.
    """
    colors = colors if colors else color_list
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
        ax.set_yscale("log")
    if log_switch_x:
        ax.set_xscale("log")

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)
    linelabels = ["\n".join(wrap(l, 30)) for l in linelabels]

    for i in range(len(raw_data)):
        x, y = raw_data[i]
        if len(x) == 0 or len(y) == 0:
            continue
        idx_selected = format_idx[i] if format_idx else i
        if i < len(linelabels):
            plt.plot(
                x,
                y,
                # linestyle=linestyle_list[idx_selected % len(linestyle_list)],
                color=colors[idx_selected % len(colors)],
                label=linelabels[i],
                linewidth=linewidth,
            )
        else:
            plt.plot(
                x,
                y,
                # linestyle=linestyle_list[idx_selected % len(linestyle_list)],
                color=colors[idx_selected % len(colors)],
                linewidth=linewidth,
            )

    if len(linelabels) > 0:
        legend_properties = {"size": legend_font}
        plt.legend(
            prop=legend_properties,
            frameon=False,
            loc=loc,
            ncol=legend_cols,
        )

    if ylim:
        plt.ylim(top=ylim)
    if ylim_bottom:
        plt.ylim(bottom=ylim_bottom)
    if xlim:
        plt.xlim(right=xlim)

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    plt.grid(True)

    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_scatter(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label,
    log_switch=False,
    log_switch_x=False,
    rotate_xaxis=False,
    ylim=None,
    xlim=None,
    ylim_bottom=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    legend_cols=1,
    title=None,
    format_idx=None,
    fig_idx=0,
    marker_size=50,
    colors=None,
    fig_size=(5, 2.3),
):
    """
    Plots multiple scatter plots for the given datasets.

    Parameters:
    - raw_data: List of datasets (each dataset is a tuple of x and y values).
    - file_name: Name of the file to save the plot.
    - linelabels: List of labels for the scatter plots.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - log_switch: Whether to use logarithmic scaling on the y-axis.
    - log_switch_x: Whether to use logarithmic scaling on the x-axis.
    - rotate_xaxis: Whether to rotate x-axis tick labels.
    - ylim: Tuple specifying y-axis limits.
    - xlim: Tuple specifying x-axis limits.
    - fontsize: Font size for axis labels and title.
    - legend_font: Font size for the legend.
    - loc: Legend location.
    - title: Title of the plot.
    - fig_idx: Figure index for the plot.
    """
    colors = colors if colors else color_list
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
        ax.set_yscale("log")
    if log_switch_x:
        ax.set_xscale("log")

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(x_label, fontsize=_fontsize)
    linelabels = ["\n".join(wrap(l, 30)) for l in linelabels]

    for i in range(len(raw_data)):
        x, y = raw_data[i]
        if len(x) == 0 or len(y) == 0:
            continue
        idx_selected = format_idx[i] if format_idx else i
        if i < len(linelabels):
            plt.scatter(
                x,
                y,
                color=colors[idx_selected % len(colors)],
                label=linelabels[i],
                s=marker_size,
                marker=markertype_list[idx_selected % len(markertype_list)],
            )
        else:
            plt.scatter(
                x,
                y,
                color=colors[idx_selected % len(colors)],
                s=marker_size,
                marker=markertype_list[idx_selected % len(markertype_list)],
            )

    legend_properties = {"size": legend_font}
    plt.legend(
        prop=legend_properties,
        frameon=False,
        loc=loc,
        ncol=legend_cols,
    )

    if ylim:
        plt.ylim(top=ylim)
    if ylim_bottom:
        plt.ylim(bottom=ylim_bottom)
    if xlim:
        plt.xlim(right=xlim)

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)
    plt.grid(True)

    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_grouped_boxplots(
    bucketed_data,
    bucket_labels,
    scenario_labels=None,  # Default to None for optional legend
    file_name=None,
    x_label="Slowdown Range",
    y_label="Relative Error (%)",
    title=None,
    fontsize=15,
    legend_font=15,
    loc="upper right",
    colors=None,
    fig_size=(7, 4),
    box_width=0.2,
    fig_idx=0,
    ylim=None,
    y_ticklabel_fontsize=None,  # Added parameter for y tick label size
):
    """
    Plots grouped boxplots for multiple scenarios.

    Parameters:
    - bucketed_data: Nested list of errors for each slowdown range and scenario.
                     Format: [[scenario1_data, scenario2_data, ...], ...]
    - bucket_labels: Labels for each slowdown range (x-axis).
    - scenario_labels: Optional labels for each scenario (legend entries).
    - file_name: Name of the file to save the plot (optional).
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - fontsize: Font size for axis labels and title.
    - legend_font: Font size for the legend.
    - loc: Legend location.
    - colors: List of colors for the scenarios.
    - fig_size: Size of the figure.
    - box_width: Width of each box in the plot.
    - fig_idx: Figure index for the plot.
    - y_ticklabel_fontsize: Font size for the y-axis tick labels.
    """
    colors = colors if colors else color_list
    x_positions = np.arange(len(bucket_labels))  # Positions for each slowdown range

    # Create the plot
    fig = plt.figure(fig_idx, figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Plot boxplots for each scenario in each slowdown range
    handles = []
    for j in range(len(bucketed_data[0])):  # Iterate over scenarios
        scenario_data = [bucketed_data[i][j] for i in range(len(bucket_labels))]
        positions = (
            x_positions + j * box_width - (box_width * (len(bucketed_data[0]) - 1)) / 2
        )
        bp = ax.boxplot(
            scenario_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[j % len(colors)], color="black"),
            medianprops=dict(color="black"),
            showfliers=False,
        )
        handles.append(bp["boxes"][0])

    # Configure x-axis labels and ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bucket_labels, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)

    # Set y-axis tick label size
    if y_ticklabel_fontsize:
        ax.tick_params(axis="y", labelsize=y_ticklabel_fontsize)

    # Add legend only if scenario_labels is provided
    if scenario_labels and len(scenario_labels) > 0:
        ax.legend(
            handles,
            scenario_labels,
            fontsize=legend_font,
            loc=loc,
            frameon=False,
        )
    if title:
        ax.set_title(title, fontsize=fontsize)
    if ylim:
        plt.ylim(top=ylim)
    # Final adjustments and save/show the plot
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_box_by_config(
    error_list,
    legend_list,
    config_list,
    config_index,
    config_name,
    n_methods=2,
    x_label=None,
    y_label="Relative Error (%)",
    title=None,
    fontsize=15,
    legend_font=15,
    loc=3,
    rotate_xaxis=False,
    file_name=None,
    remove_outliers=True,
    fig_idx=0,
):
    """
    Plot a grouped boxplot to compare relative errors across configurations.

    Parameters:
    - error_list: 2D array of errors with shape (n_samples, n_methods).
    - legend_list: List of method names corresponding to the columns of error_list.
    - config_list: 2D array of configurations with shape (n_samples, n_config_dims).
    - config_index: Index of the configuration dimension to group by.
    - config_name: Name of the configuration (used in x-axis labels and title).
    - x_label: Custom x-axis label (default: config_name).
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - fontsize: Font size for axis labels and title.
    - legend_font: Font size for the legend.
    - rotate_xaxis: Whether to rotate x-axis tick labels.
    - save_path: Path to save the plot. If None, the plot is not saved.
    - fig_idx: Figure index for the plot.
    """
    colors = [color_list[1], color_list[2]]
    unique_configs = np.unique(config_list[:, config_index])

    # Prepare data for boxplots
    data_to_plot = {
        config_value: [[] for _ in range(n_methods)] for config_value in unique_configs
    }
    for i, config_value in enumerate(config_list[:, config_index]):
        for j in range(n_methods):
            data_to_plot[config_value][j].append(error_list[i, j])

    # Create the plot
    fig = plt.figure(fig_idx, figsize=(5, 3.2))
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    x_positions = np.arange(len(unique_configs))
    box_width = 0.8 / n_methods  # Adjust width based on number of methods
    handles = []

    for j in range(n_methods):
        method_data = [data_to_plot[config_value][j] for config_value in unique_configs]
        positions = x_positions - 0.4 + (j + 0.5) * box_width  # Center the boxes
        bp = ax.boxplot(
            method_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(
                facecolor=colors[j % len(colors)],
                color=colors[j % len(colors)],
            ),
            medianprops=dict(color="black"),
            showfliers=not remove_outliers,
        )
        handles.append(bp["boxes"][0])

    # Set labels and title
    # ax.set_xlabel(x_label if x_label else config_name, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        unique_configs, rotation=45 if rotate_xaxis else 0, fontsize=fontsize - 2
    )
    ax.tick_params(axis="y", labelsize=fontsize - 2)

    if title:
        ax.set_title(title, fontsize=fontsize)

    # Add legend
    if legend_list:
        legend_properties = {"size": legend_font}
        ax.legend(handles, legend_list, loc=loc, prop=legend_properties, frameon=False)

    plt.tight_layout()

    if title:
        ax.set_title(title, fontsize=fontsize - 5)

    # Save or show the plot
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_box(
    bucketed_data,
    bucket_labels,
    legend_list,
    n_methods=2,
    x_label=None,
    y_label="Relative Error (%)",
    title=None,
    fontsize=15,
    legend_font=15,
    loc=3,
    rotate_xaxis=False,
    file_name=None,
    remove_outliers=True,
    fig_idx=0,
):
    colors = [color_list[2]]
    # Create the plot
    fig = plt.figure(fig_idx, figsize=(5, 3.2))
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # x_positions = np.arange(len(unique_configs))
    box_width = 0.8 / n_methods  # Adjust width based on number of methods
    ax.boxplot(
        [
            bucketed_data[label]
            for label in bucket_labels
            if len(bucketed_data[label]) > 0
        ],
        # labels=[label for label in bucket_labels if len(bucketed_data[label]) > 0],
        showfliers=not remove_outliers,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=colors[0], color="black"),
        medianprops=dict(color="black"),
    )

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    # ax.set_xticks(x_positions)
    ax.set_xticklabels(bucket_labels, fontsize=fontsize - 2)
    ax.tick_params(axis="y", labelsize=fontsize - 2)

    if title:
        ax.set_title(title, fontsize=fontsize)

    # Add legend
    if legend_list:
        legend_properties = {"size": legend_font}
        ax.legend(handles, legend_list, loc=loc, prop=legend_properties, frameon=False)

    plt.tight_layout()
    plt.grid(True)
    if title:
        ax.set_title(title, fontsize=fontsize - 5)

    # Save or show the plot
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
