import torch
import random
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from textwrap import wrap

color_list = [
    "cornflowerblue",
    "orange",
    "deeppink",
    "black",
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
    fig_idx=0,
):
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=(5, 2.0))  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

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
                color=color_list[(i % group_size) % len(color_list)],
                label=linelabels[i],
                linewidth=2,
            )
        else:
            plt.plot(
                bin_edges[0:-1],
                cdf,
                linestyle=linestyle_list[(i // group_size) % len(linestyle_list)],
                color=color_list[(i % group_size) % len(color_list)],
                linewidth=2,
            )

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
    # plt.grid(True)

    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def plot_lines(
    raw_data,
    file_name,
    linelabels,
    x_label,
    y_label,
    log_switch=False,
    rotate_xaxis=False,
    ylim=None,
    xlim=None,
    fontsize=15,
    legend_font=15,
    loc=2,
    legend_cols=1,
    title=None,
    format_idx=None,
    fig_idx=0,
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
    _fontsize = fontsize
    fig = plt.figure(fig_idx, figsize=(5, 2.0))
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    if log_switch:
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
                color=color_list[idx_selected % len(color_list)],
                label=linelabels[i],
                linewidth=2,
            )
        else:
            plt.plot(
                x,
                y,
                # linestyle=linestyle_list[idx_selected % len(linestyle_list)],
                color=color_list[idx_selected % len(color_list)],
                linewidth=2,
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
    if xlim:
        plt.xlim(right=xlim)

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    if rotate_xaxis:
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    if title:
        plt.title(title, fontsize=_fontsize - 5)
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)


def create_logger(log_name):
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        # format="%(asctime)s|%(levelname)s| %(processName)s [%(filename)s:%(lineno)d] %(message)s",
        format="%(asctime)s|%(filename)s:%(lineno)d|%(message)s",
        # datefmt="%Y-%m-%d:%H:%M:%S",
        datefmt="%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_name, mode="a"), logging.StreamHandler()],
    )
    for handler in logging.root.handlers:
        handler.addFilter(fileFilter())


def fix_seed(seed):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
