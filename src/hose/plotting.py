import pathlib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

# Extract from the paper template:
#
# Papers are in 2 columns with the overall line width of 6.75~inches (41~picas).
# Each column is 3.25~inches wide (19.5~picas).  The space
# between the columns is .25~inches wide (1.5~picas).  The left margin is 0.88~inches (5.28~picas).
# Use 10~point type with a vertical spacing of
# 11~points. Please use US Letter size paper instead of A4.
AISTATS_LINEWIDTH_DOUBLE = 6.75
AISTATS_TEXTWIDTH_SINGLE = 3.25


# Height-width ratio for a single row.
# Currently optimised for three columns.
# Check the plot for experiment 1 if this parameter is changed.
HEIGHT_WIDTH_RATIO_SINGLE = 0.35
HEIGHT_WIDTH_RATIO_NEW = 0.45


# Colors, markers, and linestyles. The colors are the tufte-color-scheme (I lost the link...)
COLOR_CYCLE = [
    "gray",
    "crimson",
    "#1b6989",
    "#e69f00",
    "#009e73",
    "#f0e442",
    "#50b4e9",
    "#d55e00",
    "#cc79a7",
]
MARKER_CYCLE = ["o", "d", "*", "^", "X", "P"]
LINESTYLES = ["-", "--", ":", "-.", "-", "--", ":", "-."]


# Paper-readi(er) legend entries. The paper should not read like code?!
NICER_METHOD_NAME = {
    "KroneckerEK0": "EK0 (Kron.)",
    "GPUKroneckerEK0": "EK0 (Kron.) (GPU)",
    "ReferenceEK0": "EK0 (Trad.)",
    "DiagonalEK0": "EK0 (Diag.)",
    "ReferenceEK1": "EK1 (Trad.)",
    "DiagonalEK1": "EK1 (Diag.)",
    "TruncationEK1": "EK1 (Trunc.)",
    "EarlyTruncationEK1": "EK1 (early Trunc.)",
    "EnK1": "EnK1",  # Can we get the ensemble size in here somehow?
    "RK45": "RK45 (SciPy)",
    "DOP853": "DOP853 (SciPy)",
    "Radau": "Radau (SciPy)",
}


# Custom line-widths. Used for the actual curves in the plot.
# Thin(nish) defaults are set in ./src/hose/lines_and_ticks.mplstyle.
THICK = 2.2
MEDIUM = 0.5
THIN = 0.2

# A style per method. EK0s are one color, EK1s are another, EnKX are different as well.
# Reference methods have full lines, the "sparse" ones do not.
# The rest is randomly assigned
EK0_color = "goldenrod"
EK1_color = "cadetblue"
EnK_color = "indianred"
scipy_color = "gray"
REF_LINESTYLE = "-"
MATCH_STYLE = {
    "KroneckerEK0": (EK0_color, "dotted", "o", 0.9, THICK),
    "GPUKroneckerEK0": (EK0_color, "dashed", "v", 0.9, THICK),
    "ReferenceEK0": (EK0_color, REF_LINESTYLE, "^", 0.9, THICK),
    "DiagonalEK0": (EK0_color, "-.", "s", 0.9, THICK),
    "ReferenceEK1": (EK1_color, REF_LINESTYLE, "d", 0.9, THICK),
    "DiagonalEK1": (EK1_color, "-.", "s", 0.9, THICK),
    "TruncationEK1": (EK1_color, "dashed", "p", 0.9, THICK),
    "EarlyTruncationEK1": (EK1_color, "dotted", "^", 0.9, THICK),
    "EnK1": (EnK_color, "dashed", "s", 0.9, THICK),
    "RK45": (scipy_color, "dashed", "P", 0.5, 2 * THICK),
    "DOP853": (scipy_color, "dashed", "P", 0.5, 2 * THICK),
    "Radau": (scipy_color, "dashed", "X", 0.5, 2 * THICK),
}


def plot_exp_1(run_path):
    """Plot the results from the first experiment."""
    # Open sans font with fontsize=8; default lines are thin.
    plt.style.use(["./src/hose/font.mplstyle", "./src/hose/lines_and_ticks.mplstyle"])

    # Load results
    file_path = pathlib.Path(run_path)
    dataframe_complete = pd.read_csv(file_path, sep=";")
    dataframe = dataframe_complete.loc[~dataframe_complete["jit"]]

    # Read derivative and method parameters
    nus = dataframe["nu"].unique()
    methods = [
        "ReferenceEK1",
        "DiagonalEK1",
        "ReferenceEK0",
        "KroneckerEK0",
        "DiagonalEK0",
    ]

    # Create figure
    figure_size = (
        AISTATS_LINEWIDTH_DOUBLE,
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
    )
    figure, axes = plt.subplots(
        ncols=3, nrows=1, figsize=figure_size, sharey=True, sharex=True, dpi=300
    )

    # One prior per figure
    for nu, ax, letter in zip(reversed(nus), axes, ["a", "b", "c"]):

        # Extract data for given nu
        nu_dataframe = dataframe.loc[dataframe["nu"] == nu]

        # Description for the figure
        # ax.set_title(r"$\bf a.$" + "  ",  fontweight="bold", ha="right")
        ax.set_title(rf"$\bf {letter}.$" + rf"IWP({nu})", loc="left", fontsize="medium")
        ax.set_xlabel("ODE dimension")

        # One line/curve per method
        for method in methods:
            res_dataframe = nu_dataframe.loc[nu_dataframe["method"] == method]
            color, ls, marker, alpha, linewidth = MATCH_STYLE[method]
            ax.loglog(
                res_dataframe["d"],
                res_dataframe["time_attempt_step"],
                label=NICER_METHOD_NAME[method],
                marker=marker,
                markersize=5,
                color=color,
                linestyle=ls,
                linewidth=linewidth,
                alpha=alpha,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )

            # Set minor grid, so one can read growth rates easily
            ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
            ax.grid(which="major", linewidth=MEDIUM, color="dimgray")

            # Unify the x-ticks for all plots
            ax.set_xticks((1e1, 1e3, 1e5, 1e7))
            # ax.set_xlim((0.5 * 1e1, 2e4))

    # The leftmost plot gets a y-label -- the others share the y-axis-description
    axes[0].set_ylabel("Wall time [sec]")

    # Tighten up the plot -- adjust_bottom (below) does not work with constrained layout...
    plt.tight_layout()

    # Make space for the legend.
    # Do this after tightening the layout, because otherwise the "new" space would be removed.
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fancybox=False,
        edgecolor="black",
        fontsize="medium",
        handlelength=4,
    ).get_frame().set_linewidth(MEDIUM)
    figure.subplots_adjust(bottom=0.28)

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
    plt.show()


def plot_exp_2(run_path):
    """Plot the results from the second experiment."""

    file_path = pathlib.Path(run_path)
    dataframe = pd.read_csv(file_path, sep=";")
    print(dataframe)

    all_methods = [
        "ReferenceEK1",
        "ReferenceEK0",
        "KroneckerEK0",
        "DiagonalEK0",
        "DiagonalEK1",
        "RK45",
        "Radau",
    ]

    def _inject_dataframe(_ax, _dataframe):
        """Provided axes and dataframe, plot the exp 1 data into the axis."""
        for method in all_methods:
            color, ls, marker, alpha, linewidth = MATCH_STYLE[method]
            method_dataframe = _dataframe.loc[_dataframe["method"] == method]
            nus = method_dataframe["nu"].unique()
            for nu in nus:
                res_dataframe = method_dataframe.loc[method_dataframe["nu"] == nu]
                label_method = f"{NICER_METHOD_NAME[method]}"
                label_order = rf"$\nu={nu}$"
                if method in ["Radau", "RK45"]:
                    label = label_method
                else:
                    label = label_method + ", " + label_order

                _ax.loglog(
                    res_dataframe["deviation"],
                    res_dataframe["time_solve"],
                    label=label_method,
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    linewidth=linewidth,
                    alpha=alpha,
                    markeredgecolor="black",
                    markeredgewidth=0.2,
                )

        # Line widths
        for spine in _ax.spines:
            _ax.spines[spine].set_linewidth(MEDIUM)

        _ax.tick_params(labelsize="x-small")

    # --- Plot

    figure_size = (
        # AISTATS_TEXTWIDTH_SINGLE,
        # AISTATS_TEXTWIDTH_SINGLE * HEIGHT_WIDTH_RATIO_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE * 0.7,
    )

    figure = plt.figure(figsize=figure_size)
    ax_1 = figure.add_subplot()

    # Axis 1 parameters
    ax_1.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax_1.grid(which="major", linewidth=MEDIUM, color="dimgray")
    # ax_1.set_title("Pleiades", fontsize="medium")
    ax_1.set_xlabel("RMSE at final state", fontsize="small")
    ax_1.set_ylabel("Run time [s]", fontsize="small")
    _inject_dataframe(ax_1, dataframe)

    # plt.legend(
    #     fancybox=False,
    #     edgecolor="black",
    #     fontsize="small",
    # ).get_frame().set_linewidth(MEDIUM)

    # Legend
    # handles, labels = ax_1.get_legend_handles_labels()
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fancybox=False,
        edgecolor="black",
        fontsize="xx-small",
    ).get_frame().set_linewidth(MEDIUM)
    # figure.subplots_adjust(right=0.4)
    figure.tight_layout()

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")


def plot_exp_3(run_path):
    """Plot the results from the first experiment."""

    # Load results
    file_path = pathlib.Path(run_path)
    df = pd.read_csv(file_path, sep=";")
    print(df)

    # methods = ["DiagonalEK0", "DiagonalEK1", "KroneckerEK0", "GPUKroneckerEK0"]
    methods = ["DOP853"]

    figure_size = (
        AISTATS_TEXTWIDTH_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE * 2 * HEIGHT_WIDTH_RATIO_SINGLE,
    )
    fig, ax = plt.subplots(figsize=figure_size)
    for method in methods:
        color, ls, marker, alpha, linewidth = MATCH_STYLE[method]
        ax.loglog(
            df["dimensions"],
            df[f"{method}_runtime"],
            label=NICER_METHOD_NAME[method],
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=THICK,
            markeredgecolor="black",
            markeredgewidth=0.3,
        )
    ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax.grid(which="major", linewidth=MEDIUM, color="dimgray")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Run time [sec]")
    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fancybox=False,
        edgecolor="black",
        fontsize="x-small",
        # handlelength=5,
    ).get_frame().set_linewidth(MEDIUM)
    fig.subplots_adjust(bottom=0.41)

    fig.savefig(run_path.parent / "plot.pdf")
    plt.close("all")
    return

    # Plot the #steps, just to be able to check that the results were reasonable
    fig, ax = plt.subplots(figsize=figure_size)
    for method in methods:
        color, ls, marker, alpha, linewidth = MATCH_STYLE[method]
        ax.loglog(
            df["dimensions"],
            df[f"{method}_nsteps"],
            label=NICER_METHOD_NAME[method],
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=THICK,
            markeredgecolor="black",
            markeredgewidth=0.3,
        )
    ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax.grid(which="major", linewidth=MEDIUM, color="dimgray")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("#steps")
    fig.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fancybox=False,
        edgecolor="black",
        fontsize="x-small",
        # handlelength=5,
    ).get_frame().set_linewidth(MEDIUM)
    fig.subplots_adjust(bottom=0.31)

    fig.savefig(run_path.parent / "nsteps.pdf")
    plt.close("all")


def plot_vdp_stiffness_comparison(path):
    plt.style.use(["./src/hose/font.mplstyle", "./src/hose/lines_and_ticks.mplstyle"])
    df = pd.read_csv(path, sep=";")

    # (key, label, color, linestyle)
    SOLVERS_TO_PLOT = [
        ("ReferenceEK0", "ReferenceEK0", EK0_color, "-"),
        ("KroneckerEK0", "KroneckerEK0", EK0_color, "dotted"),
        ("DiagonalEK0", "DiagonalEK0", EK0_color, "dashed"),
        ("ReferenceEK1", "ReferenceEK1", EK1_color, "-"),
        ("DiagonalEK1", "DiagonalEK1", EK1_color, "dotted"),
        ("ETruncationEK1", "EarlyTruncationEK1", EK1_color, "dashed"),
    ]

    figure_size = (
        AISTATS_LINEWIDTH_DOUBLE,
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
        # AISTATS_TEXTWIDTH_SINGLE,
        # AISTATS_TEXTWIDTH_SINGLE * HEIGHT_WIDTH_RATIO_SINGLE,
    )

    def plot_quantity(ax, quantity, ylabel):
        for (s, l, c, ls) in SOLVERS_TO_PLOT:
            ax.loglog(
                df.mu,
                df[f"{s}_{quantity}"],
                label=l,
                color=c,
                marker="o",
                linestyle=ls,
                linewidth=THICK,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Stiffness constant")
        plt.tight_layout()

    def add_legend(ax, fig):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # loc="lower center", ncol=3,
            loc="center right",
            fancybox=False,
            edgecolor="black",
            fontsize="small",
        ).get_frame().set_linewidth(MEDIUM)
        # fig.subplots_adjust(bottom=0.28)
        fig.subplots_adjust(right=0.82)

    ####################################################################################
    # Plot 1: Number of Steps vs Stiffness Constant
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot()
    plot_quantity(ax, "nsteps", "Number of steps")
    add_legend(ax, fig)
    fig.savefig(path.parent / f"nsteps_plot.pdf")

    ####################################################################################
    # Plot 2: Error vs Stiffness Constant
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot()
    plot_quantity(ax, "errors", "Error")
    add_legend(ax, fig)
    fig.savefig(path.parent / f"error_plot.pdf")

    ####################################################################################
    # Plot 3: Seconds vs Stiffness Constant
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot()
    plot_quantity(ax, "seconds", "Seconds")
    add_legend(ax, fig)
    fig.savefig(path.parent / f"seconds_plot.pdf")


def plot_figure1(result_dir):

    # Load data
    ts = jnp.load(result_dir / "times.npy")
    y_means = jnp.load(result_dir / "means.npy")
    y_stds = jnp.load(result_dir / "stddevs.npy")

    # Read off some useful quantities
    D = len(y_means[0])
    num_pts = D // 2
    num_x_points = int(num_pts ** (1 / 2))

    # Choose colormaps
    cmap_means, cmap_stds = "copper", "copper"

    # Create 2x3 image
    figure_size = (
        AISTATS_LINEWIDTH_DOUBLE,
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE * 1.5,
    )

    plt.style.use(["./src/hose/font.mplstyle", "./src/hose/lines_and_ticks.mplstyle"])

    fig, axes = plt.subplots(
        figsize=figure_size,
        nrows=2,
        ncols=4,
        dpi=200,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # First row: plot means
    for (t, y, ax) in zip(ts[1:], y_means[1:], axes[0]):
        mean = y[:num_pts].reshape(num_x_points, num_x_points)

        # Plot the interior
        mean_map = ax.imshow(
            mean,
            cmap=cmap_means,
            vmin=-1,
            vmax=1,
            interpolation="bilinear",
        )

    # Second row: plot standard deviations
    for (t, s, ax) in zip(ts, y_stds, axes[1]):
        std = s[:num_pts].reshape(num_x_points, num_x_points)
        std_map = ax.imshow(
            std,
            cmap=cmap_stds,
            vmin=0,
            vmax=3e-3,
            interpolation="bilinear",
        )

    # Set the titles
    for (t, y, ax) in zip(ts, y_means, axes[0]):
        ax.set_title(f"$t={t}$", fontsize="medium")

    # Some global configs
    for ax, letter in zip(axes.flatten(), ["a", "b", "c", "d", "e", "f", "g", "h"]):
        ax.set_xticks((0, (num_x_points - 1) // 2, (num_x_points - 1)))
        ax.set_yticks((0, (num_x_points - 1) // 2, (num_x_points - 1)))
        ax.set_aspect("equal")
        ax.set_title(rf"$\bf {letter}.$", loc="left", fontsize="small", ha="right")

    # Left column configs
    for axes_row in axes:
        axes_row[0].set_yticklabels((1.0, 0.5, 0.0), fontsize="small")
        axes_row[0].set_ylabel("$x_2$-coord.", fontsize="medium")

    # Bottom row
    for axis in axes[1]:
        axis.set_xticklabels((0.0, 0.5, 1.0), fontsize="small")
        axis.set_xlabel("$x_1$-coord.", fontsize="medium")
    mean_cb = fig.colorbar(
        mean_map, ax=axes[0, -1], extend="both", aspect=12, shrink=0.82
    )
    std_cb = fig.colorbar(std_map, ax=axes[1, -1], extend="max", aspect=12, shrink=0.82)

    # Format the colorbars
    mean_cb.ax.tick_params(labelsize="small")
    mean_cb.set_ticks((-1.0, 0.0, 1.0))
    std_cb.ax.tick_params(labelsize="small")
    std_cb.set_ticks((0.0, 1e-5, 2e-5, 3e-5))
    std_cb.set_ticklabels((0.0, "$10^{-5}$", "$2\cdot 10^{-5}$", "$3\cdot 10^{-5}$"))

    fig.savefig(result_dir / "figure1.pdf")
    plt.show()
