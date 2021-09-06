import pathlib

import matplotlib.pyplot as plt
import pandas as pd

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
    "ReferenceEK0": "EK0 (Trad.)",
    "ReferenceEK1": "EK1 (Trad.)",
    "DiagonalEK1": "EK1 (Diag.)",
    "TruncationEK1": "EK1 (Trunc.)",
    "EarlyTruncationEK1": "EK1 (early Trunc.)",
    "EnK1": "EnK1",  # Can we get the ensemble size in here somehow?
}


# A style per method. EK0s are one color, EK1s are another, EnKX are different as well.
# Reference methods have full lines, the "sparse" ones do not.
# The rest is randomly assigned
EK0_color = "goldenrod"
EK1_color = "cadetblue"
EnK_color = "indianred"
REF_LINESTYLE = "-"
MATCH_STYLE = {
    "KroneckerEK0": (EK0_color, "dotted", "o"),
    "ReferenceEK0": (EK0_color, REF_LINESTYLE, "^"),
    "ReferenceEK1": (EK1_color, REF_LINESTYLE, "d"),
    "DiagonalEK1": (EK1_color, "-.", "s"),
    "TruncationEK1": (EK1_color, "dashed", "p"),
    "EarlyTruncationEK1": (EK1_color, "dotted", "^"),
    "EnK1": (EnK_color, "dashed", "s"),
}

# Custom line-widths. Used for the actual curves in the plot.
# Thin(nish) defaults are set in ./source/lines_and_ticks.mplstyle.
THICK = 1.8
MEDIUM = 0.5
THIN = 0.2


def plot_exp_1(run_path):
    """Plot the results from the first experiment."""

    # Open sans font with fontsize=8; default lines are thin.
    plt.style.use(["./source/font.mplstyle", "./source/lines_and_ticks.mplstyle"])

    # Load results
    file_path = pathlib.Path(run_path)
    dataframe_complete = pd.read_csv(file_path, sep=";")
    dataframe = dataframe_complete.loc[
        dataframe_complete["jit"]
    ]  # Ignore the non-jit experiments

    # Read derivative and method parameters
    nus = dataframe["nu"].unique()
    methods = dataframe["method"].unique()

    # Create figure
    figure_size = (
        AISTATS_LINEWIDTH_DOUBLE,
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
    )
    figure, axes = plt.subplots(
        ncols=len(nus), nrows=1, figsize=figure_size, sharey=True, sharex=True
    )

    # One prior per figure
    for nu, ax in zip(reversed(nus), axes):

        # Extract data for given nu
        nu_dataframe = dataframe.loc[dataframe["nu"] == nu]

        # Description for the figure
        ax.set_title(rf"IWP({nu})")
        ax.set_xlabel("ODE dimension")

        # One line/curve per method
        for method in methods:
            res_dataframe = nu_dataframe.loc[nu_dataframe["method"] == method]
            color, ls, marker = MATCH_STYLE[method]
            ax.loglog(
                res_dataframe["d"],
                res_dataframe["time_attempt_step"],
                label=NICER_METHOD_NAME[method],
                marker=marker,
                markersize=5,
                color=color,
                linestyle=ls,
                linewidth=THICK,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )

            # Set minor grid, so one can read growth rates easily
            ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
            ax.grid(which="major", linewidth=MEDIUM, color="dimgray")

            # Unify the x-ticks for all plots
            ax.set_xticks((1e1, 1e2, 1e3, 1e4))
            ax.set_xlim((0.5 * 1e1, 2e4))

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
        handlelength=5,
    ).get_frame().set_linewidth(MEDIUM)
    figure.subplots_adjust(bottom=0.28)

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")


def plot_exp_2(run_path):
    """Plot the results from the second experiment."""

    file_path = pathlib.Path(run_path)
    dataframe = pd.read_csv(file_path, sep=";")

    all_methods = [
        "ek1_reference",
        "ek0_reference",
        "ek1_truncated",
        "ek0_kronecker",
        "ek1_diagonal",
    ]

    def _inject_dataframe(_ax, _dataframe):
        """Provided axes and dataframe, plot the exp 1 data into the axis."""
        for method in all_methods:
            color, ls, marker = MATCH_STYLE[method]
            method_dataframe = _dataframe.loc[_dataframe["method"] == method]
            nus = method_dataframe["nu"].unique()
            for nu in nus:
                res_dataframe = method_dataframe.loc[method_dataframe["nu"] == nu]
                label_method = f"{NICER_METHOD_NAME[method]}"
                label_order = rf"$\nu={nu}$"
                label = label_method + ", " + label_order

                _ax.plot(
                    res_dataframe["time_solve"],
                    res_dataframe["deviation"],
                    label=label,
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    linewidth=THICK,
                    markeredgecolor="black",
                    markeredgewidth=0.2,
                )

        # Line widths
        for spine in _ax.spines:
            _ax.spines[spine].set_linewidth(MEDIUM)

    # --- Plot

    figure_size = (
        AISTATS_TEXTWIDTH_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE * HEIGHT_WIDTH_RATIO_SINGLE,
    )

    figure = plt.figure(figsize=figure_size)
    ax_1 = figure.add_subplot()

    # Axis 1 parameters
    ax_1.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax_1.grid(which="major", linewidth=MEDIUM, color="dimgray")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_title("Work-precision diagram", fontsize="medium")
    ax_1.set_ylabel("relative L2 deviation")
    ax_1.set_xlabel("Run-time [sec]")
    _inject_dataframe(ax_1, dataframe)

    # Legend
    handles, labels = ax_1.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fancybox=False,
        edgecolor="black",
        fontsize="small",
    ).get_frame().set_linewidth(MEDIUM)
    figure.subplots_adjust(bottom=0.4)

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")


def plot_exp_5(run_path):
    import plotly.express as px

    file_path = pathlib.Path(run_path)

    df = pd.read_csv(file_path, sep=";")

    fig = px.parallel_coordinates(df, color="n_steps")
    fig.show()