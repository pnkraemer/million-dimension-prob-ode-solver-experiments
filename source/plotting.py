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


# Height-width ratio for a single row
HEIGHT_WIDTH_RATIO_SINGLE = 0.5


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
MARKER_CYCLE = [
    "o",
    "d",
    "*",
    "^",
    "X",
    "P",
]
LINESTYLES = [
    "-",
    "--",
    ":",
    "-.",
    "-",
    "--",
    ":",
    "-.",
]


# Paper-readi(er) legend entries
NICER_METHOD_NAME = {
    "ek0_kronecker": "EK0 (Kron.)",
    "ek0_reference": "EK0 (Trad.)",
    "ek1_reference": "EK1 (Trad.)",
    "ek1_diagonal": "EK1 (Diag.)",
    "ek1_truncated": "EK1 (Trunc.)",
}


# A style per method. EK0s are one color, EK1s are another
# Reference methods have full lines, the sparse ones do not.
# The rest is randomly assigned
EK0_color = "goldenrod"
EK1_color = "cadetblue"
REF_LINESTYLE = "-"
MATCH_STYLE = {
    "ek0_kronecker": (EK0_color, "dotted", "o"),
    "ek0_reference": (EK0_color, REF_LINESTYLE, "^"),
    "ek1_reference": (EK1_color, REF_LINESTYLE, "d"),
    "ek1_diagonal": (EK1_color, "-.", "s"),
    "ek1_truncated": (EK1_color, "dashed", "p"),
}

# Linewidths
THICK = 1.6
MEDIUM = 0.7
THIN = 0.2


def plot_exp_1(run_path):
    """Plot the results from the first experiment."""

    file_path = pathlib.Path(run_path)
    dataframe = pd.read_csv(file_path, sep=";")

    all_methods = [
        "ek1_reference",
        "ek0_reference",
        "ek1_truncated",
        "ek0_kronecker",
        "ek1_diagonal",
    ]

    jit_dataframe = dataframe.loc[dataframe["jit"]]
    no_jit_dataframe = dataframe.loc[~dataframe["jit"]]

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
                    res_dataframe["d"],
                    res_dataframe["time_attempt_step"],
                    label=label,
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    linewidth=THICK,
                    markeredgecolor="black",
                    markeredgewidth=0.2,
                )

        _ax.set_xlabel("ODE dimensions")

        # Line widths
        for spine in _ax.spines:
            _ax.spines[spine].set_linewidth(MEDIUM)

    # --- Plot

    if not jit_dataframe.empty:
        figure_size = (
            AISTATS_LINEWIDTH_DOUBLE,
            AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
        )

        figure = plt.figure(figsize=figure_size)

        # Only use a title for non-paper plots.
        # figure.suptitle("Complexity of an ODE-filter step")
        ax_1 = figure.add_subplot(1, 2, 1)
        ax_2 = figure.add_subplot(1, 2, 2, sharey=ax_1)
        ax_2.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
        ax_2.grid(which="major", linewidth=MEDIUM, color="dimgray")
        ax_2.set_xscale("log")
        ax_2.set_yscale("log")
        ax_2.set_title("JIT-compiled implementation", fontsize="medium")
        _inject_dataframe(ax_2, jit_dataframe)

    else:
        figure_size = (
            AISTATS_TEXTWIDTH_SINGLE,
            AISTATS_TEXTWIDTH_SINGLE * HEIGHT_WIDTH_RATIO_SINGLE,
        )

        figure = plt.figure(figsize=figure_size)
        print("Found no JIT experiments")
        ax_1 = figure.add_subplot()
        _inject_dataframe(ax_1, no_jit_dataframe)
        print(no_jit_dataframe)

    # Axis 1 parameters
    ax_1.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax_1.grid(which="major", linewidth=MEDIUM, color="dimgray")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_title("Standard implementation", fontsize="medium")
    ax_1.set_ylabel("Runtime [sec]")
    _inject_dataframe(ax_1, no_jit_dataframe)

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
    figure.subplots_adjust(bottom=0.3)

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")


NICER_METHOD_NAMEb = {
    "KroneckerEK0": "EK0 (Kron.)",
    "ReferenceEK0": "EK0 (Trad.)",
    "ReferenceEK1": "EK1 (Trad.)",
    "DiagonalEK1": "EK1 (Diag.)",
    "TruncationEK1": "EK1 (Trunc.)",
    "EarlyTruncationEK1": "EK1 (early Trunc.)",
    "EnK1": "EnK1",
}
# A style per method. EK0s are one color, EK1s are another
# Reference methods have full lines, the sparse ones do not.
# The rest is randomly assigned
ENK_color = "indianred"
MATCH_STYLEb = {
    "KroneckerEK0": (EK0_color, "dotted", "o"),
    "ReferenceEK0": (EK0_color, REF_LINESTYLE, "^"),
    "ReferenceEK1": (EK1_color, REF_LINESTYLE, "d"),
    "DiagonalEK1": (EK1_color, "-.", "s"),
    "TruncationEK1": (EK1_color, "dashed", "p"),
    "EarlyTruncationEK1": (EK1_color, "dotted", "^"),
    "EnK1": (ENK_color, "dashed", "s"),
}


def plot_exp_1b(run_path):
    """Plot the results from the first experiment."""

    plt.style.use("./source/font.mplstyle")

    file_path = pathlib.Path(run_path)
    dataframe = pd.read_csv(file_path, sep=";")

    dataframe = dataframe.loc[dataframe["jit"]]

    nus = dataframe["nu"].unique()
    methods = dataframe["method"].unique()
    print(methods)
    figure_size = (
        AISTATS_LINEWIDTH_DOUBLE,
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
    )
    figure, axes = plt.subplots(
        ncols=len(nus), nrows=1, figsize=figure_size, sharey=True
    )

    for nu, ax in zip(nus, axes):
        ax.set_title(rf"IWP({nu})")
        nu_dataframe = dataframe.loc[dataframe["nu"] == nu]
        for method in methods:
            res_dataframe = nu_dataframe.loc[nu_dataframe["method"] == method]
            color, ls, marker = MATCH_STYLEb[method]

            ax.loglog(
                res_dataframe["d"],
                res_dataframe["time_attempt_step"],
                label=NICER_METHOD_NAMEb[method],
                marker=marker,
                color=color,
                linestyle=ls,
                linewidth=THICK,
                markeredgecolor="black",
                markeredgewidth=0.3,
            )
            ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
            ax.grid(which="major", linewidth=MEDIUM, color="dimgray")

        ax.set_xlabel("ODE dimension")
    axes[0].set_ylabel("Time [sec]")

    plt.tight_layout()

    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fancybox=False,
        edgecolor="black",
        fontsize="small",
    ).get_frame().set_linewidth(MEDIUM)
    figure.subplots_adjust(bottom=0.3)

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
    figure.subplots_adjust(bottom=0.3)

    # Save and done.
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
