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
HEIGHT_WIDTH_RATIO_SINGLE = 1 / 2


# Colourblind friendly colours
# from Paul Tot's website: https://personal.sron.nl/~pault/
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


def plot_exp_1(run_path):

    file_path = pathlib.Path(run_path)
    df = pd.read_csv(file_path, sep=";")

    all_methods = df["method"].unique()

    jit_df = df.loc[df["jit"]]
    no_jit_df = df.loc[~df["jit"]]

    def _inject_df(_ax, _df):
        """Provided axes and dataframe, plot the exp 1 data into the axis."""
        for method, color, marker, ls in zip(
            all_methods, COLOR_CYCLE, MARKER_CYCLE, LINESTYLES
        ):
            method_df = _df.loc[_df["method"] == method]
            nus = method_df["nu"].unique()
            for nu in nus:
                res_df = method_df.loc[method_df["nu"] == nu]
                label = f"{method}, nu={nu}"
                _ax.plot(
                    res_df["d"],
                    res_df["time_attempt_step"],
                    label=label,
                    marker=marker,
                    color=color,
                    linestyle=ls,
                )

        _ax.set_xlabel("ODE dimensions")

        _ax.legend(fancybox=False, edgecolor="black", fontsize="small")

    # --- Plot

    if not jit_df.empty:
        figure_size = (
            AISTATS_LINEWIDTH_DOUBLE,
            AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE,
        )

        figure = plt.figure(figsize=figure_size, constrained_layout=True)

        figure.suptitle("Time single attempt_step on Lorenz96", fontweight="bold")
        ax_1 = figure.add_subplot(1, 2, 1)
        ax_2 = figure.add_subplot(1, 2, 2, sharey=ax_1)
        ax_2.grid("minor")
        ax_2.set_xscale("log")
        ax_2.set_yscale("log")
        ax_2.set_title("With JIT")

        _inject_df(ax_2, jit_df)

    else:
        figure = plt.figure(figsize=(5, 5))
        print("Found no JIT experiments")
        ax_1 = figure.add_subplot()

    ax_1.grid("minor")
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_title("Without JIT")
    ax_1.set_ylabel("Runtime [sec]")
    _inject_df(ax_1, no_jit_df)

    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
