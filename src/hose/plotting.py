import pathlib

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
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
    "DOP853": "DOP853\n (SciPy)",
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
    "GPUKroneckerEK0": (EK0_color, "dashed", "v", 0.8, THICK),
    "ReferenceEK0": (EK0_color, REF_LINESTYLE, "^", 0.8, THICK),
    "DiagonalEK0": (EK0_color, "-.", "s", 0.8, THICK),
    "ReferenceEK1": (EK1_color, REF_LINESTYLE, "d", 0.8, THICK),
    "DiagonalEK1": (EK1_color, "-.", "s", 0.8, THICK),
    "TruncationEK1": (EK1_color, "dashed", "p", 0.8, THICK),
    "EarlyTruncationEK1": (EK1_color, "dotted", "^", 0.8, THICK),
    "EnK1": (EnK_color, "dashed", "s", 0.8, THICK),
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

    path = "./results/1_sterilised_problem/"
    T_lorenz = jnp.load(path + "T.npy")
    X_lorenz = jnp.load(path + "X.npy")
    Y_lorenz = jnp.load(path + "Y.npy")

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
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE * 0.85,
    )
    figure, axes = plt.subplots(
        ncols=5,
        nrows=1,
        figsize=figure_size,
        dpi=300,
        gridspec_kw=dict(width_ratios=[4, 1, 4, 4, 4]),
    )

    axes[2].get_shared_x_axes().join(axes[1], axes[2])
    axes[3].get_shared_x_axes().join(axes[2], axes[3])
    axes[2].get_shared_y_axes().join(axes[1], axes[2])
    axes[3].get_shared_y_axes().join(axes[2], axes[3])

    # One prior per figure
    for nu, ax, letter in zip(reversed(nus), axes[2:], ["b", "c", "d"]):

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
            if "Reference" in method:
                stride = 1
            else:
                stride = 2
            ax.loglog(
                res_dataframe["d"][::stride],
                res_dataframe["time_attempt_step"][::stride],
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
            ax.set_xlim((0.5 * 1e1, 2e4))
            ax.set_yticks((1e-1, 1e-2, 1e-3, 1e-4))

    # The leftmost plot gets a y-label -- the others share the y-axis-description
    axes[2].set_ylabel("Wall time [sec]")
    axes[3].set_yticklabels(())
    axes[4].set_yticklabels(())

    axes[1].axis("off")

    axes[2].set_xticks((1e1, 1e3, 1e5, 1e7))
    axes[3].set_xticks((1e1, 1e3, 1e5, 1e7))
    axes[4].set_xticks((1e1, 1e3, 1e5, 1e7))
    axes[2].set_yticks((1e0, 1e-2, 1e-4))
    axes[3].set_yticks((1e0, 1e-2, 1e-4))
    axes[4].set_yticks((1e0, 1e-2, 1e-4))

    vmin = jnp.amin(Y_lorenz)
    vmax = jnp.amax(Y_lorenz)

    axes[0].contour(
        X_lorenz.T,
        T_lorenz.T,
        Y_lorenz.T,
        cmap="Greys",
        vmin=vmin,
        vmax=0.1 * vmax,
        linewidths=0.01,
        alpha=0.7,
    )
    axes[0].contourf(
        X_lorenz.T, T_lorenz.T, Y_lorenz.T, cmap="bone", vmin=vmin, vmax=vmax, alpha=0.9
    )
    axes[0].set_xlabel("State compartment")
    axes[0].set_ylabel("Time $t$")
    axes[0].set_title(rf"$\bf a.$" + rf"Lorenz96 system", loc="left", fontsize="medium")
    axes[0].set_xticks((0, len(Y_lorenz) // 2, len(Y_lorenz) - 1))
    axes[0].set_xticklabels((1, len(Y_lorenz) // 2, len(Y_lorenz)))

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
        handlelength=3,
    ).get_frame().set_linewidth(MEDIUM)
    figure.subplots_adjust(bottom=0.35, wspace=0.1)

    # plt.subplots_adjust(wspace=0.1, hspace=0.01)

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

    # --- Plot

    figure_size = (
        # AISTATS_TEXTWIDTH_SINGLE,
        # AISTATS_TEXTWIDTH_SINGLE * HEIGHT_WIDTH_RATIO_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE * 0.8,
    )

    fig = plt.figure(figsize=figure_size, constrained_layout=True, dpi=300)
    spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)
    ax_results = fig.add_subplot(spec[:, :3])
    ax_odesol = fig.add_subplot(spec[0, -1])
    ax_legend = fig.add_subplot(spec[1:3, -1])

    for ax in [ax_results, ax_odesol]:
        # Axis 1 parameters
        ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
        ax.grid(which="major", linewidth=MEDIUM, color="dimgray")

    # ax_1.set_title("Pleiades", fontsize="medium")
    ax_results.set_xlabel("RMSE at final state", fontsize="small")
    ax_results.set_ylabel("Run time [s]", fontsize="small")
    _inject_dataframe_exp_2(ax_results, dataframe, all_methods=all_methods)
    _inject_dataframe_exp_2(ax_odesol, dataframe, all_methods=all_methods)

    h, l = ax.get_legend_handles_labels()
    ax_legend.legend(
        h,
        l,
        loc="center right",
        fancybox=False,
        edgecolor="black",
        fontsize="xx-small",
    ).get_frame().set_linewidth(MEDIUM)

    # plt.legend(
    #     fancybox=False,
    #     edgecolor="black",
    #     fontsize="small",
    # ).get_frame().set_linewidth(MEDIUM)

    # Legend
    # handles, labels = ax_1.get_legend_handles_labels()

    # Save and done.
    fig.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
    plt.show()


def _inject_dataframe_exp_2(_ax, _dataframe, all_methods):
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


def plot_exp_3(run_path):
    """Plot the results from the first experiment."""

    # Load results
    file_path = pathlib.Path(run_path)
    df = pd.read_csv(file_path, sep=";", index_col=False)
    print(df)

    methods = [
        "DiagonalEK0",
        "DiagonalEK1",
        "KroneckerEK0",
        "GPUKroneckerEK0",
        "DOP853",
    ]

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
            linewidth=linewidth,
            markeredgecolor="black",
            markeredgewidth=0.3,
            alpha=alpha,
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
        ncol=3,
        fancybox=False,
        edgecolor="black",
        fontsize="x-small",
        # handlelength=5,
    ).get_frame().set_linewidth(MEDIUM)
    fig.subplots_adjust(bottom=0.41)

    fig.savefig(run_path.parent / "plot.pdf")
    plt.close("all")

    # Plot the #steps, just to be able to check that the results were reasonable
    fig, ax = plt.subplots(figsize=figure_size)
    for method in methods:
        color, ls, marker, alpha, linewidth = MATCH_STYLE[method]
        ax.loglog(
            df["dimensions"],
            df[f"{method}_nf"],
            label=NICER_METHOD_NAME[method],
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=linewidth,
            markeredgecolor="black",
            markeredgewidth=0.3,
            alpha=alpha,
        )
    ax.grid(which="both", linewidth=THIN, alpha=0.25, color="darkgray")
    ax.grid(which="major", linewidth=MEDIUM, color="dimgray")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("#fevals")
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

    fig.savefig(run_path.parent / "nf.pdf")
    plt.close("all")


def plot_vdp_stiffness_comparison(path):
    plt.style.use(["./src/hose/font.mplstyle", "./src/hose/lines_and_ticks.mplstyle"])
    df = pd.read_csv(path, sep=";")

    y = jnp.load("./results/vdp_stiffness_comparison/Y.npy")

    # (key, label, color, linestyle)
    SOLVERS_TO_PLOT = [
        ("ReferenceEK0", "ReferenceEK0", EK0_color, "-"),
        ("KroneckerEK0", "KroneckerEK0", EK0_color, "dotted"),
        ("DiagonalEK0", "DiagonalEK0", EK0_color, "dashed"),
        ("ReferenceEK1", "ReferenceEK1", EK1_color, "-"),
        ("DiagonalEK1", "DiagonalEK1", EK1_color, "dotted"),
        # ("ETruncationEK1", "EarlyTruncationEK1", EK1_color, "dashed"),
    ]

    figure_size = (
        AISTATS_TEXTWIDTH_SINGLE,
        AISTATS_TEXTWIDTH_SINGLE * 0.8,
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
        # plt.tight_layout()

    def add_legend(ax, fig):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            # loc="lower center", ncol=3,
            loc="lower right",
            fancybox=False,
            edgecolor="black",
            fontsize="small",
        ).get_frame().set_linewidth(MEDIUM)
        # fig.subplots_adjust(bottom=0.28)
        # fig.subplots_adjust(right=0.82)

    ####################################################################################
    # Plot 1: Number of Steps vs Stiffness Constant
    fig, ax = plt.subplots(figsize=figure_size, dpi=200, constrained_layout=True)
    # ax = fig.add_subplot()
    plot_quantity(ax, "nsteps", "Number of steps")
    add_legend(ax, fig)
    ax.grid(which="both", linewidth=THIN, alpha=0.3, color="darkgray")
    ax.set_ylim(top=1e6)

    ax1 = ax.inset_axes([0.05, 0.6, 0.22, 0.3])
    ax1.plot(y[0], y[1], linewidth=1.0, color="black")
    ax1.set_xticklabels(())
    ax1.set_yticklabels(())

    ax.set_title(rf"$\bf a.$ Some title", loc="left", fontsize="medium")
    ax1.set_title(rf"$\bf b.$ Van der Pol", loc="left", fontsize="small")

    fig.savefig(path.parent / f"nsteps_plot.pdf")

    # ####################################################################################
    # # Plot 2: Error vs Stiffness Constant
    # fig = plt.figure(figsize=figure_size)
    # ax = fig.add_subplot()
    # plot_quantity(ax, "errors", "Error")
    # add_legend(ax, fig)
    # fig.savefig(path.parent / f"error_plot.pdf")

    # ####################################################################################
    # # Plot 3: Seconds vs Stiffness Constant
    # fig = plt.figure(figsize=figure_size)
    # ax = fig.add_subplot()
    # plot_quantity(ax, "seconds", "Seconds")

    # add_legend(ax, fig)
    # fig.savefig(path.parent / f"seconds_plot.pdf")

    plt.show()


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
        AISTATS_LINEWIDTH_DOUBLE * HEIGHT_WIDTH_RATIO_SINGLE * 1.2,
    )

    plt.style.use(["./src/hose/font.mplstyle", "./src/hose/lines_and_ticks.mplstyle"])

    fig, axes = plt.subplots(
        figsize=figure_size,
        nrows=2,
        ncols=5,
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
            interpolation=None,
        )

    # Second row: plot standard deviations
    for (t, s, ax) in zip(ts, y_stds, axes[1]):
        std = s[:num_pts].reshape(num_x_points, num_x_points)
        std_map = ax.imshow(
            std,
            cmap=cmap_stds,
            vmin=0,
            vmax=1e-5,
            interpolation=None,
        )

    # Set the titles
    for (t, y, ax) in zip(ts, y_means, axes[0]):
        ax.set_title(f"$t={t}$", fontsize="medium")

    # Some global configs
    for ax, letter in zip(
        axes.flatten(), ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    ):
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
        mean_map, ax=axes[0, -1], extend="both", aspect=10, shrink=0.9
    )
    std_cb = fig.colorbar(std_map, ax=axes[1, -1], extend="max", aspect=10, shrink=0.9)

    # Format the colorbars
    mean_cb.ax.tick_params(labelsize="small")
    mean_cb.set_ticks((-1.0, 0.0, 1.0))
    std_cb.ax.tick_params(labelsize="small")
    # std_cb.set_ticks((0.0, 1e-5, 2e-5, 3e-5))
    std_cb.set_ticks((0.0, 1e-5, 5e-6))
    std_cb.set_ticklabels((0.0, "$10^{-5}$", "$5\cdot 10^{-6}$"))

    fig.savefig(result_dir / "figure1.pdf")
    plt.show()
