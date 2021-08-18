import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def plot_exp_1(run_path):
    print("The jitting is not handled well yet. Plot may be meaningless.")
    file_path = pathlib.Path(run_path)
    df = pd.read_csv(file_path, sep=";")

    all_methods = df["method"].unique()

    jit_df = df.loc[df["jit"]]
    no_jit_df = df.loc[~df["jit"]]

    def _inject_df(_ax, _df):
        """Provided axes and dataframe, plot the exp 1 data into the axis."""
        for method in all_methods:
            method_df = _df.loc[_df["method"] == method]
            nus = method_df["nu"].unique()
            for nu in nus:
                res_df = method_df.loc[method_df["nu"] == nu]
                label = f"{method}, nu={nu}"
                _ax.plot(
                    res_df["d"], res_df["time_attempt_step"], label=label, marker="o"
                )
                _ax.set_ylabel("Runtime [sec]")
                _ax.set_xlabel("dimensions")

        _ax.legend()

    # --- Plot

    if not jit_df.empty:
        figure = plt.figure(figsize=(10, 5))
        ax_1 = figure.add_subplot(1, 2, 1)
        ax_2 = figure.add_subplot(1, 2, 2)
        ax_2.grid()
        ax_2.set_xscale("log", base=2)
        ax_2.set_yscale("log")
        ax_2.set_title("Time single attempt_step on lorenz96 using JIT")

        _inject_df(ax_2, jit_df)

    else:
        figure = plt.figure(figsize=(5, 5))
        print("Found no JIT experiments")
        ax_1 = figure.add_subplot()

    ax_1.grid()
    ax_1.set_xscale("log", base=2)
    ax_1.set_yscale("log")
    ax_1.set_title("Time single attempt_step on lorenz96")

    _inject_df(ax_1, no_jit_df)

    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
