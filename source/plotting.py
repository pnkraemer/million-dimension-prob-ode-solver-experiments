import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def plot_exp_1(run_path):
    file_path = pathlib.Path(run_path)
    df = pd.read_csv(file_path, sep=";")

    all_methods = df["method"].unique()

    figure = plt.figure()
    ax = figure.add_subplot()
    ax.grid()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Time single attempt_step on lorenz96")
    for method in all_methods:
        method_df = df.loc[df["method"] == method]
        nus = method_df["nu"].unique()
        for nu in nus:
            res_df = method_df.loc[method_df["nu"] == nu]
            label = f"{method}, nu={nu}"
            ax.plot(res_df["d"], res_df["time_attempt_step"], label=label, marker="o")
            ax.set_ylabel("Runtime [sec]")
            ax.set_xlabel("dimensions")

    ax.legend()
    figure.savefig(file_path.parent / f"{file_path.stem}_plot.pdf")
