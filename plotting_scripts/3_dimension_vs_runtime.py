"""Visualisation of the 2nd experiment."""
import pathlib

from hose import plotting

path = pathlib.Path("./results/3_dimension_vs_runtime/results.csv")
plotting.plot_exp_3(path)
