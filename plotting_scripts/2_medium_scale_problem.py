"""Visualisation of the 2nd experiment."""
import pathlib

from hose import plotting

path = pathlib.Path("./results/2_medium_scale_problem/results.csv")
plotting.plot_exp_2(path)
