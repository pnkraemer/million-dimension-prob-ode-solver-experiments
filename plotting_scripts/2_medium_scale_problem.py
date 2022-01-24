"""Visualisation of the 2nd experiment."""
import pathlib

from hose import plotting

path = pathlib.Path("./results/2_medium_scale_problem/results.csv")
plotting.plot_2_medium_scale_problem(path)
