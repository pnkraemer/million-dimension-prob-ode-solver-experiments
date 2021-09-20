"""Visualisation of the 1st experiment."""
import pathlib

from hose import plotting

path = pathlib.Path("./results/1_sterilised_problem/results.csv")
plotting.plot_exp_1(path)
