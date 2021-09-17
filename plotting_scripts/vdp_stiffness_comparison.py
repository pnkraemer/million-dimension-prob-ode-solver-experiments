import pathlib

from hose import plotting

path = pathlib.Path("./results/vdp_stiffness_comparison/results.csv")
plotting.plot_vdp_stiffness_comparison(path)
