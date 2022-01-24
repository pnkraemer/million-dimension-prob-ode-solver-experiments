import pathlib

from hose import plotting

path = pathlib.Path("./results/4_vdp_stiffness_comparison/results.csv")
plotting.plot_4_vdp_stiffness_comparison(path)
