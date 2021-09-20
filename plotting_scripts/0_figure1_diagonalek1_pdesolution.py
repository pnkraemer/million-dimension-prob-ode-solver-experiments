import pathlib

from hose import plotting

result_dir = pathlib.Path("./results/0_figure1_diagonalek1_pdesolution")
plotting.plot_figure1(result_dir)
