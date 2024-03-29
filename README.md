# "Probabilistic ODE Solvers in Millions of Dimensions" - Experiments

This repo contains the experiment code for the paper "Probabilistic ODE Solvers in Millions of Dimensions", which is accepted at ICML 2022, and currently available on [arXiv](https://arxiv.org/abs/2110.11812).

## Run Experiments
Before running the scripts, install the package with
```
pip install -e .
```

`./experiments/` contains the scripts to run the experiments and create the corresponding figures.
They can be run directly from the root directory.
For example, to fully reproduce figure 0, run
```
python experiments/0_diagonalek1_pdesolution.py
```

`./plotting_scripts/` contains scripts to only create figures without running the actual experiments; they plot the data stored in the corresponding `./results/` folder.


## Reference
The paper is currently on [arXiv](https://arxiv.org/abs/2110.11812):
```
@misc{https://doi.org/10.48550/arxiv.2110.11812,
  doi = {10.48550/ARXIV.2110.11812},
  url = {https://arxiv.org/abs/2110.11812},
  author = {Krämer, Nicholas and Bosch, Nathanael and Schmidt, Jonathan and Hennig, Philipp},
  title = {Probabilistic ODE Solutions in Millions of Dimensions},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
