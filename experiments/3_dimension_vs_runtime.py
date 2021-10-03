import itertools
import pathlib
import sys
import time
import timeit
from collections import defaultdict

import jax
import jax.numpy as jnp
import pandas as pd
import tornadox
from scipy.integrate import solve_ivp

from hose import plotting

TMAX = 20

# widths = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]  # For the GPU KroneckerEK0
widths = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]  # CPU

methods = [
    tornadox.ek0.KroneckerEK0,
    tornadox.ek0.DiagonalEK0,
    tornadox.ek1.DiagonalEK1,
]
steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-1)
solvers = [
    method(
        num_derivatives=3,
        steprule=steprule,
        initialization=tornadox.init.RungeKutta(use_df=False),
    )
    for method in methods
]
results = defaultdict(list)

# Precompile
for solver in solvers:
    state = solver.simulate_final_state(
        tornadox.ivp.fhn_2d(dx=0.1, tmax=0.01),
    )

# Perform the experiment
# for dx in dxs:
for width in widths:
    IVP = tornadox.ivp.fhn_2d(
        # bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=dx,
        bbox=[[0.0, 0.0], [width, width]],
        tmax=TMAX,
    )
    # results["dxs"].append(dx)
    results["width"].append(width)
    dim = len(IVP.y0)
    results["dimensions"].append(dim)
    print(f"\nwidth={width}; dimension={len(IVP.y0)}")

    # print(f"Start: DOP853")
    # start = time.time()
    # reference_sol = solve_ivp(
    #     fun=IVP.f,
    #     t_span=(IVP.t0, IVP.tmax),
    #     y0=IVP.y0,
    #     # method="RK45",
    #     method="DOP853",
    #     atol=1e-13,
    #     rtol=1e-13,
    #     t_eval=[TMAX],
    # )
    # elapsed = time.time() - start
    # reference_state = reference_sol.y[:, -1]
    # results["DOP853_runtime"].append(elapsed)
    # print(f"Done in {elapsed}")

    for solver in solvers:
        solvername = solver.__class__.__name__
        # if dx < 0.02 and not isinstance(solver, tornadox.ek0.KroneckerEK0):
        if dim > 10 ** 5 and not isinstance(solver, tornadox.ek0.KroneckerEK0):
            print(f"Skipping: {solver}")
            results[f"{solvername}_runtime"].append(None)
            results[f"{solvername}_errors"].append(None)
            continue

        print(f"Start: {solvername}")
        start = time.time()
        state, info = solver.simulate_final_state(IVP, progressbar=True)
        elapsed = time.time() - start
        results[f"{solvername}_runtime"].append(elapsed)
        results[f"{solvername}_nsteps"].append(info["num_steps"])
        results[f"{solvername}_nf"].append(info["num_f_evaluations"])
        print(f"Done in {elapsed}")
        # error = jnp.linalg.norm(state.y.mean[0] - reference_state)
        # rmse = jnp.linalg.norm(
        #     (state.y.mean[0] - reference_state) / reference_state
        # ) / jnp.sqrt(reference_state.size)
        # results[f"{solvername}_errors"].append(rmse)


result_dir = pathlib.Path("./results/3_dimension_vs_runtime")
result_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
result_file = result_dir / "results.csv"
df.to_csv(result_file, sep=";", index=False)

# Plotting
plotting.plot_exp_3(result_file)
