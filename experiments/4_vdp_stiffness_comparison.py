"""
Goal: Compare the truncated EK1 to the full EK1

Ideally, either show:
- the truncated version has worse stability properties
  => show how it fails on the stiff VdP
- the truncated version has basically the same stability properties
  => work-precision diagram to show how they are the same
"""
import pathlib
import time

import jax
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
import tornadox
from jax import numpy as jnp
from tornadox.ek0 import *
from tornadox.ek1 import *
from tqdm import tqdm

from hose import plotting


def myvanderpol(t0=0.0, tmax=6.3, y0=None, stiffness_constant=1e1):

    y0 = y0 or jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([Y[1], mu * ((1.0 - Y[0] ** 2) * Y[1] - Y[0])])

    df_vanderpol = jax.jit(jax.jacfwd(f_vanderpol, argnums=1))

    @jax.jit
    def df_diagonal_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([0.0, mu * (1.0 - Y[0] ** 2)])

    return tornadox.ivp.InitialValueProblem(
        f=f_vanderpol,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_vanderpol,
        df_diagonal=df_diagonal_vanderpol,
    )


MUS = 10 ** jnp.arange(7)

ATOL, RTOL = 1e-6, 1e-3
STEPRULE = tornadox.step.AdaptiveSteps(abstol=ATOL, reltol=RTOL, min_step=1e-10)

ORDER = 5
SOLVERS = [
    ("ReferenceEK0", ReferenceEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("KroneckerEK0", KroneckerEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("DiagonalEK0", DiagonalEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("ReferenceEK1", ReferenceEK1(num_derivatives=ORDER, steprule=STEPRULE)),
    ("DiagonalEK1", DiagonalEK1(num_derivatives=ORDER, steprule=STEPRULE)),
    # ("ETruncationEK1", EarlyTruncationEK1(num_derivatives=ORDER, steprule=STEPRULE)),
]


def reference_solve(vdp):
    radau_sol = scipy.integrate.solve_ivp(
        vdp.f,
        t_span=(vdp.t0, vdp.tmax),
        y0=vdp.y0,
        method="Radau",
        atol=1e-10,
        rtol=1e-10,
    )
    return radau_sol


def solve_vdp_and_save_results(
    name, solver, vdp, reference_solution, result_dict, totry, pbar
):
    pbar.write(f"\n[START] {name}")
    if totry[name]:
        try:
            start = time.time()
            solution = solver.solve(vdp)
            state, info = solver.simulate_final_state(vdp, progressbar=True)
            seconds = time.time() - start
            nsteps = info["num_steps"]
            error = jnp.linalg.norm(state.y.mean[0] - reference_solution.y[:, -1])
            pbar.write(f"[DONE] ({nsteps} steps)")
        except ValueError:
            pbar.write("[FAILED]")
            totry[name] = False
            nsteps, error, seconds = None, None, None
    else:
        nsteps, error, seconds = None, None, None
    result_dict[f"{name}_nsteps"].append(nsteps)
    result_dict[f"{name}_errors"].append(error)
    result_dict[f"{name}_seconds"].append(seconds)


def main():
    totry = {n: True for (n, _) in SOLVERS}
    result_dict = {"mu": MUS}
    for n, _ in SOLVERS:
        result_dict[f"{n}_nsteps"] = []
        result_dict[f"{n}_errors"] = []
        result_dict[f"{n}_seconds"] = []

    outer_pbar = tqdm(MUS)
    for mu in outer_pbar:
        vdp = myvanderpol(stiffness_constant=mu)

        reference_solution = reference_solve(vdp)
        outer_pbar.write(f"[REFERENCE] Radau took {len(reference_solution.t)} steps")

        pbar = tqdm(SOLVERS)
        for name, solver in pbar:
            if "EK0" in name and mu > 1000:
                result_dict[f"{name}_nsteps"].append(None)
                result_dict[f"{name}_errors"].append(None)
                result_dict[f"{name}_seconds"].append(None)
                continue

            solve_vdp_and_save_results(
                name,
                solver,
                vdp,
                reference_solution,
                result_dict,
                totry,
                pbar,
            )

    RESULT_DIR = pathlib.Path("./results/4_vdp_stiffness_comparison")
    if not RESULT_DIR.is_dir():
        RESULT_DIR.mkdir(parents=True)
    df = pd.DataFrame(result_dict)
    result_file = RESULT_DIR / "results.csv"
    df.to_csv(result_file, sep=";", index=False)

    plotting.plot_4_vdp_stiffness_comparison(result_file)


if __name__ == "__main__":
    main()
