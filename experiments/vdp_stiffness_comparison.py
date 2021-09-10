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


MUS = 10 ** jnp.arange(6)

ATOL, RTOL = 1e-6, 1e-3
STEPRULE = tornadox.step.AdaptiveSteps(abstol=ATOL, reltol=RTOL, min_step=1e-10)

ORDER = 5
SOLVERS = [
    ("ReferenceEK0", ReferenceEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("KroneckerEK0", KroneckerEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("DiagonalEK0", DiagonalEK0(num_derivatives=ORDER, steprule=STEPRULE)),
    ("ReferenceEK1", ReferenceEK1(num_derivatives=ORDER, steprule=STEPRULE)),
    ("DiagonalEK1", DiagonalEK1(num_derivatives=ORDER, steprule=STEPRULE)),
    ("ETruncationEK1", EarlyTruncationEK1(num_derivatives=ORDER, steprule=STEPRULE)),
]


def reference_solve(vdp):
    print("Radau solve")
    radau_sol = scipy.integrate.solve_ivp(
        vdp.f,
        t_span=(vdp.t0, vdp.tmax),
        y0=vdp.y0,
        method="Radau",
        atol=1e-10,
        rtol=1e-10,
    )
    print(f"SOLVED ({len(radau_sol.t)} steps)")
    return radau_sol


def solve_vdp_and_save_results(
    name, solver, vdp, reference_solution, result_dict, totry
):
    print(f"{name} start")
    if totry[name]:
        try:
            start = time.time()
            solution = solver.solve(vdp)
            seconds = time.time() - start
            nsteps = len(solution.t)
            error = jnp.linalg.norm(solution.mean[-1][0] - reference_solution.y[:, -1])
            print(f"SOLVED ({nsteps} steps)")
        except ValueError:
            print("FAILED")
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

    for mu in tqdm(MUS):
        vdp = myvanderpol(stiffness_constant=mu)

        reference_solution = reference_solve(vdp)

        for name, solver in SOLVERS:
            if "EK0" in name and mu > 100:
                result_dict[f"{name}_nsteps"].append(nsteps)
                result_dict[f"{name}_errors"].append(error)
                result_dict[f"{name}_seconds"].append(seconds)
                continue

            solve_vdp_and_save_results(
                name,
                solver,
                vdp,
                reference_solution,
                result_dict,
                totry,
            )

    RESULT_DIR = pathlib.Path("./results/vdp_stiffness_comparison")
    if not RESULT_DIR.is_dir():
        RESULT_DIR.mkdir(parents=True)
    df = pd.DataFrame(result_dict)
    df.to_csv(RESULT_DIR / "results.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
