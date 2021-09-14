import itertools
import pathlib
import timeit

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tornadox
from scipy.integrate import solve_ivp
from tornadox.ek0 import *
from tornadox.ek1 import *

from hose import plotting

# IVP = tornadox.ivp.pleiades()
IVP = tornadox.ivp.lorenz96()
reference_sol = solve_ivp(
    fun=IVP.f,
    t_span=(IVP.t0, IVP.tmax),
    y0=IVP.y0,
    method="LSODA",
    jac=IVP.df,
    atol=1e-13,
    rtol=1e-13,
)
reference_state = reference_sol.y[:, -1]


class MediumScaleExperiment:
    def __init__(
        self,
        alg,
        atol,
        rtol,
        num_derivatives,
    ) -> None:

        self.alg = alg
        self.atol, self.rtol = atol, rtol
        self.num_derivatives = num_derivatives

        self.result = dict()

    def check_solve(self):
        steprule = tornadox.step.AdaptiveSteps(abstol=self.atol, reltol=self.rtol)
        solver = self.alg(num_derivatives=self.num_derivatives,
                          steprule=steprule,
                          initialization=tornadox.init.RungeKutta(),
                          )

        def _run_solve():
            return solver.solve(IVP)

        solution = _run_solve()
        end_state = solution.mean[-1, 0, :]

        self.result["n_steps"] = len(solution.t)
        self.result["time_solve"] = self.time_function(_run_solve)
        self.result["deviation"] = (
            jnp.linalg.norm((end_state - reference_state) / reference_state)
            / reference_state.size
        )
        return None

    def to_dataframe(self):
        def _aslist(arg):
            try:
                return list(arg)
            except TypeError:
                return [arg]

        results = {k: _aslist(v) for k, v in self.result.items()}
        return pd.DataFrame(
            dict(
                method=self.alg.__name__,
                atol=self.atol,
                rtol=self.rtol,
                nu=self.num_derivatives,
                **results,
            ),
        )

    def __repr__(self) -> str:
        s = f"{self.alg} "
        s += "{\n"
        s += f"\tatol={self.atol}\n"
        s += f"\trtol={self.rtol}\n"
        s += f"\tnum_derivatives={self.num_derivatives}\n"
        s += f"\tresults={self.result}\n"
        return s + "}"

    def time_function(self, fun):
        # Average time, not minimum time, because we do not want to accidentally
        # run into some of JAX's lazy-execution-optimisation.
        avg_time = timeit.Timer(fun).timeit(number=2)
        return avg_time


# ######################################################################################
#   BEGIN EXPERIMENTS
# ######################################################################################

result_dir = pathlib.Path("./results/2_medium_scale_problem")
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

ALGS = [
    ReferenceEK0,
    # KroneckerEK0,
    DiagonalEK0,
    ReferenceEK1,
    DiagonalEK1,
]

ATOLS = 1 / 10 ** jnp.arange(5, 11)
RTOLS = 1 / 10 ** jnp.arange(2, 8)
NUM_DERIVS = (5,)

EXPERIMENTS = [
    MediumScaleExperiment(
        alg=alg,
        atol=atol,
        rtol=rtol,
        num_derivatives=nu,
    )
    for (alg, (atol, rtol), nu) in itertools.product(
        ALGS, zip(ATOLS, RTOLS), NUM_DERIVS
    )
]

# Actual runs
exp_data_frames = []
for exp in EXPERIMENTS:
    exp.check_solve()
    print(exp)
    exp_data_frames.append(exp.to_dataframe())

# Merge experiments into single data frame
merged_data_frame = pd.concat(exp_data_frames, ignore_index=True)


# Save results as CSV
result_file = result_dir / "results.csv"
merged_data_frame.to_csv(result_file, sep=";", index=False)


# Plot results
plotting.plot_exp_2(result_file)
