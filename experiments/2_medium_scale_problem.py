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
IVP = tornadox.ivp.fhn_2d(bbox=[[-1.0, -1.0], [1.0, 1.0]], dx=0.5)
IVP = tornadox.ivp.pleiades()
reference_sol = solve_ivp(
    fun=IVP.f,
    t_span=(IVP.t0, IVP.tmax),
    y0=IVP.y0,
    method="LSODA",
    jac=IVP.df,
    atol=1e-12,
    rtol=1e-12,
)
reference_state = reference_sol.y[:, -1]

print("Ref done")


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
        solver = self.alg(
            num_derivatives=self.num_derivatives,
            steprule=steprule,
            initialization=tornadox.init.CompiledRungeKutta(dt=0.01, use_df=True),
        )

        def _run_solve(i=10):
            return solver.simulate_final_state(
                IVP, compile_step=True, compile_init=False
            )

        final_state, info = _run_solve(i=1)
        end_state = final_state.y.mean[0]

        self.result["n_steps"] = info["num_steps"]
        self.result["nf"] = info["num_f_evaluations"]
        self.result["time_solve"] = self.time_function(_run_solve)
        self.result["deviation"] = jnp.linalg.norm(
            (end_state - reference_state) / reference_state
        ) / jnp.sqrt(reference_state.size)
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
        avg_time = timeit.Timer(fun).timeit(number=1)
        return avg_time


class ScipyExperiment(MediumScaleExperiment):
    def __init__(
        self,
        alg,
        atol,
        rtol,
    ) -> None:
        super().__init__(alg, atol, rtol, num_derivatives=0)

    def check_solve(self):
        def _run_solve(i=10):
            return solve_ivp(
                fun=IVP.f,
                y0=IVP.y0,
                t_span=(IVP.t0, IVP.tmax),
                method=self.alg,
                atol=self.atol,
                rtol=self.rtol,
                jac=IVP.df,
            )

        sol = _run_solve(i=1)
        end_state = sol.y[:, -1]

        self.result["n_steps"] = len(sol.t)
        self.result["nf"] = sol.nfev
        self.result["time_solve"] = self.time_function(_run_solve)
        self.result["deviation"] = jnp.linalg.norm(
            (end_state - reference_state) / reference_state
        ) / jnp.sqrt(reference_state.size)
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
                method=self.alg,  # line different to superclass
                atol=self.atol,
                rtol=self.rtol,
                nu=self.num_derivatives,
                **results,
            ),
        )


def experiment_generator(algs, atols, rtols, num_derivs, ignore_exp):
    """Generate all the experiments one by one."""

    for (alg, (atol, rtol), nu) in itertools.product(
        algs, zip(atols, rtols), num_derivs
    ):

        if not ignore_exp(alg, atol, rtol, nu):
            yield MediumScaleExperiment(
                alg=alg,
                atol=atol,
                rtol=rtol,
                num_derivatives=nu,
            )


def experiment_generator_scipy(algs, atols, rtols):
    """Generate all the experiments one by one."""

    for (alg, (atol, rtol)) in itertools.product(algs, zip(atols, rtols)):
        yield ScipyExperiment(
            alg=alg,
            atol=atol,
            rtol=rtol,
        )


# ######################################################################################
#   BEGIN EXPERIMENTS
# ######################################################################################

result_dir = pathlib.Path("./results/2_medium_scale_problem")
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

ALGS = [
    KroneckerEK0,
    DiagonalEK0,
    DiagonalEK1,
    ReferenceEK1,
    ReferenceEK0,
]

ALGS_SCIPY = ["RK45", "Radau"]

ATOLS = 1 / 10 ** jnp.arange(3, 13)
RTOLS = 1 / 10 ** jnp.arange(3, 13)
NUM_DERIVS = (4,)

REF_ALGS = [ReferenceEK0, ReferenceEK1]

IGNORE_EXP = lambda alg, atol, rtol, nu: (
    # For reference implementations, ignore too low accuracies
    (alg in REF_ALGS and (atol < 1e-10 or rtol < 1e-10))
)

EXPERIMENTS = experiment_generator(ALGS, ATOLS, RTOLS, NUM_DERIVS, IGNORE_EXP)

EXPERIMENTS_SCIPY = experiment_generator_scipy(ALGS_SCIPY, ATOLS, RTOLS)

# Actual runs
exp_data_frames = []
for exp in EXPERIMENTS_SCIPY:
    exp.check_solve()
    print(exp)
    exp_data_frames.append(exp.to_dataframe())

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
