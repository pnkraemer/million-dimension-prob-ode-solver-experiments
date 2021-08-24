import itertools
import pathlib
import timeit

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tornado
from scipy.integrate import solve_ivp

from source import plotting, problems


class MediumScaleExperiment:
    def __init__(
        self,
        method,
        tolerance,
        num_derivatives,
        hyper_param_dict,
    ) -> None:
        self.hyper_param_dict = hyper_param_dict

        self.method = method
        self.tolerance = tolerance
        self.num_derivatives = num_derivatives

        self.ivp = problems.lorenz96_jax(
            params=(hyper_param_dict["ode_dimension"], hyper_param_dict["forcing"]),
            t0=hyper_param_dict["t0"],
            tmax=hyper_param_dict["tmax"],
            y0=jnp.arange(hyper_param_dict["ode_dimension"]),
        )

        self.result = dict()
        reference_sol = solve_ivp(
            fun=self.ivp.f,
            t_span=(hyper_param_dict["t0"], hyper_param_dict["tmax"]),
            y0=np.arange(hyper_param_dict["ode_dimension"]),
            method="LSODA",
            atol=1e-13,
            rtol=1e-13,
        )
        self.reference_state = reference_sol.y[:, -1]

    def check_solve(self):
        def _run_solve():
            state, solver = tornado.ivpsolve.solve(
                self.ivp,
                method=self.method,
                num_derivatives=self.num_derivatives,
                adaptive=True,
                abstol=self.tolerance,
                reltol=self.tolerance,
                save_every_step=False,
            )
            try:
                res = solver.P0 @ state.y.mean
            except:
                res = state.y.mean[0, :]
            return res

        _run_solve()

        elapsed_time = self.time_function(_run_solve)
        end_state = _run_solve()
        self.result["time_solve"] = elapsed_time
        deviation = (
            jnp.linalg.norm((end_state - self.reference_state) / self.reference_state)
            / self.reference_state.size
        )
        self.result["deviation"] = deviation
        return elapsed_time, deviation

    def to_dataframe(self):
        def _aslist(arg):
            try:
                return list(arg)
            except TypeError:
                return [arg]

        results = {k: _aslist(v) for k, v in self.result.items()}
        return pd.DataFrame(
            dict(
                method=self.method,
                tolerance=self.tolerance,
                nu=self.num_derivatives,
                **results,
            ),
        )

    @property
    def hyper_parameters(self):
        def _aslist(arg):
            try:
                return list(arg)
            except TypeError:
                return [arg]

        hparams = {k: _aslist(v) for k, v in self.hyper_param_dict.items()}
        return pd.DataFrame(hparams)

    def __repr__(self) -> str:
        s = f"{self.method} "
        s += "{\n"
        s += f"\ttolerance={self.tolerance}\n"
        s += f"\tnum_derivatives={self.num_derivatives}\n"
        s += f"\tresults={self.result}\n"
        return s + "}"

    def time_function(self, fun):
        # Average time, not minimum time, because we do not want to accidentally
        # run into some of JAX's lazy-execution-optimisation.
        avg_time = timeit.Timer(fun).timeit(number=1)
        return avg_time


# ######################################################################################
#   BEGIN EXPERIMENTS
# ######################################################################################

result_dir = pathlib.Path("./results/2_medium_scale_problem")
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

HYPER_PARAM_DICT = {
    "dt": 0.5,
    "t0": 0.0,
    "tmax": 3.0,
    "forcing": 5.0,
    "ode_dimension": 10,
}

METHODS = tuple(tornado.ivpsolve._SOLVER_REGISTRY.keys())
TOLERANCES = np.logspace(-6, -1, num=6, endpoint=True)
NUM_DERIVS = (5,)

EXPERIMENTS = [
    MediumScaleExperiment(
        method=M, tolerance=TOL, num_derivatives=NU, hyper_param_dict=HYPER_PARAM_DICT
    )
    for (M, TOL, NU) in itertools.product(METHODS, TOLERANCES, NUM_DERIVS)
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
exp.hyper_parameters.to_json(result_dir / "hparams.json", indent=2)


# Plot results
plotting.plot_exp_2(result_file)
