"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import itertools
import json
import pathlib
import timeit
from datetime import datetime

import jax.numpy as jnp
import tornado

from source import problems


class SterilisedExperiment:
    def __init__(
        self, method, num_derivatives, ode_dimension, hyper_param_dict
    ) -> None:
        self.hyper_param_dict = hyper_param_dict

        self.method = method
        self.num_derivatives = num_derivatives
        self.ode_dimension = ode_dimension

        self.ivp = problems.lorenz96_jax(
            params=(ode_dimension, hyper_param_dict["forcing"]),
            t0=hyper_param_dict["t0"],
            tmax=hyper_param_dict["tmax"],
            y0=jnp.arange(ode_dimension),
        )

        self.solver = method(
            ode_dimension=ode_dimension,
            steprule=tornado.step.ConstantSteps(hyper_param_dict["dt"]),
            num_derivatives=num_derivatives,
        )
        self.init_state = self.solver.initialize(self.ivp)

        self.result = dict()

    def _run_initialize(self):
        self.solver.initialize(self.ivp)

    def _run_attempt_step(self):
        return self.solver.attempt_step(
            state=self.init_state, dt=self.hyper_param_dict["dt"]
        )

    def time_initialize(self):
        elapsed_time = timeit.Timer(self._run_initialize).timeit(number=1)
        self.result["time_initialize"] = elapsed_time
        return elapsed_time

    def time_attempt_step(self):
        elapsed_time = timeit.Timer(self._run_attempt_step).timeit(number=1)
        self.result["time_attempt_step"] = elapsed_time
        return elapsed_time

    def pickle(self):
        return {
            "method": str(self.method),
            "d": self.ode_dimension,
            "nu": self.num_derivatives,
            "hyperparams": self.hyper_param_dict,
            "results": self.result,
        }

    def __repr__(self) -> str:
        s = f"{self.method} "
        s += "{\n"
        s += f"\td={self.ode_dimension}\n"
        s += f"\tnu={self.num_derivatives}\n"
        s += f"\tresults={self.result}\n"
        return s + "}"


# ######################################################################################
#   BEGIN EXPERIMENTS
# ######################################################################################

result_dir = pathlib.Path("./results/1_sterilised_problem")
if not result_dir.is_dir():
    result_dir.mkdir(parents=True)

HYPER_PARAM_DICT = {
    "dt": 0.5,
    "t0": 0.0,
    "tmax": 3.0,
    "forcing": 5.0,
}
METHODS = tuple(tornado.ivpsolve._SOLVER_REGISTRY.values())
NUM_DERIVS = (3, 8)
ODE_DIMS = (4, 8, 16, 64, 128)

EXPERIMENTS = [
    SterilisedExperiment(
        method=M, num_derivatives=NU, ode_dimension=D, hyper_param_dict=HYPER_PARAM_DICT
    )
    for (M, NU, D) in itertools.product(METHODS, NUM_DERIVS, ODE_DIMS)
]

pickled_results = []

# Actual runs
for exp in EXPERIMENTS:
    elapsed_time = exp.time_attempt_step()
    pickled_results.append(exp.pickle())
    print(exp)


# Save results
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_file = result_dir / f"{time_stamp}_results.json"
with open(result_file, "x") as file_handle:
    json.dump(pickled_results, file_handle, indent=4)
