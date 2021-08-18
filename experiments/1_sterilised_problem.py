"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import itertools
import pathlib
import timeit

import jax.numpy as jnp
import pandas as pd
import tornado

from source import plotting, problems


class SterilisedExperiment:
    def __init__(
        self,
        method,
        num_derivatives,
        ode_dimension,
        hyper_param_dict,
        jit,
        num_repetitions=3,
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

        self.solver = tornado.ivpsolve._SOLVER_REGISTRY[method](
            ode_dimension=ode_dimension,
            steprule=tornado.step.ConstantSteps(hyper_param_dict["dt"]),
            num_derivatives=num_derivatives,
        )
        self.init_state = self.solver.initialize(self.ivp)

        self.result = dict()

        # Should the step function be jitted?
        self.jit = jit

        # How often should each experiment be run?
        self.num_repetitions = num_repetitions

    def time_initialize(self):
        def _run_initialize():
            self.solver.initialize(self.ivp)

        if self.jit:
            _run_initialize = jax.jit(_run_initialize)
        _run_initialize()

        elapsed_time = self.time_function(_run_initialize)
        self.result["time_initialize"] = elapsed_time
        return elapsed_time

    def time_attempt_step(self):
        def _run_attempt_step():
            self.solver.attempt_step(
                state=self.init_state, dt=self.hyper_param_dict["dt"]
            )

        if self.jit:
            _run_attempt_step = jax.jit(_run_attempt_step)
        _run_attempt_step()

        elapsed_time = self.time_function(_run_attempt_step)
        self.result["time_attempt_step"] = elapsed_time
        return elapsed_time

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
                d=self.ode_dimension,
                nu=self.num_derivatives,
                jit=self.jit,
                **results,
            ),
        )

    @property
    def hyper_parameters(self):
        return pd.DataFrame(self.hyper_param_dict)

    def __repr__(self) -> str:
        s = f"{self.method} "
        s += "{\n"
        s += f"\td={self.ode_dimension}\n"
        s += f"\tnu={self.num_derivatives}\n"
        s += f"\tjit={self.jit}\n"
        s += f"\tresults={self.result}\n"
        return s + "}"

    def time_function(self, fun):
        return min(timeit.Timer(fun).repeat(repeat=self.num_repetitions, number=1))


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
METHODS = tuple(tornado.ivpsolve._SOLVER_REGISTRY.keys())
NUM_DERIVS = (8,)
ODE_DIMS = (4, 8, 16, 64, 128, 256)

EXPERIMENTS = [
    SterilisedExperiment(
        method=M, num_derivatives=NU, ode_dimension=D, hyper_param_dict=HYPER_PARAM_DICT
    )
    for (M, NU, D) in itertools.product(METHODS, NUM_DERIVS, ODE_DIMS)
]

# Actual runs
exp_data_frames = []
for exp in EXPERIMENTS:
    exp.time_attempt_step()
    print(exp)
    exp_data_frames.append(exp.to_dataframe())

# Merge experiments into single data frame
merged_data_frame = pd.concat(exp_data_frames, ignore_index=True)


# Save results as CSV
result_file = result_dir / "results.csv"
merged_data_frame.to_csv(result_file, sep=";", index=False)

# Plot results
plotting.plot_exp_1(result_file)
