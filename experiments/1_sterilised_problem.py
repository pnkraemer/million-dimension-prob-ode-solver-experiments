"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import itertools
import pathlib
import timeit

import jax
import jax.numpy as jnp
import pandas as pd
import tornadox

from source import plotting, problems


class SterilisedExperiment:
    def __init__(
        self,
        method,
        num_derivatives,
        ode_dimension,
        hyper_param_dict,
        jit,
        num_repetitions,
    ) -> None:
        self.hyper_param_dict = hyper_param_dict

        self.method = method
        self.num_derivatives = num_derivatives
        self.ode_dimension = ode_dimension

        self.ivp = tornadox.ivp.lorenz96(
            num_variables=ode_dimension,
            forcing=hyper_param_dict["forcing"],
            t0=hyper_param_dict["t0"],
            tmax=hyper_param_dict["tmax"],
        )
        steprule = tornadox.step.ConstantSteps(0.1)
        init = tornadox.init.RungeKutta()
        self.solver = method(
            num_derivatives=num_derivatives,
            steprule=steprule,
            initialization=init,
        )
        self.init_state = self.solver.initialize(self.ivp)

        self.result = dict()

        # Whether the step function is jitted before timing
        self.jit = jit

        # How often each experiment is run
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
            """Manually do the repeated number of runs, because otherwise jax notices how the outputs are not reused anymore."""

            state = self.solver.attempt_step(
                state=self.init_state, dt=self.hyper_param_dict["dt"]
            )
            for _ in range(self.num_repetitions):
                state = self.solver.attempt_step(
                    state=state, dt=self.hyper_param_dict["dt"]
                )
            return state.y.mean

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
                method=self.solver.__class__.__name__,
                d=self.ode_dimension,
                nu=self.num_derivatives,
                jit=self.jit,
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
        s = f"{self.solver.__class__.__name__} "
        s += "{\n"
        s += f"\td={self.ode_dimension}\n"
        s += f"\tnu={self.num_derivatives}\n"
        s += f"\tjit={self.jit}\n"
        s += f"\tresults={self.result}\n"
        return s + "}"

    def time_function(self, fun):
        # Average time, not minimum time, because we do not want to accidentally
        # run into some of JAX's lazy-execution-optimisation.
        avg_time = timeit.Timer(fun).timeit(number=1) / self.num_repetitions
        return avg_time


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
NUM_REPETITIONS = 10

key = jax.random.PRNGKey(1)
METHODS = (
    tornadox.ek0.ReferenceEK0,
    tornadox.ek0.KroneckerEK0,
    tornadox.ek1.ReferenceEK1,
    tornadox.ek1.DiagonalEK1,
    tornadox.ek1.TruncationEK1,
    tornadox.ek1.EarlyTruncationEK1,
    lambda *args, **kwargs: tornadox.enkf.EnK1(
        *args, **kwargs, ensemble_size=100, prng_key=key
    ),
)
NUM_DERIVS = (2, 4, 8)
ODE_DIMS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
JIT = (True,)

# Define predicate to specify experiments that are not executed later
ignore_exp = lambda method, nu, d, is_jit: (
    # For reference implementations, ignore too high dims
    (method in [tornadox.ek0.ReferenceEK0, tornadox.ek1.ReferenceEK1] and d > 128)
    # For truncated EK1 implementations, ignore even higher dims
    or (
        method
        in [
            tornadox.ek1.TruncationEK1,
            tornadox.ek1.EarlyTruncationEK1,
            tornadox.enkf.EnK1,
        ]
        and d > 256
    )
    # For truncated and diagonal EK1, do not jit
    # or (is_jit and method in ["ek1_truncated", "ek1_diagonal"])
)


EXPERIMENTS = [
    SterilisedExperiment(
        method=M,
        num_derivatives=NU,
        ode_dimension=D,
        hyper_param_dict=HYPER_PARAM_DICT,
        jit=J,
        num_repetitions=NUM_REPETITIONS,
    )
    for (M, NU, D, J) in itertools.product(METHODS, NUM_DERIVS, ODE_DIMS, JIT)
    if not ignore_exp(M, NU, D, J)
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
exp.hyper_parameters.to_json(result_dir / "hparams.json", indent=2)

# Plot results
plotting.plot_exp_1b(result_file)
