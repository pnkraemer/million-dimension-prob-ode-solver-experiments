"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import itertools
import pathlib
import sys
import timeit

import jax
import jax.numpy as jnp
import pandas as pd
import tornadox

from hose import plotting


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

        # Store relevant keys
        self.method = method
        self.num_derivatives = num_derivatives
        self.ode_dimension = ode_dimension
        self.jit = jit  # Whether the step function is jitted before timing
        self.num_repetitions = num_repetitions  # How often each experiment is run

        # Set up problem
        self.ivp = tornadox.ivp.lorenz96(
            num_variables=ode_dimension,
            forcing=hyper_param_dict["forcing"],
            t0=hyper_param_dict["t0"],
            tmax=hyper_param_dict["tmax"],
        )

        # Set up solver and compute initial state
        steprule = tornadox.step.ConstantSteps(0.1)
        init = tornadox.init.CompiledRungeKutta(dt=0.001, use_df=False)
        self.solver = method(
            num_derivatives=num_derivatives,
            steprule=steprule,
            initialization=init,
        )
        self.init_state = self.solver.initialize(*self.ivp)

        # Prepare results
        self.result = dict()

    def time_attempt_step(self):
        def _run_attempt_step():
            """Manually do the repeated number of runs, because otherwise jax notices how the outputs are not reused anymore."""

            state, _ = self.solver.attempt_step(
                self.init_state, self.hyper_param_dict["dt"], *self.ivp
            )
            for _ in range(self.num_repetitions):
                state, _ = self.solver.attempt_step(
                    state, self.hyper_param_dict["dt"], *self.ivp
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


def experiment_generator(
    methods, num_derivs, ode_dims, jit, hyper_param_dict, num_repetitions, ignore_exp
):
    """Generate sterilised experiment runners.

    (Generators, because we do only ever want to have a single object in memory.)
    """
    for (m, nu, d, j) in itertools.product(methods, num_derivs, ode_dims, jit):
        if not ignore_exp(m, nu, d, j):
            yield SterilisedExperiment(
                method=m,
                num_derivatives=nu,
                ode_dimension=d,
                hyper_param_dict=hyper_param_dict,
                jit=j,
                num_repetitions=num_repetitions,
            )


def experiment_result_dataframe_generator(experiments):
    """Carry out experiments and store results in a dataframe.

    (Generators, because we do only ever want to have a single object in memory.)
    """
    for exp in experiments:
        exp.time_attempt_step()
        print(exp)
        yield exp.to_dataframe()


def dict_to_dataframe(hyper_params):
    def _aslist(arg):
        try:
            return list(arg)
        except TypeError:
            return [arg]

    hparams = {k: _aslist(v) for k, v in hyper_params.items()}
    return pd.DataFrame(hparams)


# ######################################################################################
#   BEGIN EXPERIMENTS
# ######################################################################################

RESULT_DIR = pathlib.Path("./results/1_sterilised_problem")
if not RESULT_DIR.is_dir():
    RESULT_DIR.mkdir(parents=True)

# Set up benchmarking hyperparameters
HYPER_PARAM_DICT = {
    "dt": 0.5,
    "t0": 0.0,
    "tmax": 3.0,
    "forcing": 8.0,
}
NUM_REPETITIONS = 10

# Select what is to be benchmarked.
# Do not benchmark the truncation variants for now (because otherwise the plots become too cluttered)
METHODS = (
    tornadox.ek0.KroneckerEK0,
    tornadox.ek1.DiagonalEK1,
    tornadox.ek0.DiagonalEK0,
    tornadox.ek0.ReferenceEK0,
    tornadox.ek1.ReferenceEK1,
)
NUM_DERIVS = (6, 4, 2)
ODE_DIMS = (
    16777216,
    8388608,
    4194304,
    2097152,
    1048576,
    524288,
    262144,
    131072,
    65536,
    32768,
    16384,
    8192,
    2048,
    512,
    128,
    32,
    8,
)
JIT = (False,)


# Ignore specific, super costly combinations
_REF_METHODS = [tornadox.ek0.ReferenceEK0, tornadox.ek1.ReferenceEK1]
_CUBIC_METHODS = [
    tornadox.ek1.TruncationEK1,
    tornadox.ek1.EarlyTruncationEK1,
    tornadox.enkf.EnK1,
]
IGNORE_EXP = lambda method, nu, d, is_jit: (
    # For reference implementations, ignore too high dims
    (method in _REF_METHODS and d > 128)
    # For truncated EK1 implementations, ignore even higher dims
    or (method in _CUBIC_METHODS and d > 256)
    # Use only the KroneckerEK0 in high dimensions
    or (method != tornadox.ek0.KroneckerEK0 and d > 8192 * 4)
)


# Actual experiment runs
EXP_GEN = experiment_generator(
    METHODS, NUM_DERIVS, ODE_DIMS, JIT, HYPER_PARAM_DICT, NUM_REPETITIONS, IGNORE_EXP
)
EXP_RESULTS = experiment_result_dataframe_generator(EXP_GEN)
EXP_DATA_FRAMES = list(EXP_RESULTS)


# Merge experiments into single data frame
MERGED_DATA_FRAME = pd.concat(EXP_DATA_FRAMES, ignore_index=True)


# Save results as CSV
RESULT_FILE = RESULT_DIR / "results.csv"
MERGED_DATA_FRAME.to_csv(RESULT_FILE, sep=";", index=False)
dict_to_dataframe(HYPER_PARAM_DICT).to_json(RESULT_DIR / "hparams.json", indent=2)

# Plot results
plotting.plot_exp_1(RESULT_FILE)
