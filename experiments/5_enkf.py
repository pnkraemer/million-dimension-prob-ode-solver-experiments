import itertools

import jax
import tornadox

MUS = [10.0]  # [10.0, 100.0, 1000.0]
NUS = [4]  # [3, 4, 5]
TOLERANCES = [10.0 ** (-p) for p in range(1, 9)]
ENSEMBLE_SIZES = [10 ** p for p in range(1, 6)]
SEEDS = [1, 42, 54321]


results = {
    "stiffness_constant": [],
    "num_derivatives": [],
    "tol": [],
    "ensemble_size": [],
    "seed": [],
    "n_steps": [],
}

for stiffness_constant, num_derivatives, tol, ensemble_size, seed in itertools.product(
    MUS, NUS, TOLERANCES, ENSEMBLE_SIZES, SEEDS
):
    vdp = tornadox.ivp.vanderpol(
        t0=0.0, tmax=float(stiffness_constant), stiffness_constant=stiffness_constant
    )
    steprule = tornadox.step.AdaptiveSteps(abstol=tol, reltol=tol)
    ek1 = tornadox.enkf.EnK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        ensemble_size=ensemble_size,
        prng_key=jax.random.PRNGKey(seed),
    )
    solution_generator = ek1.solution_generator(vdp)
    state, info = None, None
    try:
        for state, info in solution_generator:
            pass

    except ValueError:
        print("Interrupted after not finding good step width")

    n_steps = info["num_steps"]
