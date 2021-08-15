"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import timeit

import jax.numpy as jnp
import tornado
from scipy.integrate import solve_ivp

from source import problems
import tqdm


dt = 0.5
steprule = tornado.step.ConstantSteps(dt)
t0, tmax = 0.0, 3.0
num_steps = (tmax - t0) / dt
constant_forcing = 5.0  # a Lorenz parameter

# Keys: (num_derivatives, ode_dimension)
# Values: (time_reference, time_diagonal)
results = {
    (3, 4): None,
    (3, 8): None,
    (3, 16): None,
    (3, 64): None,
    (3, 128): None,
    (8, 4): None,
    (8, 8): None,
    (8, 16): None,
    (8, 64): None,
    (8, 128): None,
}

print()
for num_derivatives, ode_dimension in tqdm.tqdm(results.keys()):
    # Build a lorenz problem of appropriate size
    ivp = problems.lorenz96_jax(
        params=(ode_dimension, constant_forcing),
        t0=t0,
        tmax=tmax,
        y0=jnp.arange(ode_dimension),
    )

    def timing_initialize():
        """Time the Taylor-mode initialization.

        To be able to compare pure step-speed, we remove this from the overall run time.
        """
        ek1 = tornado.ek1.ReferenceEK1(
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
            steprule=steprule,
        )
        ek1.initialize(ivp=ivp)

    def timing_diagonal():
        """Time the solve() with a diagonal EK1."""

        # Note (Aug 15, 5:13)
        ek1diag = tornado.ek1.DiagonalEK1(
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
            steprule=steprule,
        )
        diagonal_solution = ek1diag.solution_generator(ivp=ivp)
        for idx, state in enumerate(diagonal_solution):
            pass
        return ek1diag.P0 @ state.y.mean

    def timing_reference():
        """Time the solve with a reference (i.e. full) EK1."""
        ek1ref = tornado.ek1.ReferenceEK1(
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
            steprule=steprule,
        )
        reference_solution = ek1ref.solution_generator(ivp=ivp)
        for idx, state in enumerate(reference_solution):
            pass
        return ek1ref.P0 @ state.y.mean

    # Assert they deliver similar results.
    diagonal_approx = timing_diagonal()
    reference_approx = timing_reference()
    assert jnp.allclose(diagonal_approx, reference_approx, rtol=1e-1, atol=1e-1), (
        diagonal_approx,
        reference_approx,
    )

    # Do the timing
    time_initialize = timeit.Timer(timing_initialize).timeit(number=1)
    time_reference = timeit.Timer(timing_reference).timeit(number=1)
    time_diagonal = timeit.Timer(timing_diagonal).timeit(number=1)

    # Save results
    results[(num_derivatives, ode_dimension)] = (
        time_reference - time_initialize,
        time_diagonal - time_initialize,
    )


for n, d in results.keys():
    t_full, t_diag = results[(n, d)]
    # Print results (all in seconds)
    print()
    print("------------------------")
    print("Number of derivatives:", n)
    print("ODE dimension:", d)
    print()

    print("Reference (no init):", t_full)
    print("Diagonal (no init):", t_diag)
    print()
