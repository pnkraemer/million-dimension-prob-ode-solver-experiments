"""How good is the diagonal EK1 compared to the full EK1?


This script evaluates how much faster a diagonal EK1 is than a full EK1 for increasing dimension.
"""

import timeit

import jax.numpy as jnp
import tornado
from scipy.integrate import solve_ivp

from source import problems

print()
for ode_dimension in [10, 20, 50, 100, 200]:
    print()
    print("ODE dimension:", ode_dimension)
    print()
    num_derivatives = 4
    ivp = problems.lorenz96_jax(
        params=(ode_dimension, 5.0), tmax=5.0, y0=jnp.arange(ode_dimension)
    )
    dt = 0.5
    num_steps = (ivp.tmax - ivp.t0) / dt
    steps = tornado.step.ConstantSteps(dt)

    def timing_initialize():
        ek1 = tornado.ek1.ReferenceEK1(
            num_derivatives=num_derivatives, ode_dimension=ode_dimension, steprule=steps
        )
        ek1.initialize(ivp=ivp)

    def timing_diagonal():
        ek1diag = tornado.ek1.DiagonalEK1(
            num_derivatives=num_derivatives, ode_dimension=ode_dimension, steprule=steps
        )
        diagonal_solution = ek1diag.solution_generator(ivp=ivp)
        for idx, state in enumerate(diagonal_solution):
            pass
        return ek1diag.P0 @ state.y.mean

    def timing_reference():
        ek1ref = tornado.ek1.ReferenceEK1(
            num_derivatives=num_derivatives, ode_dimension=ode_dimension, steprule=steps
        )
        reference_solution = ek1ref.solution_generator(ivp=ivp)
        for idx, state in enumerate(reference_solution):
            pass
        return ek1ref.P0 @ state.y.mean

    diagonal_approx = timing_diagonal()
    reference_approx = timing_reference()

    assert jnp.allclose(diagonal_approx, reference_approx, rtol=1e-1, atol=1e-1), (
        diagonal_approx,
        reference_approx,
    )

    time_initialize = timeit.Timer(timing_initialize).timeit(number=1)
    time_reference = timeit.Timer(timing_reference).timeit(number=1)
    time_diagonal = timeit.Timer(timing_diagonal).timeit(number=1)

    print("Initialization:", time_initialize)
    print("Reference (no init):", (time_reference - time_initialize) / num_steps)
    print("Diagonal (no init):", (time_diagonal - time_initialize) / num_steps)
    print()
