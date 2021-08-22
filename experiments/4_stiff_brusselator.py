"""Stiff brusselator. Can we solve it as successfully with the truncated EK1 as with the full EK1?

This questions shall answer this in terms of answering two questions:
1) does the truncated ek1 solver -- like the non-truncated ek1 solver and non-probabilistic, implicit methods -- require constant number of steps independent of the dimension (which for the brusselator can be translated to: independent of stiffness) to achieve prescribed accuracy?
2) Does it do so while scaling better with the size of the problem than the reference solver?

If the answer is "yes" to both problems, the truncated solver is a success.
Then, it will be interesting to find out at which dimensionality one should start using the truncated version.
"""


import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.integrate
import tornado
from matplotlib import cm


def solve(ivp, solver):
    for idx, state in enumerate(solver.solution_generator(ivp)):
        pass
    if isinstance(solver, tornado.ek1.ReferenceEK1):
        return solver.P0 @ state.y.mean, idx
    return state.y.mean[0], idx


def solve_scipy(ivp, tolerance):
    radau_sol = scipy.integrate.solve_ivp(
        ivp.f,
        t_span=(ivp.t0, ivp.tmax),
        y0=ivp.y0,
        method="Radau",
        atol=tolerance,
        rtol=tolerance,
    )
    return radau_sol.y[:, -1], len(radau_sol.t)


# Careful with the large values here -- the benchmark will take a while
for N in [10, 20, 50, 100]:
    bruss = tornado.ivp.brusselator(N=N)
    t1 = time.time()
    y_ref, num_steps_ref = solve_scipy(bruss, tolerance=1e-10)
    t_ref = time.time() - t1
    for tol in [1e-1, 1e-3, 1e-5, 1e-7]:

        print(f"Results for N={N}, tol={tol}:")

        # Set up solvers
        steps = tornado.step.AdaptiveSteps(0.1, tol, tol)

        # Truncated EK1
        for n in [3, 8]:
            solver = tornado.ek1.TruncationEK1(
                num_derivatives=n, ode_dimension=bruss.dimension, steprule=steps
            )
            t1 = time.time()
            y_pn, num_steps_pn = solve(ivp=bruss, solver=solver)
            t_pn = time.time() - t1
            error_pn = jnp.linalg.norm(y_pn - y_ref) / jnp.sqrt(y_ref.size)

            # Reference EK1
            solver_dense = tornado.ek1.ReferenceEK1(
                num_derivatives=n, ode_dimension=bruss.dimension, steprule=steps
            )
            t1 = time.time()
            y_dense, num_steps_dense = solve(ivp=bruss, solver=solver_dense)
            t_dense = time.time() - t1
            error_dense = jnp.linalg.norm(y_dense - y_ref) / jnp.sqrt(y_ref.size)

            print(
                f"\tTruncatedEK1 (nu={n}): error={error_pn}, N={num_steps_pn}, t={t_pn} sec (t/N={t_pn/num_steps_pn} sec)"
            )
            print(
                f"\tReferenceEK1 (nu={n}): error={error_dense}, N={num_steps_dense}, t={t_dense} sec (t/N={t_dense/num_steps_dense} sec)"
            )

        # Scipy (Radau)
        t1 = time.time()
        y_scipy, num_steps_scipy = solve_scipy(bruss, tolerance=tol)
        t_scipy = time.time() - t1
        error_scipy = jnp.linalg.norm(y_scipy - y_ref) / jnp.sqrt(y_ref.size)

        print(
            f"\tScipy (Radau): error={error_scipy}, N={num_steps_scipy}, t={t_scipy} sec (t/N={t_scipy/num_steps_scipy} sec)"
        )
        print()

    print()
