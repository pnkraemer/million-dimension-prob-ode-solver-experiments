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
import tornadox
from matplotlib import cm


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

    # Dimension of the brusselator is d=2*N
    bruss = tornadox.ivp.brusselator(N=N)

    # Reference solution
    t1 = time.time()
    y_ref, num_steps_ref = solve_scipy(bruss, tolerance=1e-10)
    t_ref = time.time() - t1

    for tol in [1e-1, 1e-3, 1e-5, 1e-7]:

        print(f"Results for N={N}, tol={tol}:")

        # Set up solvers
        steps = tornadox.step.AdaptiveSteps(abstol=tol, reltol=tol)

        # Truncated EK1
        for n in [3, 8]:
            solver = tornadox.ek1.TruncationEK1(num_derivatives=n, steprule=steps)
            t1 = time.time()
            solution = solver.solve(bruss)
            y_pn, num_steps_pn = solution.mean, len(solution.t)
            t_pn = time.time() - t1
            error_pn = jnp.linalg.norm(y_pn - y_ref) / jnp.sqrt(y_ref.size)
            print(
                f"\tTruncatedEK1 (nu={n}): error={error_pn}, N={num_steps_pn}, t={t_pn} sec (t/N={t_pn/num_steps_pn} sec)"
            )

            # Reference EK1
            solver_dense = tornadox.ek1.ReferenceEK1(num_derivatives=n, steprule=steps)
            t1 = time.time()
            solution = solver.solve(bruss)
            y_dense, num_steps_dense = solution.mean, len(solution.t)
            t_dense = time.time() - t1
            error_dense = jnp.linalg.norm(y_dense - y_ref) / jnp.sqrt(y_ref.size)
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
