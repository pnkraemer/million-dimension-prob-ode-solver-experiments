"""This experiment evaluates whether one can 'properly solve' a mediumm-high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *faster* (as opposed to *possible*).
"""


import numpy as np
from scipy.integrate import solve_ivp
import timeit

from source import problems
from probnum.diffeq import probsolve_ivp
from probnum.problems.zoo.diffeq import lorenz96

sec_to_ms = lambda t: 1000 * t

import tqdm

# Prepare results in a dict
# Keys are (method, ode_dim)
results = {
    ("EK0", 4): None,
    ("EK0", 8): None,
    ("RK45", 4): None,
    ("RK45", 16): None,
    ("RK45", 64): None,
    ("Radau", 4): None,
    ("Radau", 16): None,
}

for method, d in tqdm.tqdm(results.keys()):

    # Define problem
    y0 = np.arange(
        d
    )  # the default y0 in probnum is an equilibrium, we dont want that here
    problem = lorenz96(num_variables=d, y0=y0, t0=0.0, tmax=1.0)

    f = problem.f
    t0, tmax = problem.t0, problem.tmax
    t_span = (t0, tmax)
    y0 = problem.y0

    if method in ["RK45", "Radau"]:
        # Return the solution once for some statistics
        sol = solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)
        num_steps = len(sol.t)

        def time_func():
            solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    else:
        assert method == "EK0"
        sol = probsolve_ivp(
            f,
            t0=t0,
            tmax=tmax,
            y0=y0,
            rtol=1e-3,
            atol=1e-3,
            method=method,
            diffusion_model="dynamic",
            algo_order=4,
        )
        num_steps = len(sol.locations)

        def time_func():
            probsolve_ivp(
                f,
                t0=t0,
                tmax=tmax,
                y0=y0,
                rtol=1e-3,
                atol=1e-3,
                method=method,
                diffusion_model="dynamic",
                algo_order=4,
            )

    timed_solve = timeit.Timer(time_func).timeit(number=1)  # in seconds
    time_per_step = timed_solve / num_steps

    results[(method, d)] = (time_per_step, num_steps)


# Print the results
for method, d in results.keys():
    time_per_step, num_steps = results[(method, d)]
    print()
    print(f"METHOD={method}, d={d}")
    print(f"\ttime_per_step={sec_to_ms(time_per_step)} ms ({num_steps} steps)")
    print()
