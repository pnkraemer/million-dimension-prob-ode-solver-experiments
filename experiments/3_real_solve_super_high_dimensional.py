"""This experiment evaluates whether one can 'properly solve' a high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *possible.*

The example is taken from:
https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320


For the moment, there is no point in running this benchmark with the probabilistic solver as well.
"""

import numpy as np
from scipy.integrate import solve_ivp

import timeit

from source import problems

sec_to_ms = lambda t: 1000 * t

N = 8
problem = problems.advection_diffusion(N=8)
f = problem.f
t_span = (problem.t0, problem.tmax)
y0 = problem.y0


results = {
    ("RK45", 4): None,
    ("RK45", 8): None,
    ("RK45", 16): None,
    ("RK45", 32): None,
    ("Radau", 4): None,
    ("Radau", 8): None,
}

for method, N in results.keys():

    # Define problem
    problem = problems.advection_diffusion(N)
    f = problem.f
    t_span = (problem.t0, problem.tmax)
    y0 = problem.y0

    # Return the solution once for some statistics
    sol = solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    def time_func():
        solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    timed_solve = timeit.Timer(time_func).timeit(number=1)  # in seconds
    num_steps = len(sol.t)
    time_per_step = timed_solve/num_steps

    results[(method, N)] = (time_per_step, num_steps)


# Print the results
for method, N in results.keys():
    time_per_step, num_steps = results[(method, N)]
    print()
    print(f"METHOD={method}, N={N} ({3*N*N}-dimensional problem)")
    print(f"\ttime_per_step={sec_to_ms(time_per_step)} ms ({num_steps} steps)")
    print()


