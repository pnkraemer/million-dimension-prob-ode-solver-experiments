"""This experiment evaluates whether one can 'properly solve' a mediumm-high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *faster* (as opposed to *possible*).
"""


import numpy as np
from scipy.integrate import solve_ivp
import timeit

from source import problems

from probnum.problems.zoo.diffeq import lorenz96

sec_to_ms = lambda t: 1000 * t


# Prepare results in a dict
# Keys are (method, ode_dim)
results = {
    ("RK45", 4): None,
    ("RK45", 16): None,
    ("RK45", 64): None,
    ("RK45", 256): None,
    ("Radau", 4): None,
    ("Radau", 16): None,
    ("Radau", 64): None,
}

for method, d in results.keys():

    # Define problem
    y0 = np.arange(d) # the default y0 in probnum is an equilibrium, we dont want that here
    problem = lorenz96(num_variables=d, y0=y0, t0=0., tmax=1.)


    f = problem.f
    t0, tmax = problem.t0, problem.tmax
    t_span = (t0, tmax)
    y0 = problem.y0

    # Return the solution once for some statistics
    sol = solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)
    num_steps = len(sol.t)



    def time_func():
        solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    timed_solve = timeit.Timer(time_func).timeit(number=1)  # in seconds
    time_per_step = timed_solve/num_steps

    results[(method, d)] = (time_per_step, num_steps)


# Print the results
for method, d in results.keys():
    time_per_step, num_steps = results[(method, d)]
    print()
    print(f"METHOD={method}, d={d}")
    print(f"\ttime_per_step={sec_to_ms(time_per_step)} ms ({num_steps} steps)")
    print()