"""This experiment evaluates whether one can 'properly solve' a high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *possible.*

The example is taken from:
https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320
"""

import numpy as np
from scipy.integrate import solve_ivp

import timeit

from source import problems



N = 8
problem = problems.advection_diffusion(N=8)
f = problem.f
t_span = (problem.t0, problem.tmax)
y0 = problem.y0


for method in ["RK45", "Radau"]:
    # return the solution once for some statistics

    sol = solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    def time_func():
        solve_ivp(f, t_span=t_span, y0=y0, rtol=1e-8, atol=1e-8, method=method)

    timed_solve = timeit.Timer(time_func).timeit(number=1)
    time_per_step = timed_solve/len(sol.t)
    sec_to_ms = lambda t: 1000 * t

    print()
    print(f"METHOD={method}")
    print()
    print(f"The solve for N={N} (d={problem.y0.shape[0]} ODE dimensions) took {timed_solve} s.")
    print(f"The average time per step taken was {sec_to_ms(time_per_step)} ms ({len(sol.t)} steps)")
    print()