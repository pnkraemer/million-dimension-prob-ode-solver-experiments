"""This experiment evaluates whether one can 'properly solve' a high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *possible.*

The example is taken from:
https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320
"""

import numpy as np
from scipy.integrate import solve_ivp

import timeit


a2 = 1.0
a3 = 1.0
b1 = 1.0
b2 = 1.0
b3 = 1.0
r1 = 1.0
r2 = 1.0
_DD = 100.0
g1 = 0.1
g2 = 0.1
g3 = 0.1

# Try 8, 16, 32: 8 is
N = 8

X = np.reshape([j + 1 for i in range(N) for j in range(N)], (N, N))
Y = np.reshape([i + 1 for i in range(N) for j in range(N)], (N, N))
a1 = 1.0 * (X >= 4 * N / 5)

Mx = np.diag([1.0 for i in range(N - 1)], -1) + np.diag(
    [-2.0 for i in range(N)], 0) + np.diag([1.0 for i in range(N - 1)], 1)
My = np.diag([1.0 for i in range(N - 1)], -1) + np.diag(
    [-2.0 for i in range(N)], 0) + np.diag([1.0 for i in range(N - 1)], 1)
Mx[1, 0] = 2.0
Mx[N - 2, N - 1] = 2.0
My[0, 1] = 2.0
My[N - 1, N - 2] = 2.0

u0 = np.ndarray.flatten(np.zeros((3, N, N)))

# Define the discretized PDE as an ODE function
def f(t, _u):
    u = np.reshape(_u, (3, N, N))
    A = u[0, :, :]
    B = u[1, :, :]
    C = u[2, :, :]

    # MyA = My@A
    top = -2 * A[0, :] + 2 * A[1, :]
    bottom = 2 * A[N - 2, :] - 2 * A[N - 1, :]
    MyA = np.vstack(
        (top, A[0:N - 2, :] - 2 * A[1:N - 1, :] + A[2:N, :], bottom))

    # AMx = A@Mx
    left = (-2 * A[:, 0] + 2 * A[:, 1]).reshape(N, 1)
    right = (2 * A[:, N - 2] - 2 * A[:, N - 1]).reshape(N, 1)
    AMx = np.hstack(
        (left, A[:, 0:N - 2] - 2 * A[:, 1:N - 1] + A[:, 2:N], right))

    DA = _DD * (MyA + AMx)
    dA = DA + a1 - b1 * A - r1 * A * B + r2 * C
    dB = a2 - b2 * B - r1 * A * B + r2 * C
    dC = a3 - b3 * C + r1 * A * B - r2 * C
    return np.ndarray.flatten(np.concatenate([dA, dB, dC]))


tspan = (0., 10.)
t = np.linspace(0, 10, 101)

for method in ["RK45", "Radau"]:
    # return the solution once for some statistics
    sol = solve_ivp(f, t_span=tspan, y0=u0, rtol=1e-8, atol=1e-8, method=method)

    def time_func():
        solve_ivp(f, t_span=tspan, y0=u0, rtol=1e-8, atol=1e-8, method=method)

    timed_solve = timeit.Timer(time_func).timeit(number=1)
    time_per_step = timed_solve/len(sol.t)
    sec_to_ms = lambda t: 1000 * t

    print()
    print(f"METHOD={method}")
    print()
    print(f"The solve for N={N} (d={len(u0)} ODE dimensions) took {timed_solve} s.")
    print(f"The average time per step taken was {sec_to_ms(time_per_step)} ms ({len(sol.t)} steps)")
    print()