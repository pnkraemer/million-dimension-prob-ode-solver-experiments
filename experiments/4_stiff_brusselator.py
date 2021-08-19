"""Stiff brusselator."""


import tornado
import scipy.integrate

import matplotlib.pyplot as plt

import jax.numpy as jnp


from matplotlib import cm

import jax


def extract_filter(N):
    ivp = tornado.ivp.brusselator(N=N, tmax=10.0)
    odefilter_sol, solver = tornado.ivpsolve.solve(
        ivp=ivp,
        method="ek1_truncated",
        adaptive=True,
        save_every_step=False,
        num_derivatives=3,
        abstol=1e-2,
        reltol=1e-2,
    )
    return (solver.P0 @ jnp.stack(odefilter_sol.y.mean))[:N]


def extract_scipy(N):
    ivp = tornado.ivp.brusselator(N=N, tmax=10.0)

    lsoda_sol = scipy.integrate.solve_ivp(
        ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0, method="LSODA"
    )
    return lsoda_sol.y[:N, -1]


for N in [10, 20, 50, 100]:

    y_ref = extract_scipy(N)
    y = extract_filter(N)

    print(jnp.linalg.norm(y_ref - y) / jnp.sqrt(y.size))

    # x = jnp.arange(1, N + 1) / N
    # t = jnp.stack(sol.t)

    # T, X = jnp.meshgrid(t, x)
    # U = jnp.stack(sol.mean)[:, :N].T

    # assert T.shape == X.shape == U.shape, (T.shape, X.shape, U.shape)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(T, X, U, cmap=cm.viridis)
    # ax.set_title(f"d={N}, N={len(sol.t)}")
    # plt.show()
