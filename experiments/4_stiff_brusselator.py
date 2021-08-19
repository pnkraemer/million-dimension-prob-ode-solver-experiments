"""Stiff brusselator."""


import tornado
import scipy.integrate

import matplotlib.pyplot as plt

import jax.numpy as jnp


from matplotlib import cm


for N in [10, 20, 50, 100]:
    ivp = tornado.ivp.brusselator(N=N)

    sol = scipy.integrate.solve_ivp(
        ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0, method="RK45"
    )

    x = jnp.arange(1, N + 1) / N
    t = sol.t

    T, X = jnp.meshgrid(t, x)
    U = sol.y[:N]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(T, X, U, cmap=cm.viridis)
    ax.set_title(f"d={N}, N={len(sol.t)}")
    plt.show()
