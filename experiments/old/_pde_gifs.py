import matplotlib.pyplot as plt
import numpy as np
import tornadox
from celluloid import Camera
from scipy.integrate import solve_ivp


def wave_1d_gif():
    IVP = tornadox.ivp.wave_1d()
    N = 100
    t_eval = np.linspace(IVP.t0, IVP.tmax, N)
    reference_sol = solve_ivp(
        fun=IVP.f,
        t_span=(IVP.t0, IVP.tmax),
        y0=IVP.y0,
        t_eval=t_eval,
        # method="LSODA",
        # jac=IVP.df,
    )
    d = len(IVP.y0)
    t, y = reference_sol.t, reference_sol.y[: d // 2, :]
    d = len(y[:, 1])
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(N):
        plt.plot(range(d), y[:, i], color="C0")
        plt.text(0, 1.25, f"t={t[i]:.10f}")
        plt.ylim(-1.5, 1.5)
        camera.snap()
    animation = camera.animate()
    animation.save("wave_1d.gif")


def burgers_1d_gif():
    IVP = tornadox.ivp.burgers_1d()
    N = 100
    t_eval = np.linspace(IVP.t0, IVP.tmax, N)
    reference_sol = solve_ivp(
        fun=IVP.f,
        t_span=(IVP.t0, IVP.tmax),
        y0=IVP.y0,
        t_eval=t_eval,
        # method="LSODA",
        # jac=IVP.df,
    )
    d = len(IVP.y0)
    t, y = reference_sol.t, reference_sol.y
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(N):
        plt.plot(range(d), y[:, i], color="C0")
        plt.text(0, 1.25, f"t={t[i]:.10f}")
        plt.ylim(-1.5, 1.5)
        camera.snap()
    animation = camera.animate()
    animation.save("burgers_1d.gif")


def wave_2d_gif():
    IVP = tornadox.ivp.wave_2d()
    N = 100
    t_eval = np.linspace(IVP.t0, IVP.tmax, N)
    reference_sol = solve_ivp(
        fun=IVP.f,
        t_span=(IVP.t0, IVP.tmax),
        y0=IVP.y0,
        t_eval=t_eval,
        # method="LSODA",
        # jac=IVP.df,
    )
    d = len(IVP.y0) // 2
    t, y = reference_sol.t, reference_sol.y[:d, :]
    _d = int(d ** (1 / 2))
    N = len(reference_sol.t)

    fig = plt.figure()
    camera = Camera(fig)
    for i in range(N):
        plt.imshow(y[:, i].reshape(_d, _d), cmap="bwr", vmin=-0.5, vmax=0.5)
        plt.text(0, 2, f"t={t[i]:.10f}")
        camera.snap()
    plt.colorbar(extend="both")
    animation = camera.animate()
    animation.save("wave_2d.gif")


def wave_2d_gif_3d():
    IVP = tornadox.ivp.wave_2d()
    N = 100
    t_eval = np.linspace(IVP.t0, IVP.tmax, N)
    reference_sol = solve_ivp(
        fun=IVP.f,
        t_span=(IVP.t0, IVP.tmax),
        y0=IVP.y0,
        t_eval=t_eval,
        # method="LSODA",
        # jac=IVP.df,
    )
    d = len(IVP.y0) // 2
    t, y = reference_sol.t, reference_sol.y[:d, :]
    _d = int(d ** (1 / 2))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    _x, _y = np.meshgrid(np.arange(_d), np.arange(_d))
    camera = Camera(fig)
    for i in range(0, N, 2):
        _z = y[:, i].reshape(_d, _d)
        surf = ax.plot_surface(_x, _y, _z, cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.text(0, 30, 1, f"t={t[i]:.10f}")
        ax.set_zlim(-1, 1)
        camera.snap()
    fig.colorbar(surf, extend="both")
    animation = camera.animate()
    animation.save("wave_2d_3d.gif")


def fhn_2d_gif():
    IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]])
    N = 100
    t_eval = np.linspace(IVP.t0, IVP.tmax, N)
    reference_sol = solve_ivp(
        fun=IVP.f,
        t_span=(IVP.t0, IVP.tmax),
        y0=IVP.y0,
        t_eval=t_eval,
        # method="LSODA",
        # jac=IVP.df,
    )
    d = len(IVP.y0) // 2
    t, y = reference_sol.t, reference_sol.y[:d, :]
    _d = int(d ** (1 / 2))

    fig = plt.figure()
    camera = Camera(fig)
    for i in range(0, N, 2):
        plt.imshow(y[:, i].reshape(_d, _d), cmap="coolwarm", vmin=-1, vmax=1)
        plt.text(0, 2, f"t={t[i]:.10f}")
        camera.snap()
    plt.colorbar(extend="both")
    animation = camera.animate()
    animation.save("fhn_2d.gif")