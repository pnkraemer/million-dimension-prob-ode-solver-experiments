"""This experiment evaluates whether one can 'properly solve' a high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *possible.*
"""
import pathlib
import tornadox
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from celluloid import Camera

########################################################################################
# Initial Value Problem to Solve
# IVP = tornadox.ivp.fhn_2d(dx=0.2)  # works
# IVP = tornadox.ivp.fhn_2d(dx=0.05)  # 800
IVP = tornadox.ivp.fhn_2d()  # works
# IVP = tornadox.ivp.fhn_2d(bbox=[[-1.0, -1.0], [1.0, 1.0]], dx=0.02)
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.05)  # 12.8k
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.04)
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.02)  # 80k
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.01)  # 320k
# IVP = tornadox.ivp.fhn_2d(bbox=[[-4.0, -4.0], [4.0, 4.0]], dx=0.01)  # 1.28m
D = len(IVP.y0)
print(f"Dimension: {D}")


########################################################################################
# Solve
def solve(method, IVP, saveat_dt=0.25):
    steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-2)
    solver = method(
        num_derivatives=3,
        steprule=steprule,
        initialization=tornadox.init.RungeKutta(use_df=False),
    )

    # Solve:
    saveat = jnp.arange(saveat_dt, IVP.tmax, saveat_dt)  # 20 == IVP.tmax
    ts, y_means, y_vars = [], [], []
    for state, info in solver.solution_generator(IVP, stop_at=saveat, progressbar=True):
        if state.t in saveat:
            print(f"saving! t={state.t}")
            ts.append(state.t)
            y_means.append(state.y.mean[0, :])
            y_vars.append(state.y.cov[:, 0, 0])
    print(f"saving! t={state.t}")
    ts.append(state.t)
    y_means.append(state.y.mean[0, :])
    y_vars.append(state.y.cov[:, 0, 0])
    y_stds = [jnp.sqrt(v) for v in y_vars]
    return ts, y_means, y_stds


########################################################################################
# Visualization
def plot_y(y, fig=None, colorbar=True, **kwargs):
    if fig is None:
        fig = plt.figure()
    d = len(y) // 2
    _d = int(d ** (1 / 2))
    cm = plt.imshow(y[:d].reshape(_d, _d), **kwargs)
    if colorbar:
        fig.colorbar(cm, extend="both")
    return fig


def make_gif(ts, ys, path, fps=12, **kwargs):
    d = len(ys[-1]) // 2
    _d = int(d ** (1 / 2))

    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(ts)):
        plot_y(ys[i], fig=fig, colorbar=False, **kwargs)
        plt.text(0, 2, f"t={ts[i]:.10f}")
        camera.snap()
    plt.colorbar(extend="both")
    animation = camera.animate()
    animation.save(path, fps=fps)


########################################################################################
# Actually solve the problem
# IVP = tornadox.ivp.fhn_2d(tmax=10, dx=0.025)  # 800
method, solvername = tornadox.ek1.DiagonalEK1, "diagonalek1"
# method, solvername = tornadox.ek0.DiagonalEK0, "diagonalek0"
# method, solvername = tornadox.ek0.KroneckerEK0, "kroneckerek0"
ts, y_means, y_stds = solve(method, IVP)


########################################################################################
# Make plots
result_dir = pathlib.Path("./results/3_real_solve_super_high_dimensional")
method_result_dir = result_dir / solvername
method_result_dir.mkdir(parents=True, exist_ok=True)

# for i, (t, ym, ystd) in enumerate(zip(ts, y_means, y_stds)):

    i=0
    fig = plot_y(ym, cmap="coolwarm", vmin=-1, vmax=1)
    fig.savefig(method_result_dir / f"{solvername}_mean_{i}.png")

    # fig = plot_y(jnp.log10(ystd),
                 # vmin=-6, vmax=-2
                 # )
    fig = plot_y(jnp.log10(ystd))
    fig.savefig(method_result_dir / f"{solvername}_std_{i}.png")

#     plt.close("all")

make_gif(
    ts,
    y_means,
    path=method_result_dir / f"means.gif",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
make_gif(
    ts,
    [jnp.log10(y) for y in y_stds],
    method_result_dir / f"stddevs.gif",
    vmin=-10,
    vmax=-4,
)
