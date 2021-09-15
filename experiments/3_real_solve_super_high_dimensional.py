"""This experiment evaluates whether one can 'properly solve' a high-dimensional problem.

The goal is to show that with all the changes, high-dimensional ODE solvers are *possible.*
"""
import tornadox
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

########################################################################################
# Reference
# IVP = tornadox.ivp.fhn_2d(dx=0.1)  # works
# IVP = tornadox.ivp.fhn_2d() # works
# IVP = tornadox.ivp.fhn_2d(bbox=[[-1.0, -1.0], [1.0, 1.0]], dx=0.05)
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.05)
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.04)
# IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.02)
IVP = tornadox.ivp.fhn_2d(bbox=[[-2.0, -2.0], [2.0, 2.0]], dx=0.01)
len(IVP.y0)


########################################################################################
# Reference
# reference_sol = solve_ivp(
#     fun=IVP.f, t_span=(IVP.t0, IVP.tmax), y0=IVP.y0, atol=1e-10, rtol=1e-10
# )


########################################################################################
# Ours
steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-2)
solver = tornadox.ek0.KroneckerEK0(
    num_derivatives=3,
    steprule=steprule,
    initialization=tornadox.init.RungeKutta(use_df=False),
)
# Solve:
saveat = jnp.arange(1, 20)  # 20 == IVP.tmax
ts, ys = [], []
for state, info in solver.solution_generator(IVP, stop_at=saveat, progressbar=True):
    if state.t in saveat:
        print(f"saving! t={state.t}")
        ts.append(state.t)
        ys.append(state.y.mean[0, :])
print(f"saving! t={state.t}")
ts.append(state.t)
ys.append(state.y.mean[0, :])


########################################################################################
# Plotting
def plot_y(y):
    d = len(y) // 2
    _d = int(d ** (1 / 2))
    fig = plt.figure()
    plt.imshow(y[:d].reshape(_d, _d), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(extend="both")
    return fig


# Plot the reference solution
# fig = plot_y(reference_sol.y[:, -1])
# plt.title("RK45")
# fig.savefig("RK45.png")

# Plot our solution
fig = plot_y(ys[-1])
plt.title("KroneckerEK0")
# plt.show()
fig.savefig("KroneckerEK0.pdf")
