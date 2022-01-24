import pathlib

import jax.numpy as jnp
import jax.random
import tornadox

from hose import plotting

# Specify the IVP
key = jax.random.PRNGKey(seed=2)
IVP = tornadox.ivp.fhn_2d(
    bbox=[[-1.25, -1.25], [1.25, 1.25]], dx=0.01, tmax=20.0, prng_key=key
)
D = len(IVP.y0)
print(f"Dimension: {D}")


# Specify the solver
steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-1)
solver = tornadox.ek1.DiagonalEK1(
    num_derivatives=2,
    steprule=steprule,
    initialization=tornadox.init.RungeKutta(use_df=False),
)


# Solve the problem
saveat = [0.0, 4.0, 8.0, 16.0, 20.0]
ts, y_means, y_vars = [], [], []
for state, info in solver.solution_generator(IVP, stop_at=saveat, progressbar=True):
    if state.t in saveat:
        ts.append(state.t)
        y_means.append(state.y.mean[0, :])
        y_vars.append(state.y.cov[:, 0, 0])
ts.append(state.t)
y_means.append(state.y.mean[0, :])
y_vars.append(state.y.cov[:, 0, 0])
y_stds = [jnp.sqrt(v) for v in y_vars]
ts = jnp.stack(ts)
y_means = jnp.stack(y_means)
y_stds = jnp.stack(y_stds)


# Save the solution
result_dir = pathlib.Path("./results/0_diagonalek1_pdesolution")
result_dir.mkdir(parents=True, exist_ok=True)
jnp.save(result_dir / "times.npy", ts)
jnp.save(result_dir / "means.npy", y_means)
jnp.save(result_dir / "stddevs.npy", y_stds)

# Plot the results
plotting.plot_0_diagonalek1_pdesolution(result_dir)
