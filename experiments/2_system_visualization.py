import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

ivp = tornadox.ivp.pleiades()


t_eval = jnp.linspace(ivp.t0, ivp.tmax, 200, endpoint=True)
sol = solve_ivp(
    ivp.f, ivp.t_span, y0=ivp.y0, atol=1e-8, rtol=1e-8, method="RK45", t_eval=t_eval
)


path = "./results/2_medium_scale_problem/"
jnp.save(path + "Y.npy", sol.y)
jnp.save(path + "T.npy", sol.t)

fig, ax = plt.subplots(dpi=200)


for i, marker in zip(range(4), ["D", "*", "s", "o"]):
    ax.plot(sol.y.T[:, i], sol.y.T[:, i + 7], color="black", linewidth=2)
    ax.plot(sol.y.T[-1, i], sol.y.T[-1, i + 7], marker=marker, markersize=12)
    ax.plot(sol.y.T[0, i], sol.y.T[0, i + 7], marker="o", markersize=3, color="k")

plt.show()
