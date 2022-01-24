import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=10_000, t0=0.0, tmax=2.0)


sol = solve_ivp(
    ivp.f, ivp.t_span, y0=ivp.y0, atol=1e-8, rtol=1e-8, method="Radau"
)


path = "./results/4_vdp_stiffness_comparison/"
jnp.save(path + "Y.npy", sol.y)


plt.plot(sol.y[0], sol.y[1])
plt.show()
