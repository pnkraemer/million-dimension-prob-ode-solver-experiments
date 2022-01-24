import jax.numpy as jnp
import tornadox
from scipy.integrate import solve_ivp

ode_dimension = 140

hyper_param_dict = {
    "dt": 0.5,
    "t0": 0.0,
    "tmax": 10.0,
    "forcing": 8.0,
}

lorenz = tornadox.ivp.lorenz96(
    num_variables=ode_dimension,
    forcing=hyper_param_dict["forcing"],
    t0=hyper_param_dict["t0"],
    tmax=hyper_param_dict["tmax"],
)


t_eval = jnp.linspace(hyper_param_dict["t0"], hyper_param_dict["tmax"], 200)

sol = solve_ivp(
    lorenz.f,
    lorenz.t_span,
    y0=lorenz.y0,
    t_eval=t_eval,
    atol=1e-12,
    rtol=1e-12,
    method="LSODA",
)

print(sol.t)
print(sol.y.T)

ygrid = jnp.arange(ode_dimension)

T, Y = jnp.meshgrid(t_eval, ygrid)
print(T.shape, Y.shape, sol.y.shape)


path = "./results/1_sterilised_problem/"
jnp.save(path + "T.npy", T)
jnp.save(path + "X.npy", Y)
jnp.save(path + "Y.npy", sol.y)


import matplotlib.pyplot as plt

vmin = jnp.amin(sol.y)
vmax = jnp.amax(sol.y)

plt.contourf(Y.T, T.T, sol.y.T, cmap="copper", vmin=vmin, vmax=vmax)
plt.contourf(Y.T, T.T, sol.y.T, cmap="copper", vmin=vmin, vmax=vmax)
plt.show()
