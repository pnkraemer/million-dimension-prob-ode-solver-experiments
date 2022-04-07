"""How expensive is the initialization?"""

from memory_profiler import memory_usage
import time
import jax.numpy as jnp
import jax
from tornadox import ek0, ek1, init, step, ivp, enkf

from functools import partial


def time_fun(fun):
    t0 = time.time()
    fun()
    fun()
    tmax = time.time() - t0
    print(f"Run time: \n\tt={tmax / 2.}")
    return tmax / 2.0


def peakmem_fun(fun):
    mem_usage = memory_usage(fun)

    # print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print(f"Maximum memory usage: \n\tm={max(mem_usage)}")
    return max(mem_usage)


def initialize(routine, ode, num_derivatives):

    a, b = routine(
        f=ode.f, df=ode.df, y0=ode.y0, t0=ode.t0, num_derivatives=num_derivatives
    )
    a.block_until_ready()
    b.block_until_ready()
    return a, b


print()
num_derivatives = 6

DXs = 0.5 ** jnp.arange(1, 12, step=2)
dims = []
times_runge = []
times_taylor = []
mems_runge = []
mems_taylor = []

for dx in DXs:
    with jax.disable_jit():
        t0 = time.time()

        ode = ivp.fhn_2d(dx=dx)
        taylor_dx = partial(
            initialize,
            routine=init.TaylorMode(),
            ode=ode,
            num_derivatives=num_derivatives,
        )
        runge_dx = partial(
            initialize,
            routine=init.CompiledRungeKutta(dt=1e-4, use_df=False),
            ode=ode,
            num_derivatives=num_derivatives,
        )

        # taylor_dx()

        print()
        print(f"ODE dimension: \n\td={ode.y0.shape[0]}")

        print("Taylor:")
        mem_taylor = peakmem_fun(taylor_dx)
        time_taylor = time_fun(taylor_dx)
        print()
        print("Runge-Kutta:")
        mem_runge = peakmem_fun(runge_dx)
        time_runge = time_fun(runge_dx)
        print()

        dims.append(ode.y0.shape[0])

        times_runge.append(time_runge)
        mems_runge.append(mem_runge)
        mems_taylor.append(mem_taylor)
        times_taylor.append(time_taylor)

print(times_runge, times_taylor, mems_runge, mems_taylor)

import matplotlib.pyplot as plt
from tueplots import bundles, axes


plt.rcParams.update(bundles.icml2022(column="full", usetex=False, ncols=2, nrows=1))
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.grid())
plt.rcParams.update(axes.lines())

fig, ax = plt.subplots(ncols=2, sharey=True, dpi=300)


ax[0].loglog(mems_runge, dims, label="Runge-Kutta")
ax[0].loglog(mems_taylor, dims, label="Taylor")
ax[0].set_ylabel("ODE dimension")
ax[0].set_xlabel("Peak memory")
ax[0].legend()
ax[0].grid()

ax[1].loglog(times_runge, dims, label="Runge-Kutta")
ax[1].loglog(times_taylor, dims, label="Taylor")
ax[1].set_xlabel("Run time [s]")
ax[1].legend()
ax[1].grid()
plt.savefig("INITs.pdf")
plt.show()
