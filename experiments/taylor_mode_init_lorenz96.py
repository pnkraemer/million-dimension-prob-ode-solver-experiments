"""Check initialisation for lorenz.


Currently does not work with the following error message:

  File "/home/kraemer/Projects/high-dimensional-ode-solver/pyenv/lib/python3.8/site-packages/jax/experimental/jet.py", line 127, in process_primitive
    rule = jet_rules[primitive]

with "KeyError: squeeze" for the vectorised implementation, and "KeyError: scatter" for the loopy implementation.
This might have to be fixed in jet.

See also 
https://github.com/google/jax/issues/5365


With the current workaround via linear operators, the code works.
In this case, time is not an issue at all, but memory becomes crazy.
"""
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn
import probnum.problems.zoo.diffeq as diffeq_zoo

from source import problems


import tracemalloc

def initialise(ordint, spatialdim, y0):
    """Initialise the Lorenz96 problem for increased order."""

    # THe initial rv is ignored for Taylor mode init
    ibm_transition = pn.statespace.IBM(ordint=ordint, spatialdim=spatialdim)
    ibm_initrv = pn.randvars.Normal(
        mean=np.zeros(ibm_transition.dimension), cov=np.eye(ibm_transition.dimension)
    )
    ibm_initarg = 1.0
    prior_process = pn.randprocs.MarkovProcess(
        transition=ibm_transition, initrv=ibm_initrv, initarg=ibm_initarg
    )

    threebody = problems.lorenz96_jax(y0=y0, params=(spatialdim, 8.0))
    initial_state = pn.diffeq.initialize_odefilter_with_taylormode(
        f=threebody.f, y0=threebody.y0, t0=threebody.t0, prior_process=prior_process
    )
    return initial_state


num_reps = 1
all_times = []
all_mallocs = []

sizes = [64, 128, 256, 512, 1024]
orders = [2, 3, 5, 8]

for order in orders:
    times = []
    mallocs = []
    for num_variables in sizes:

        y0 = np.arange(1, 1 + num_variables)

        tracemalloc.start()
        start_time = time.time()
        for _ in range(num_reps):
            init = initialise(ordint=order, spatialdim=num_variables, y0=y0)
        end_time = time.time()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        top_two_mallocs = top_stats[0].size + top_stats[1].size

        times.append((end_time - start_time) / num_reps)
        mallocs.append(top_two_mallocs)

    all_times.append(times)
    all_mallocs.append(mallocs)
    print(".")

fig, ax = plt.subplots(ncols=2)

for q, times, mallocs in zip(orders, all_times, all_mallocs):
    ax[0].loglog(sizes, times, label=f"$\\nu = {q}$")
    ax[1].loglog(sizes, mallocs, label=f"$\\nu = {q}$")

ax[1].axhline(7.812e+6, label="8GB")
ax[1].axhline(2*7.812e+6, label="16GB")
ax[1].axhline(4*7.812e+6, label="32GB")


ax[0].set_title("Run time")
ax[1].set_title("Allocated memory (roughly)")

ax[1].legend()
ax[0].legend()
ax[0].set_xlabel("Number of variables")
ax[0].set_ylabel("Time [s]")
ax[1].set_xlabel("Number of variables")
ax[1].set_ylabel("Memory [KiB]")


ax[0].grid(which="minor")
ax[1].grid(which="minor")
plt.savefig("../figures/lorenz_taylormode_complexity.pdf")
plt.show()
