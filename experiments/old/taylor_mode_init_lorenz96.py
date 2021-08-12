"""Check initialisation for lorenz.


Currently does not work with the following error message:

  File "/home/kraemer/Projects/high-dimensional-ode-solver/pyenv/lib/python3.8/site-packages/jax/experimental/jet.py", line 127, in process_primitive
    rule = jet_rules[primitive]

with "KeyError: squeeze" for the vectorised implementation, and "KeyError: scatter" for the loopy implementation.
This might have to be fixed in jet.

See also 
https://github.com/google/jax/issues/5365
https://github.com/google/jax/issues/2431


With the current workaround via linear operators, the code works.
In this case, time is not an issue at all, but memory becomes crazy.
"""
import time
import timeit
import tracemalloc

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn
import probnum.problems.zoo.diffeq as diffeq_zoo
from memory_profiler import memory_usage, profile

from source import problems



class FakeIBM:
    """For Taylor-mode, we do not need the full IBM.

    For high-dimensional problems, full IBM processes take up too much memory.
    The only quantities that are accessed by initialize_odefilter_with_taylormode
    are `ordint`, `spatialdim`, and `dimension`.
    """

    def __init__(self, ordint, spatialdim):
        self.ordint = ordint
        self.spatialdim = spatialdim

    @property
    def dimension(self):
        return (self.ordint + 1) * self.spatialdim


def initialise(ordint, spatialdim, y0):
    """Initialise the Lorenz96 problem for increased order.

    This function will be subject to multiple benchmarks
    (for large number of variables, the code bottleneck will be in the
    initialization, and the remaining code overhead becomes neglectible.
    It is left in for the sake of a simple implementation).
    """

    # The initial rv is ignored for Taylor mode init
    # ibm_transition = pn.statespace.IBM(ordint=ordint, spatialdim=spatialdim)
    ibm_transition = FakeIBM(ordint=ordint, spatialdim=spatialdim)

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


num_reps = 2
all_times = []
all_memories = []

sizes = [4, 8, 16, 32, 64, 128, 256, 512]
orders = [2, 3, 5, 8]

for order in orders:

    times = []
    memories = []

    for num_variables in sizes:

        # Nonzero y0 to avoid "clever" autodiff
        # (which might happen with zeros, for instance)
        y0 = np.arange(1, 1 + num_variables)

        # Call initialisation and time
        start_time = time.time()
        for _ in range(num_reps):
            init = initialise(ordint=order, spatialdim=num_variables, y0=y0)
        end_time = time.time()
        run_time = (end_time - start_time) / num_reps

        # Call initialisation and peakmem
        mem_usage = memory_usage(
            (initialise, (), {"ordint": order, "spatialdim": num_variables, "y0": y0})
        )
        peakmem = max(mem_usage)  # in MB

        # Append results
        times.append(run_time)
        memories.append(peakmem)

    all_times.append(times)
    all_memories.append(memories)


# Plot the results
fig, ax = plt.subplots(ncols=2, figsize=(6, 3))

for q, times, memories in zip(orders, all_times, all_memories):
    ax[0].loglog((sizes), (times), label=f"$\\nu = {q}$")
    ax[1].loglog((sizes), (memories), label=f"$\\nu = {q}$")

# Some reference numbers
# ax[1].axhline(7.812e+6, label="8GB")
# ax[1].axhline(2*7.812e+6, label="16GB")
# ax[1].axhline(4*7.812e+6, label="32GB")

# Legend
ax[0].legend()
ax[1].legend()

# Grid
ax[0].grid(which="minor")
ax[1].grid(which="minor")

# Title and labels
ax[0].set_title("Run time")
ax[1].set_title("Memory usage (roughly)")
ax[0].set_xlabel("Number of variables")
ax[0].set_ylabel("Time [s]")
ax[1].set_xlabel("Number of variables")
ax[1].set_ylabel("Memory [MB]")

# Save the results
plt.tight_layout()
plt.savefig("../figures/lorenz_taylormode_complexity.pdf")
plt.show()
