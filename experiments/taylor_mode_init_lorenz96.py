"""Check initialisation for lorenz.


Currently does not work with the following error message:

  File "/home/kraemer/Projects/high-dimensional-ode-solver/pyenv/lib/python3.8/site-packages/jax/experimental/jet.py", line 127, in process_primitive
    rule = jet_rules[primitive]

with "KeyError: squeeze" for the vectorised implementation, and "KeyError: scatter" for the loopy implementation.
This might have to be fixed in jet.

See also 
https://github.com/google/jax/issues/5365

"""
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn
import probnum.problems.zoo.diffeq as diffeq_zoo

from source import problems


def initialise(ordint):
    """Initialise the Lorenz96 problem for increased order."""

    spatialdim = 4

    # THe initial rv is ignored for Taylor mode init
    ibm_transition = pn.statespace.IBM(ordint=ordint, spatialdim=spatialdim)
    ibm_initrv = pn.randvars.Normal(
        mean=np.zeros(ibm_transition.dimension), cov=np.eye(ibm_transition.dimension)
    )
    ibm_initarg = 1.0
    prior_process = pn.randprocs.MarkovProcess(
        transition=ibm_transition, initrv=ibm_initrv, initarg=ibm_initarg
    )

    threebody = problems.lorenz96_jax()
    initial_state = pn.diffeq.initialize_odefilter_with_taylormode(
        f=threebody.f, y0=threebody.y0, t0=threebody.t0, prior_process=prior_process
    )
    return initial_state


num_reps = 1
orders = range(1, 3)
times = []

for ordint in orders:

    start_time = time.time()
    for _ in range(num_reps):
        initialise(ordint)
    end_time = time.time()

    times.append((end_time - start_time) / num_reps)


plt.semilogy(orders, times)
plt.xlabel("Orders")
plt.ylabel("Time [s]")
plt.savefig("../figures/lorenz_taylormode_complexity.pdf")
plt.show()
