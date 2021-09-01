"""Check"""
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import probnum as pn
import probnum.problems.zoo.diffeq as diffeq_zoo

from hose import problems


def initialise(ordint):
    """Initialise the three body problem for increased order."""

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

    threebody = diffeq_zoo.threebody_jax()
    initial_state = pn.diffeq.initialize_odefilter_with_taylormode(
        f=threebody.f, y0=threebody.y0, t0=threebody.t0, prior_process=prior_process
    )
    return initial_state


num_reps = 1
orders = range(20)
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
plt.savefig("../figures/threebody_taylormode_complexity.pdf")
plt.show()
