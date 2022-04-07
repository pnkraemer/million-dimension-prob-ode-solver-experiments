"""How good is the calibration?"""


import jax.numpy as jnp
from tornadox import ek0, ek1, init, step, ivp
import scipy.integrate
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from tueplots import bundles, axes

def error(*, sol, state_ref):
    return jnp.linalg.norm(sol.y.mean[0] - state_ref) / jnp.sqrt(sol.y.mean[0].size)

def chi2(*, sol, state_ref, solver):

    if sol.y.cov_sqrtm.shape == (2,5,5):
        cov = jnp.diag(sol.y.cov[:, 0, 0])

    elif sol.y.cov_sqrtm.shape == (10, 10):
        cov = solver.P0 @ sol.y.cov @ solver.P0.T

    difference = sol.y.mean[0] - state_ref

    assert cov.shape == (2,2), cov.shape
    assert difference.shape == (2,), difference.shape

    return difference @ jnp.linalg.pinv(cov) @ difference

# Create an IVP
vdp = ivp.vanderpol(t0=0., tmax=10., stiffness_constant=0.5)


# Reference solution
reference_sol = scipy.integrate.solve_ivp(
    fun=vdp.f,
    t_span=(vdp.t0, vdp.tmax),
    y0=vdp.y0,
    method="LSODA",
    jac=vdp.df,
    atol=1e-12,
    rtol=1e-12,
)
reference_state = reference_sol.y[:, -1]




# Create a solver.
errors_diagonal = []
errors_reference = []
chi2s_diagonal = []
chi2s_reference = []
tols = []
times_diagonal = []
times_reference = []


for tol in tqdm(0.1 ** jnp.arange(1., 10., step=0.5)):

    # Create solver.
    step_rule = step.AdaptiveSteps(abstol=tol, reltol=tol)
    ek1_diagonal = ek1.DiagonalEK1(steprule=step_rule)
    ek1_reference = ek1.ReferenceEK1(steprule=step_rule)

    # Solve IVP.
    sol_diagonal, _ = ek1_diagonal.simulate_final_state(vdp)
    sol_reference, _ = ek1_reference.simulate_final_state(vdp)

    # Time the solution, now that the solver has been compiled
    time_start = time.time()
    ek1_diagonal.simulate_final_state(vdp)
    time_end = time.time()
    times_diagonal.append(time_end - time_start)

    time_start = time.time()
    ek1_reference.simulate_final_state(vdp)
    time_end = time.time()
    times_reference.append(time_end - time_start)

    # Compute errors
    error_diagonal = error(sol=sol_diagonal, state_ref=reference_state)
    error_reference = error(sol=sol_reference, state_ref=reference_state)

    chi2_diagonal = chi2(sol=sol_diagonal, state_ref=reference_state, solver=ek1_diagonal)
    chi2_reference = chi2(sol=sol_reference, state_ref=reference_state, solver=ek1_reference)

    # Append to results
    tols.append(tol)
    errors_diagonal.append(error_diagonal)
    errors_reference.append(error_reference)
    chi2s_diagonal.append(chi2_diagonal)
    chi2s_reference.append(chi2_reference)


errors_diagonal = jnp.asarray(errors_diagonal)
chi2s_diagonal = jnp.asarray(chi2s_diagonal)
times_diagonal = jnp.asarray(times_diagonal)

errors_reference = jnp.asarray(errors_reference)
chi2s_reference = jnp.asarray(chi2s_reference)
times_reference = jnp.asarray(times_reference)



plt.rcParams.update(axes.lines())
plt.rcParams.update(bundles.icml2022(column="full"))
fig, ax = plt.subplots(ncols=2, nrows=1,sharey=True, dpi=400)

ax[0].set_title("Work-precision")
ax[0].loglog(times_reference, errors_reference, "o-", label=r"Reference ($\nu=4$)")
ax[0].loglog(times_diagonal, errors_diagonal, "x-", label=r"Diagonal ($\nu=4$)")
ax[0].legend()
ax[0].set_xlabel("Runtimes")
ax[0].set_ylabel("Final state RMSE")
ax[0].grid()

ax[1].set_title(r"$\chi^2$ (Calibration)")
ax[1].loglog(chi2s_reference, errors_reference, "o-", label=r"Reference ($\nu=4$)")
ax[1].loglog(chi2s_diagonal, errors_diagonal,  "x-",label=r"Diagonal ($\nu=4$)")
ax[1].legend()
ax[1].set_xlabel(r"$\chi^2$ value")
ax[1].axvline(2., color="k")
ax[1].grid()
plt.show()
