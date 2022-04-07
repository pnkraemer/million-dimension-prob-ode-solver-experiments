"""How good is the calibration?"""


import jax.numpy as jnp
from tornadox import ek0, ek1, init, step, ivp
import scipy.integrate
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from tueplots import bundles, axes
import pathlib


NUM_DERIVATIVES = 4


def error(*, sol, state_ref):
    difference = sol.y.mean[0] - state_ref
    return jnp.linalg.norm(difference / state_ref) / jnp.sqrt(sol.y.mean[0].size)


def chi2(*, sol, state_ref, solver):

    if sol.y.cov_sqrtm.shape[0] == 2:
        cov = jnp.diag(sol.y.cov[:, 0, 0])

    elif sol.y.cov_sqrtm.shape[0] > 2:
        cov = solver.P0 @ sol.y.cov @ solver.P0.T
    else:
        raise ValueError
    difference = sol.y.mean[0] - state_ref

    assert cov.shape == (2, 2), cov.shape
    assert difference.shape == (2,), difference.shape

    return difference @ jnp.linalg.pinv(cov) @ difference


def tolerance_to_final_state_diagonal(*, tolerance):
    step_rule = step.AdaptiveSteps(abstol=tolerance, reltol=tolerance)
    ek1_diagonal = ek1.DiagonalEK1(steprule=step_rule)
    sol_diagonal, _ = ek1_diagonal.simulate_final_state(vdp)
    return sol_diagonal.y.mean[0], jnp.diag(sol_diagonal.y.cov[:, 0, 0])


def tolerance_to_final_state_reference(*, tolerance):
    step_rule = step.AdaptiveSteps(abstol=tolerance, reltol=tolerance)
    ek1_reference = ek1.DiagonalEK1(steprule=step_rule)
    sol_reference, _ = ek1_reference.simulate_final_state(vdp)
    return (
        sol_diagonal.y.mean[0],
        ek1_reference.P0 @ sol_reference.y.cov @ ek1_reference.P0.T,
    )


# Create an IVP
vdp = ivp.vanderpol(t0=0.0, tmax=1.0, stiffness_constant=1)


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


for tol in tqdm(0.1 ** jnp.arange(1.0, 12.0, step=0.5)):

    # Create solver.
    step_rule = step.AdaptiveSteps(abstol=tol, reltol=tol)
    init_ = init.CompiledRungeKutta()
    ek1_diagonal = ek1.DiagonalEK1(
        steprule=step_rule, initialization=init_, num_derivatives=NUM_DERIVATIVES
    )
    ek1_reference = ek1.ReferenceEK1(
        steprule=step_rule, initialization=init_, num_derivatives=NUM_DERIVATIVES
    )

    # Solve IVP.
    sol_diagonal, _ = ek1_diagonal.simulate_final_state(vdp, compile_step=True)
    sol_reference, _ = ek1_reference.simulate_final_state(vdp, compile_step=True)

    # Time the solution, now that the solver has been compiled
    time_start = time.time()
    res, _ = ek1_diagonal.simulate_final_state(vdp, compile_step=True)
    res.y.mean.block_until_ready()
    time_end = time.time()
    times_diagonal.append(time_end - time_start)

    time_start = time.time()
    res, _ = ek1_reference.simulate_final_state(vdp, compile_step=True)
    res.y.mean.block_until_ready()
    time_end = time.time()
    times_reference.append(time_end - time_start)

    # Compute errors
    error_diagonal = error(sol=sol_diagonal, state_ref=reference_state)
    error_reference = error(sol=sol_reference, state_ref=reference_state)

    chi2_diagonal = chi2(
        sol=sol_diagonal, state_ref=reference_state, solver=ek1_diagonal
    )
    chi2_reference = chi2(
        sol=sol_reference, state_ref=reference_state, solver=ek1_reference
    )

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


RESULT_DIR = pathlib.Path("./results/5_extra_experiments")
if not RESULT_DIR.is_dir():
    RESULT_DIR.mkdir(parents=True)


path = "./results/5_extra_experiments/"
jnp.save(path + f"errors_ek1_diagonal_{NUM_DERIVATIVES}.npy", errors_diagonal)
jnp.save(path + f"chi2s_ek1_diagonal_{NUM_DERIVATIVES}.npy", chi2s_diagonal)
jnp.save(path + f"times_ek1_diagonal_{NUM_DERIVATIVES}.npy", times_diagonal)

jnp.save(path + f"errors_ek1_reference_{NUM_DERIVATIVES}.npy", errors_reference)
jnp.save(path + f"chi2s_ek1_reference_{NUM_DERIVATIVES}.npy", chi2s_reference)
jnp.save(path + f"times_ek1_reference_{NUM_DERIVATIVES}.npy", times_reference)
