"""Test problems."""

import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn
from jax.config import config

config.update("jax_enable_x64", True)


__all__ = ["lorenz96", "lorenz96_jax", "lorenz96_jax_loop"]


def lorenz96(t0=0.0, tmax=30.0, y0=None, params=(5, 8.0)):
    """Lorenz 96 system in standard (numpy) implementation."""

    num_variables, constant_forcing = params
    if y0 is None:
        y0 = np.ones(num_variables) * constant_forcing

    def lorenz96_f_vec(t, y, constant_forcing=8):
        """Lorenz 96 model with constant forcing."""

        A = np.roll(y, shift=-1)
        B = np.roll(y, shift=2)
        C = np.roll(y, shift=1)
        D = y
        return (A - B) * C - D + constant_forcing

    return pn.problems.InitialValueProblem(f=lorenz96_f_vec, t0=t0, tmax=tmax, y0=y0)


def lorenz96_jax(t0=0.0, tmax=30.0, y0=None, params=(5, 8.0)):
    """Lorenz 96 system in JAX implementation."""

    num_variables, constant_forcing = params

    if y0 is None:
        y0 = jnp.ones(num_variables) * constant_forcing

    # Operator representation
    roll_minus_one = _roll_as_matrix_jax(length=num_variables, shift=-1)
    roll_two = _roll_as_matrix_jax(length=num_variables, shift=-1)
    roll_one = _roll_as_matrix_jax(length=num_variables, shift=-1)

    def lorenz_rhs(t, y):
        A = roll_minus_one @ y
        B = roll_two @ y
        C = roll_one @ y
        D = y
        return (A - B) * C - D + constant_forcing

    def lorenz_rhs_smartly(t, y):
        A = jnp.roll(y, shift=-1)
        B = jnp.roll(y, shift=2)
        C = jnp.roll(y, shift=1)
        D = y
        return (A - B) * C - D + constant_forcing

    rhs = lorenz_rhs
    df = jax.jacfwd(rhs, argnums=1)
    ddf = jax.jacrev(df, argnums=1)

    return pn.problems.InitialValueProblem(
        f=rhs, t0=t0, tmax=tmax, y0=y0, df=df, ddf=ddf
    )


def lorenz96_jax_loop(t0=0.0, tmax=30.0, y0=None, params=(5, 8.0)):
    """Lorenz 96 system in JAX implementation, where the RHS is implemented in a loop."""

    num_variables, constant_forcing = params

    if y0 is None:
        y0 = np.ones(num_variables) * constant_forcing

    @jax.jit
    def L96(t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        N = num_variables
        d = jnp.zeros(N)

        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            A = x[(i + 1) % N] - x[i - 2]
            B = x[i - 1]
            C = x[i]
            val = A * B - C + constant_forcing
            d = d.at[i].set(val)
        return d

    lorenz96_df_jax = jax.jacfwd(L96, argnums=1)

    return pn.problems.InitialValueProblem(
        f=L96, df=lorenz96_df_jax, t0=t0, tmax=tmax, y0=y0
    )


def _roll_as_matrix(length, shift, axis=0):
    if axis != 0:
        raise ValueError

    identity = np.eye(length)
    return np.roll(identity, shift=shift, axis=0)


def _roll_as_matrix_jax(length, shift, axis=0):
    if axis != 0:
        raise ValueError

    identity = jnp.eye(length)
    return jnp.roll(identity, shift=shift, axis=0)
