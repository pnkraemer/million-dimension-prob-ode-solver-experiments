"""Test problems."""

import numpy as np
import jax.numpy as jnp
import jax
import probnum as pn

__all__ = ["lorenz96", "lorenz96_jax"]









def lorenz96(t0=0., tmax=30., y0=None, params=(5, 8.0)):
    
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


def lorenz96_jax(t0=0., tmax=30., y0=None, params=(5, 8.0)):
    num_variables, constant_forcing = params

    if y0 is None:
        y0 = jnp.ones(num_variables) * constant_forcing


    def lorenz96_f_vec_jax(t, y):
        """Lorenz 96 model with constant forcing."""

        A = jnp.roll(y, shift=-1)
        B = jnp.roll(y, shift=2)
        C = jnp.roll(y, shift=1)
        D = y
        return (A - B) * C - D + constant_forcing

    lorenz96_df_jax = jax.jacfwd(lorenz96_f_vec_jax, argnums=1)



    return pn.problems.InitialValueProblem(f=lorenz96_f_vec_jax, df=lorenz96_df_jax, t0=t0, tmax=tmax, y0=y0)
