"""Test problems."""

import numpy as np


__all__ = ["lorenz96_f", "lorenz96_f_vec"]



def lorenz96_f(t, y, constant_forcing=8):
    """Lorenz 96 model with constant forcing."""

    # Setting up vector
    num_variables = len(y)
    d = np.zeros(num_variables)

    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(num_variables):
        d[i] = (y[(i + 1) % num_variables] - y[i - 2]) * y[i - 1] - y[i] + constant_forcing
    return d




def lorenz96_f_vec(t, y, constant_forcing=8):
    """Lorenz 96 model with constant forcing."""

    A = np.roll(y, shift=-1)
    B = np.roll(y, shift=2)
    C = np.roll(y, shift=1)
    D = y
    return (A - B) * C - D + constant_forcing


def lorenz96_df(t, y, constant_forcing=8):
    """Jacobian of the Lorenz 96 model with constant forcing."""

    raise NotImplementedError