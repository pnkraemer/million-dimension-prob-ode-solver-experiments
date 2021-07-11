
import numpy as np


__all__ = ["lorenz96"]


def lorenz96(t, y, num_variables=5, constant_forcing=8):
    """Lorenz 96 model with constant forcing."""

    # Setting up vector
    d = np.zeros(num_variables)

    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(num_variables):
        d[i] = (y[(i + 1) % num_variables] - y[i - 2]) * y[i - 1] - y[i] + constant_forcing
    return d
