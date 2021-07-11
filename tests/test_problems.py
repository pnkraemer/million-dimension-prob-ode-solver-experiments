from source import problems


import numpy as np




def test_lorenz96_call():
    y0 = np.arange(10)
    t0 = 2.
    f0 = problems.lorenz96(t=t0, y=y0, num_variables=10)
    np.testing.assert_allclose(y0.shape, f0.shape)