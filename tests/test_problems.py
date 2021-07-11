from source import problems


import numpy as np




def test_lorenz96_call():
    y0 = np.arange(10)
    t0 = 2.
    f0 = problems.lorenz96_f(t=t0, y=y0)
    np.testing.assert_allclose(y0.shape, f0.shape)



def test_lorenz96_f_vec_and_no_vec():
    y0 = np.arange(1, 14)
    t0 = 2.

    f0 = problems.lorenz96_f(t=t0, y=y0)
    f1 = problems.lorenz96_f_vec(t=t0, y=y0)
    
    np.testing.assert_allclose(f0, f1)