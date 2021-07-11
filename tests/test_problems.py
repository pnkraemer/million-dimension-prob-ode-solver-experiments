import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn
import pytest

from source import problems


@pytest.mark.parametrize("ivp", [problems.lorenz96()])
def test_isinstance(ivp):
    assert isinstance(ivp, pn.problems.InitialValueProblem)


@pytest.mark.parametrize("ivp", [problems.lorenz96()])
def test_eval_numpy(ivp):
    f0 = ivp.f(ivp.t0, ivp.y0)
    assert isinstance(f0, np.ndarray)
    if ivp.df is not None:
        df0 = ivp.df(ivp.t0, ivp.y0)
        assert isinstance(df0, np.ndarray)
    if ivp.ddf is not None:
        ddf0 = ivp.ddf(ivp.t0, ivp.y0)
        assert isinstance(ddf0, np.ndarray)


@pytest.mark.parametrize("ivp", [problems.lorenz96_jax(), problems.lorenz96_jax_loop()])
def test_eval_jax(ivp):
    f0 = ivp.f(ivp.t0, ivp.y0)
    assert isinstance(f0, jnp.ndarray)
    if ivp.df is not None:
        df0 = ivp.df(ivp.t0, ivp.y0)
        assert isinstance(df0, jnp.ndarray)
    if ivp.ddf is not None:
        ddf0 = ivp.ddf(ivp.t0, ivp.y0)
        assert isinstance(ddf0, jnp.ndarray)


@pytest.mark.parametrize("ivp", [problems.lorenz96()])
def test_df0_numpy(ivp):
    if ivp.df is not None:
        step = 1e-6

        time = ivp.t0 + 0.1 * np.random.rand()
        direction = step * (1.0 + 0.1 * np.random.rand(len(ivp.y0)))
        increment = step * direction
        point = ivp.y0 + 0.1 * np.random.rand(len(ivp.y0))

        fd_approx = (
            ivp.f(time, point + increment) - ivp.f(time, point - increment)
        ) / (2.0 * step)

        np.testing.assert_allclose(
            fd_approx, ivp.df(time, point) @ direction, rtol=1e-3, atol=1e-3
        )


@pytest.mark.parametrize("ivp", [problems.lorenz96_jax(), problems.lorenz96_jax_loop()])
def test_df0_jax(ivp):

    key = jax.random.PRNGKey(0)

    if ivp.df is not None:
        step = 1e-6

        time = ivp.t0 + 0.1 * jax.random.uniform(key=key)
        key, subkey = jax.random.split(key)

        direction = step * (
            1.0 + 0.1 * jax.random.uniform(key=key, shape=(len(ivp.y0),))
        )
        key, subkey = jax.random.split(key)

        increment = step * direction
        point = ivp.y0 + 0.1 * jax.random.uniform(key=key, shape=(len(ivp.y0),))
        key, subkey = jax.random.split(key)

        fd_approx = (
            ivp.f(time, point + increment) - ivp.f(time, point - increment)
        ) / (2.0 * step)

        np.testing.assert_allclose(
            fd_approx, ivp.df(time, point) @ direction, rtol=1e-3, atol=1e-3
        )
