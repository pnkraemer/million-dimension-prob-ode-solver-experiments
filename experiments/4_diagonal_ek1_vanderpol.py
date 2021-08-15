"""How good is the diagonal EK1 compared to the full EK1."""

import jax.numpy as jnp
from scipy.integrate import solve_ivp

import tornado

import timeit

num_derivatives = 4
ode_dimension = 2
ivp = tornado.ivp.vanderpol(t0=0.0, tmax=10.0, stiffness_constant=0.1)
dt = 0.1
num_steps = 1 / dt
steps = tornado.step.ConstantSteps(dt)

ek1ref = tornado.ek1.ReferenceEK1(
    num_derivatives=num_derivatives, ode_dimension=ode_dimension, steprule=steps
)
ek1diag = tornado.ek1.DiagonalEK1(
    num_derivatives=num_derivatives, ode_dimension=ode_dimension, steprule=steps
)


reference_solution = ek1ref.solution_generator(ivp=ivp)
diagonal_solution = ek1diag.solution_generator(ivp=ivp)


def timing_diagonal():
    for idx, state in enumerate(diagonal_solution):
        pass


def timing_reference():
    for idx, state in enumerate(reference_solution):
        pass


time_reference = timeit.Timer(timing_diagonal).timeit(number=1)
time_diagonal = timeit.Timer(timing_reference).timeit(number=1)

print("Reference:", time_reference / num_steps)
print("Diagonal:", time_diagonal / num_steps)
