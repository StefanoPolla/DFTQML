import numpy as np
import tensorflow as tf
from scipy import optimize


def energy(density, functional, potential):
    return functional(density).numpy().ravel() + np.dot(density, potential)


@tf.function
def model_jac(in_tensor, model):
    with tf.GradientTape() as tape:
        tape.watch(in_tensor)
        outputs = model(in_tensor)
    return tape.jacobian(outputs, in_tensor)


def energy_jac(density, model, potential):
    in_tensor = tf.convert_to_tensor(density)
    return model_jac(in_tensor, model).numpy().ravel() + potential


def optimize_dftio(model, potential, init_density, verbose=False):
    N = round(np.sum(init_density))
    L = len(init_density)
    eq_cons = {"type": "eq", "fun": lambda x: np.sum(x) - N, "jac": lambda x: np.ones_like(x)}
    optres = optimize.minimize(
        energy,
        init_density,
        jac=energy_jac,
        args=(model, potential),
        method="SLSQP",
        constraints=[eq_cons],
        bounds=[[0, 2]] * L,
    )
    dft_energy = model(optres.x).numpy().ravel() 
    if verbose:
        print(optres)
    return optres.x, dft_energy