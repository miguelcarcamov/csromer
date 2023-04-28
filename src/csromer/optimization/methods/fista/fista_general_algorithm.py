import numpy as np


def fista_general_algorithm(
    x=None,
    F=None,
    fx=None,
    g_prox=None,
    lipschitz_constant=None,
    max_iter=None,
    n=None,
    verbose=True,
):
    if x is None and n is not None:
        x = np.zeros(n, dtype=np.float32)

    if lipschitz_constant is None:
        lipschitz_constant = 1.

    t = 1
    z = x.copy()
    g_prox.set_lambda(reg=g_prox.get_lambda() / lipschitz_constant)

    if max_iter is None:
        max_iter = 100
        if verbose:
            print("Iterations set to " + str(max_iter))

    for it in range(0, max_iter):
        x_old = x.copy()
        z = z - fx(z) / lipschitz_constant
        x = g_prox.calc_prox(z)

        t0 = t
        t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        z = x + ((t0 - 1.0) / t) * (x - x_old)

        if verbose and it % 10 == 0:
            cost = F(x)
            print("Iteration: ", it, " objective function value: {0:0.5f}".format(cost))

    min_cost = F(x)
    return min_cost, x
