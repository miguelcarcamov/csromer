import numpy as np


def calculate_Q(x, y, fx, fx_grad, lipschitz_constant):
    x_minus_y = x - y
    res = fx(y) + np.dot(x_minus_y, fx_grad(y)) + 0.5 * lipschitz_constant * np.sum(x_minus_y**2)
    return res


def fista_backtracking_algorithm(
    x=None,
    F=None,
    fx=None,
    fx_grad=None,
    g_prox=None,
    lipschitz_constant=None,
    eta=None,
    max_iter=None,
    n=None,
    tol=None,
    verbose=True,
):
    if x is None and n is not None:
        x = np.zeros(n, dtype=np.float32)

    mu = 1
    z = x.copy()

    if eta is None:
        eta = 1.1

    if max_iter is None:
        max_iter = 100
        if verbose:
            print("Iterations set to " + str(max_iter))

    cost_values = np.zeros(max_iter, dtype=np.float32)
    original_lambda = g_prox.get_lambda()

    for it in range(0, max_iter):
        x_old = x.copy()
        f_eval_old = F(x_old)
        cost_values[it] = f_eval_old

        while True:
            zk = z - (fx_grad(z) / lipschitz_constant)
            g_prox.set_lambda(reg=original_lambda / lipschitz_constant)
            xk = g_prox.calc_prox(zk)
            f_eval_inner_loop = fx(xk)
            q_eval = calculate_Q(xk, z, fx, fx_grad, lipschitz_constant)
            if f_eval_inner_loop <= q_eval:
                break
            lipschitz_constant *= eta

        lipschitz_constant /= eta
        mu_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * mu**2))
        x_temp = xk

        if F(x_temp) < f_eval_old:
            #z = (1.0 + (mu - 1.0) / mu_new) * x_temp + ((1.0 - mu) / mu_new) * x
            z = x_temp + ((mu - 1.0) / mu_new) * (x_temp - x)
            x = x_temp
        else:
            #z = (mu / mu_new) * x_temp + (1 - (mu / mu_new)) * x;
            z = x + (mu / mu_new) * (x_temp - x)

        e = np.sum(np.abs(x - x_old)) / len(x)

        if e <= tol:
            if verbose:
                print("Exit due to tolerance: ", e, " < ", tol)
                print("Iterations: ", it + 1)
            break

        if verbose and it % 10 == 0:
            cost = F(x)
            print("Iteration: ", it, " objective function value: {0:0.5f}".format(cost))
        mu = mu_new

    min_cost = F(x)
    return min_cost, x
