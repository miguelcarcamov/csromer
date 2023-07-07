import numpy as np


def calculate_Q(x, y, q_core, q_core_grad, lipschitz_constant):
    x_minus_y = x - y
    res = q_core + np.dot(x_minus_y, q_core_grad) + 0.5 * lipschitz_constant * np.sum(x_minus_y**2)
    return res


def fista_backtracking_algorithm(
    x_init=None,
    F_obj=None,
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
    if x_init is None and n is not None:
        x_init = np.zeros(n, dtype=np.float32)

    min_iter = 100
    iterations_delta = 50
    mu = 1
    x = x_init.copy()
    z = x

    if eta is None:
        eta = 1.1

    if tol is None:
        tol = np.finfo(np.float32).tiny

    if max_iter is None:
        max_iter = 10000
        if verbose:
            print("Iterations set to " + str(max_iter))

    cost_values = np.zeros(max_iter, dtype=np.float32)
    original_lambda = g_prox.get_lambda()

    for it in range(0, max_iter):
        x_old = x.copy()
        f_eval_old = F_obj.evaluate(x_old)
        cost_values[it] = f_eval_old
        if verbose and it % 100 == 0:
            print("Iteration: ", it, " objective function value: {0:0.5f}".format(cost_values[it]))

        q_core = fx(z)
        grad_z = fx_grad(z)
        while True:
            zk = z - (grad_z / lipschitz_constant)
            g_prox.set_lambda(reg=original_lambda / lipschitz_constant)
            xk = g_prox.calc_prox(zk)
            f_eval_inner_loop = fx(xk)
            q_eval = calculate_Q(xk, z, q_core, grad_z, lipschitz_constant)
            if f_eval_inner_loop <= q_eval:
                break
            lipschitz_constant *= eta

        lipschitz_constant /= eta
        mu_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * mu**2))
        x_temp = xk

        if F_obj.evaluate(x_temp) < f_eval_old:
            z = x_temp + ((mu - 1.0) / mu_new) * (x_temp - x)
            x = x_temp
        else:
            z = x + (mu / mu_new) * (x_temp - x)

    # e = np.sum(np.abs(x - x_old)) / len(x)
        if it > min_iter and cost_values[it - iterations_delta] - cost_values[it] < tol:
            break
    # if e <= tol:
    #     if verbose:
    #         print("Exit due to tolerance: ", e, " < ", tol)
    #         print("Iterations: ", it + 1)
    #      break

        mu = mu_new

    min_cost = F_obj.evaluate(x)
    return min_cost, x
