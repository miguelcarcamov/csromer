import numpy as np

# EPS
EPS = np.finfo(np.float32).eps


def sum_operation(mu, x, z, u, rho=[]):
    M = len(rho)
    sum = np.zeros(len(x), dtype=x.dtype)
    for j in range(0, M):
        sum += (mu / rho[j]) * (x - z[j] + u[j])
    return sum


def end_condition(x1, zk1, zk, rho, epsilon_primal=EPS, epsilon_dual=EPS):
    r = x1 - zk1
    s = (1 / rho) * (zk1 - zk)

    norm2_r = np.linalg.norm(r)
    norm2_s = np.linalg.norm(s)

    if norm2_r <= epsilon_primal and norm2_s <= epsilon_dual:
        return True
    else:
        return False


def sdmm(x_init, fx, g, mu, niter, rho=[]):
    M = len(rho)  # Number of priors
    n = len(x_init)
    xk = x_init
    zk = np.ones((M, n))
    zk_old = np.ones((M, n))
    end_condition_array = np.full(M, False)
    zk[:] = x_init
    uk = np.zeros((M, n))
    for k in range(1, niter + 1):

        xk = fx.calculate_prox(xk - sum_operation(mu, xk, zk, uk, rho))
        zk_old = zk
        for i in range(0, M):
            zk[i] = g[i].calculate_prox(xk + uk[i])
            uk[i] = uk[i] + xk - zk[i]
            end_condition_array[i] = end_condition(xk, zk, zk_old, rho[i])
        if end_condition_array.all():
            break
    return xk
