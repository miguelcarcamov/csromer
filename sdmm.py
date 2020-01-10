import numpy as np


def sum_operation(mu, x, z, u, rho=[]):
    M = len(rho)
    sum = np.zeros(len(x), dtype=x.dtype)
    for j in range(0, M):
        sum += (mu / rho[j]) * (x - z[j] + u[j])
    return sum


def sdmm(x_init, fx, g, mu, niter, rho=[]):
    M = len(rho)  # Number of priors
    n = len(x_init)
    xk = x_init
    zk = np.ones((M, n))
    zk[:] = x_init
    uk = np.zeros((M, n))
    for k in range(1, niter + 1):

        xk = fx.prox(xk - sum_operation(mu, xk, zk, uk, rho))
        for i in range(0, M):
            zk[i] = g[i].prox(xk + uk[i])
            uk[i] = uk[i] + xk - zk[i]
