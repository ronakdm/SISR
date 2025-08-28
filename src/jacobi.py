import numpy as np

# Givens rotation matrix parametrized by angle
def G_numpy(i, j, theta, n):
    J = np.eye(n)
    J[i, i] = np.cos(theta)
    J[j, j] = np.cos(theta)
    J[i, j] = -np.sin(theta)
    J[j, i] = np.sin(theta)
    return J
    
# Jade Iteration for Approximate Diagonalization.
def jade_iter(matrices, Q, l, j):
    r = len(matrices)
    n, _ = matrices[0].shape
    g = np.zeros((2, r))
    for i, m in enumerate(matrices):
        g[0, i] = m[l, l] - m[j, j]
        g[1, i] = m[l, j] + m[j, l]
    g = g @ g.T
    t_on = g[0, 0] - g[1, 1]
    t_off = g[0, 1] + g[1, 0]
    theta = 0.5 * np.arctan2(t_off, t_on + np.sqrt(t_on**2 + t_off**2))
    rot = G_numpy(l, j, theta, n)
    for i, m in enumerate(matrices):
        matrices[i] = rot.T @ m @ rot
    Q = Q @ rot
    return matrices, Q, abs(theta)

# Cyclic Jacobi Algorithm for Approximate Diagonalization
def jacobi_cyclic(matrices, max_iter=10**3, abs_tol=1e-10):
    matrices = np.copy(matrices)
    n, _ = matrices[0].shape
    Q = np.eye(n)
    for _ in range(max_iter):
        max_theta = 0
        for l in range(n-1):
            for j in range(l+1, n):
                matrices, Q, theta = jade_iter(matrices, Q, l, j)
                max_theta = max(max_theta, theta)
        if max_theta < abs_tol:
            break
    return matrices, Q
