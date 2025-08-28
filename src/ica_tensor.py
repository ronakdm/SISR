from functools import partial

import numpy as np
import torch

from src.jacobi import jacobi_cyclic


def whiten(X):
    c, t = X.shape
    X_c = X - np.mean(X, axis=1, keepdims=True)
    D, U = np.linalg.eigh(X_c @ X_c.T / t)
    V = np.diag(1 / np.sqrt(D)) @ U.T
    return V @ X_c, V

def process_multi_trial(X, n_timepoints=10**3):
    c, nt = X.shape
    n_trials = nt // n_timepoints
    X_c = np.empty_like(X)
    for i in range(n_trials):
        start, end = i * n_timepoints, (i + 1) * n_timepoints
        X_c[:, start:end] = X[:, start:end] - np.mean(X[:, start:end], axis=1, keepdims=True)
    return X_c


def cumulant(matrix, X):
    c, t = X.shape
    cumulant = np.zeros((c, c))
    for i in range(t):
        x = np.ascontiguousarray(X[:, i])
        cumulant += (1 / t) * np.dot(x, np.dot(matrix, x)) * np.outer(x, x)
    return cumulant


def cumulant_jade(X, i, j):
    c, t = X.shape
    e_i = np.eye(1, c, i)
    e_j = np.eye(1, c, j)
    matrix = np.outer(e_i, e_j)
    cumulant = np.zeros((c, c))
    for k in range(t):
        x = np.ascontiguousarray(X[:, k])
        cumulant += (1 / t) * x[i] * x[j] * np.outer(x, x)
    return cumulant - matrix - matrix.T - np.trace(matrix) * np.eye(c)

def jade(X, max_iter=10**5, abs_tol=1e-10):
    c, t = X.shape
    X_w, V = whiten(X)
    matrices = np.array([cumulant_jade(X_w, i, j) for i in range(c) for j in range(i, c)])
    matrices, Q = jacobi_cyclic(matrices, max_iter, abs_tol)
    B = Q.T @ V
    return B


def fobi(X):
    c, t = X.shape
    X_w, V = whiten(X)
    cumulant_fobi = cumulant(np.eye(c), X_w)
    _, Q = np.linalg.eigh(cumulant_fobi)
    B = Q.T @ V
    return B


def jade_ica(x, max_iter, abs_tol=1e-10):
    N, C, T = x.shape

    unmixing_matrices = []
    for i in range(N):
        unmixing_matrices.append(
            torch.from_numpy(jade(x[i, :, :], max_iter=max_iter, abs_tol=abs_tol))
        )

    sources = [unmixing_matrices[i] @ x[i, :, :] for i in range(N)]

    return {
        "sources": sources,
        "unmixing_matrices": unmixing_matrices,
    }


def fobi_ica(x):
    N, C, T = x.shape

    unmixing_matrices = []
    for i in range(N):
        unmixing_matrices.append(torch.from_numpy(fobi(x[i, :, :])))

    sources = [unmixing_matrices[i] @ x[i, :, :] for i in range(N)]

    return {
        "sources": sources,
        "unmixing_matrices": unmixing_matrices,
    }
