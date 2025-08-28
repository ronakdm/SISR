import numpy as np
import torch
import torch.nn.functional as F
import scipy
from tqdm import tqdm
import time

from src.utils import to_dict_of_lists, check_input, check_labels, get_density_oracles, get_l2_norm_squared, evaluate, OptimizationError
   
def update_W(c, W, A_c, B_c, lr):
    C = len(W)
    if lr is None:
        # stochastic alternating minimization
        K = W @ (A_c @ W.T)
        b = -W @ B_c
    elif lr < 1e-16:
        return W
    else:
        # stochastic gradient update
        K = W @ ((A_c + torch.eye(C) / lr) @ W.T)
        b = W @ (W[c] / lr - B_c)
    e = np.zeros(shape=(C,))
    e[c] = 1
    v_e = torch.from_numpy(scipy.linalg.solve(0.5 * (K + K.T).numpy(), e, assume_a='pos')).float()
    v_b = torch.from_numpy(scipy.linalg.solve(0.5 * (K + K.T).numpy(), b, assume_a='pos')).float()

    r_cc = np.sqrt(v_e[c] + 0.25 * v_b[c] ** 2) + 0.5 * v_b[c]
    r_c = v_e / r_cc + v_b
    W[c] = r_c @ W 
    return W

@torch.no_grad()
def multi_ica(
        x, 
        density="huber", 
        lr_unmix=1e-3, 
        lr_model=1e-5, 
        tasks=None,
        models=None,
        lam=3e-5,
        labels=None,
        batch_size_trials=None, 
        batch_size_samples=None, 
        weight_decay=0.01,
        seed=0, 
        max_iter=5000,
        eval_iter=500,
        x_test=None,
        labels_test=None,
        mixing_mat=None,
        optim="adam",
        verbose=False,
        W_init=None
    ):

    # check types
    x = check_input(x)
    labels = check_labels(labels) if not (labels is None) else labels
    x_test = check_input(x_test) if not (x_test is None) else x_test
    labels_test = check_labels(labels_test) if not (labels_test is None) else labels_test

    N, C, T = x.shape
    if batch_size_trials is None:
        batch_size_trials = N
    if batch_size_samples is None:
        batch_size_samples = T
    G_func, update_u = get_density_oracles(density)

    # initialize table
    np.random.seed(seed)
    if W_init is None:
        W = 0.5 * torch.randn(C, C)
    else:
        W = torch.clone(W_init)

    # create model for supervision
    is_supervised = (not ((tasks is None) or tasks == [])) and lam > 0.0
    if is_supervised:
        assert not (lam is None), "balancing parameter 'lam' cannot be None for supervised task"
        assert not (labels is None), "labels 'labels' cannot be None for supervised task"
        assert not (models is None), "'models' cannot be None for supervised task"
        
        # adap parameters
        if optim == "adam":
            # default parameters
            betas = (0.9, 0.999)
            eps = 1e-8
            momentum = [[torch.zeros(param.shape) for param in model.parameters()] for model in models]
            variance = [[torch.zeros(param.shape) for param in model.parameters()] for model in models]
    else:
        models = None
        B = torch.zeros(*W.shape)

    # adaptivity parameters
    running_weight = 0.5
    factor = 2
    cond = 100 # new approximate condition number

    metrics = []
    elapsed = 0
    if is_supervised:
        ckpt_losses = [None for model in models]
        running_losses = [None for model in models]
    for k in tqdm(range(max_iter)):

        # log step
        if k % eval_iter == 0:
            metrics.append(
                evaluate(
                    k, W, x, x_test, 
                    G_func, models, mixing_mat, tasks, elapsed, 
                    lam, weight_decay, lr_unmix, lr_model,
                    labels_train=labels, labels_test=labels_test, verbose=verbose
                )
            )
            current_objective = metrics[-1]["train_error"]
            if k == 0:
                ckpt_objective = current_objective
            if current_objective >= 1.2 * ckpt_objective:
                lr_unmix /= factor
                print(f"Objective increased from {ckpt_objective} to {current_objective}! Setting W learning rate to {lr_unmix:0.7f}")
                ckpt_objective = current_objective


        tic = time.time()

        # compute coupling matrix
        i_idx = np.random.choice(N, size=batch_size_trials, replace=False)
        t_idx = np.random.choice(T, size=batch_size_samples, replace=False)
        x_batch = x[i_idx][:, :, t_idx]
        if is_supervised:
            y_batch = labels[i_idx]
        u_new = update_u(np.matmul(W, x_batch))
        
        ux = u_new.permute(dims=[1, 0, 2]).unsqueeze(2) * x_batch
        x_perm = x_batch.permute(dims=[0, 2, 1])
        A = []
        for c in range(C):
            A.append(torch.bmm(ux[c], x_perm))
        A = torch.stack(A).sum(dim=1) / (len(i_idx) * len(t_idx))
        A = 0.5 * (A + A.permute(dims=[0, 2, 1]))

        # apply supervision
        if is_supervised:
            B = torch.zeros(*W.shape)
            for model_id, model in enumerate(models):

                # update unmixing matrix and model parameters manually
                with torch.enable_grad():
                    W.requires_grad = True
                    loss = model(W[model_id] @ x[i_idx], y_batch[:, model_id])[0]
                    # loss = model(W @ x_batch, y_batch)[0]
                    grads = torch.autograd.grad(loss, inputs=[W] + list(model.parameters()))
                    W.requires_grad = False

                B += grads[0].T
                if optim == "sgd":
                    for grad, param in zip(grads[1:], list(model.parameters())):
                        # proximal step
                        param -= lr_model * grad
                        param /= (1 + weight_decay * lr_model)
                elif optim == "adam":
                    for grad, param, m_param, v_param in zip(grads[1:], list(model.parameters()), momentum[model_id], variance[model_id]):

                        # adjust gradient
                        grad = grad + weight_decay * param

                        # update momentum in place
                        m_param *= betas[0]
                        m_param += (1.0 - betas[0]) * grad

                        # update variance in place
                        v_param *= betas[1]
                        v_param += (1.0 - betas[1]) * grad ** 2

                        # update parameters
                        m = m_param / (1.0 - betas[0] ** (k + 1))
                        v = v_param / (1.0 - betas[1] ** (k + 1))
                        param -= lr_model * (m / (torch.sqrt(v) + eps))
                elif optim == "none":
                    pass
                else:
                    raise ValueError(f"Unrecognized optimizer '{optim}'! options: ['sgd', 'adam', 'none']")
                
                # running estimate of supervised loss
                current_loss = (0.5 * weight_decay * get_l2_norm_squared(model) + loss).item()
                if k == 0:
                    ckpt_losses[model_id] = current_loss
                    running_losses[model_id] = current_loss
                running_losses[model_id] = running_weight * current_loss + (1 - running_weight) * running_losses[model_id]

        # update decision variables coordinate-wise
        for c in range(C):
            try:
                if lam is None:
                    W = update_W(c, W, A[c], B[c], lr_unmix)
                else:
                    W = update_W(c, W, A[c], lam * B[c], lr_unmix)
                if torch.isnan(W).sum() > 0:
                    raise OptimizationError("Optimization failed. This is most likely caused by lr_model being set to high (should be 1e-4 or less in most cases).")
            except (np.linalg.LinAlgError, ValueError):
                lr_unmix /= factor
                lr_model /= factor
                print(f"Singular matrix! Setting umixing learning rate to {lr_unmix:0.7f} and model learning rate to {lr_model:0.7f}")
                U, S, V = torch.svd(W)
                W = (1 - 1 / cond) * W + (1 / cond) * S[0] * U @ V.T

        toc = time.time()
        elapsed += toc - tic
    
    metrics.append(
        evaluate(
            k, W, x, x_test,
            G_func, models, mixing_mat, tasks, elapsed, 
            lam, weight_decay, lr_unmix, lr_model,
            labels_train=labels, labels_test=labels_test, verbose=verbose
        )
    )
    return {
        "sources": np.stack([W @ x[i] for i in range(N)]), 
        "unmixing_matrix": W.detach().numpy(),
        "metrics": to_dict_of_lists(metrics),
        "models": models,
    }


