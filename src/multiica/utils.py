import itertools
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class OptimizationError(Exception):
    def __init__(self, message):
        super().__init__(message)

def to_dict_of_lists(lst):
    return {key: [i[key] for i in lst] for key in lst[0]}

def to_list_of_dicts(d):
    for key in d:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
    return [dict(zip(d, x)) for x in itertools.product(*d.values())]

def check_input(x):
    assert isinstance(x, np.ndarray) or torch.is_tensor(x), "Inputs must be Numpy arrays or PyTorch tensors."
    x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
    assert len(x.shape) == 3, "Inputs must be three-dimensional [n_trials * n_channels * n_samples] arrays"
    return x

def check_mat(A):
    assert isinstance(A, np.ndarray) or torch.is_tensor(A), "Mixing matrix must be Numpy array or PyTorch tensor."
    A = torch.from_numpy(A).float() if isinstance(A, np.ndarray) else A
    assert len(A.shape) == 2, "Mixing matrix must be two-dimensional [n_channels * n_channels] array."
    return A

def check_labels(y):
    assert isinstance(y, np.ndarray) or torch.is_tensor(y), "Labels must be Numpy arrays or PyTorch tensors."
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    return y

def s_func(r):
    return torch.sum(torch.sum(r ** 2, dim=1) / torch.max(r ** 2, dim=1)[0] - 1)

def compute_amari_distance(W, mixing_mat):
    if mixing_mat is None:
        return None
    # source: https://github.com/pierreablin/mmica/blob/master/mmica/_utils.py
    P = W @ mixing_mat
    return ((s_func(torch.abs(P)) + s_func(torch.abs(P.T))) / (2 * P.shape[0])).item()

def get_l2_norm_squared(model):
    total_norm = torch.tensor(0.)
    for param in model.parameters():
        total_norm += torch.linalg.norm(param) ** 2
    return total_norm

def update_u_infomax(y):
    return torch.nn.functional.tanh(y) / y

def update_u_student(y):
    return 1. / (1. + y ** 2)

def update_u_huber(y):
    ret = torch.ones(*y.shape)
    ret[torch.abs(y) >= 1] = 1. / torch.abs(y[torch.abs(y) >= 1.])
    return ret

def update_u_laplace(y, eps=1e-8):
    ret = torch.zeros(*y.shape)
    ret[torch.abs(y) > eps] = 1. / torch.abs(y[torch.abs(y) > eps])
    return ret

def get_density_oracles(density):
    if density=="infomax":
        return (lambda y: torch.log(torch.cosh(y)), update_u_infomax)
    elif density=="student":
        return (lambda y: 0.5 * torch.log(1 + y ** 2), update_u_student)
    elif density=="huber":
        return (lambda y: 0.5 * y ** 2 * (torch.abs(y) < 1) + (torch.abs(y) - 0.5) * (torch.abs(y) >= 1), update_u_huber)
    elif density=="laplace":
        return (lambda y: torch.abs(y), update_u_laplace)
    

@torch.no_grad()
def compute_loss(W, x, G_func, models, lam, labels, tasks, weight_decay):
    if x is None:
        return None
    N, C, T = x.shape

    # unsupervised term
    unsup_term = (G_func(torch.stack([W @ x_i for x_i in x])).sum() / (N * T) - torch.log(torch.abs(torch.linalg.det(W)))).item()

    # supervised term
    accuracy = []
    sup_term = 0
    if lam > 0.0:
        for model_id, model in enumerate(models):
            task = tasks[model_id]
            #loss, y_pred = model(W[model_id] @ x, labels[:, model_id])
            x_src = torch.einsum('c,bct->bt', W[model_id], x)  # (B, T)
            x_src = x_src.unsqueeze(1)                                # (B, 1, T)
            loss, y_pred = model(x_src, labels[:, model_id])
            if task == "classification":
                accuracy.append((torch.argmax(y_pred, dim=1) == labels[:, model_id]).float().mean().item())
            elif task == "regression":
                #accuracy.append(r2_score(labels[:, model_id].numpy(), y_pred.numpy(), multioutput='variance_weighted'))
                accuracy.append(mean_squared_error(labels[:, model_id].numpy(), y_pred.numpy()))
    
            sup_term += lam * (loss.item() + 0.5 * weight_decay * get_l2_norm_squared(model).item())

    return unsup_term, sup_term, accuracy

def evaluate(k, W, x_train, x_test, G_func, models, mixing_mat, tasks, elapsed, lam, weight_decay, lr_unmix, lr_model, labels_train=None, labels_test=None, verbose=True):
    train_unsup, train_sup, train_accs = compute_loss(W, x_train, G_func, models, lam, labels_train, tasks, weight_decay)
    out = {
        "train_error": train_unsup + train_sup,
        "train_unsup": train_unsup,
        "train_sup": train_sup,
        "amari_distance": compute_amari_distance(W, mixing_mat),
        "lr_unmix": lr_unmix,
        "lr_model": lr_model,
        "elapsed": elapsed,
        "iterations": k,
        "W": W.detach().clone(),
    }
    for t, acc in enumerate(train_accs):
        out[f"train_accuracy_task_{t}"] = acc
    if not (x_test is None):
        test_unsup, test_sup, test_accs = compute_loss(W, x_test, G_func, models, lam, labels_test, tasks, weight_decay)
        out["test_error"] = test_unsup + test_sup
        out["test_unsup"] = test_unsup
        out["test_sup"] = test_sup
        for t, acc in enumerate(test_accs):
            out[f"test_accuracy_task_{t}"] = acc
    if verbose:
        print(out)
    return out
