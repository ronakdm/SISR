import sys

sys.path.extend([".", ".."])
from src.utils import to_list_of_dicts

experiments = {
    "multitarget_simulated_cond_5": {
        "lr_unmix": [1e-4, 1e-3],
        "lr_model": [1e-6, 1e-5],
        "lam": [0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "seed": [i for i in range(20)],
        "density": ["huber"],
        "optim": ["sgd", "adam"],
        "cond_number": [5],
    },
    "multitarget_simulated_cond_7": {
        "lr_unmix": [1e-4, 1e-3],
        "lr_model": [1e-6, 1e-5],
        "lam": [0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "seed": [i for i in range(20)],
        "density": ["huber"],
        "optim": ["sgd", "adam"],
        "cond_number": [5],
    },
    "amari_distance_tensorial": {
        "method": ["jade", "fobi"],
        "abs_tol": [1e-10],
        "max_iter": [10**4],    # for MultiICA
        "dataset": ["laplace_mini"],
        "seed": [i for i in range(50)],
    },
    "amari_distance_tensorial_concatenation": {
        "method": ["jade", "fobi"],
        "abs_tol": [1e-10],
        "max_iter": [10**4],    # for MultiICA
        "dataset": ["laplace_mini"],
        "seed": [i for i in range(50)],
    },
}

if __name__ == "__main__":
    experiment = "multitarget_simulated_cond_5_incr"
    assert experiment in experiments
    lst = to_list_of_dicts(experiments[experiment])
    print(f"array needed for experiment '{experiment}': 0-{len(lst) - 1}")
