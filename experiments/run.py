import time

import numpy as np
import torch

from src.sisr_algorithm import sisr
from src.utils import OptimizationError, compute_amari_distance
from src.sisr_tensor import fobi_ica, jade_ica, fobi, jade, process_multi_trial
from src.data import load_dataset
from src.models import CNN, MeanNet, SpectrogramMLP

def run(experiment, setting):
    if "multitarget_simulated" in experiment:

        dataset = "multioutput_simulated"
        n_tasks = 3

        x_train, y_train, x_test, y_test, metadata = load_dataset(
            dataset=dataset, 
            data_path="data",
            params={
                "noise_scale": 0.3, 
                "cond_number": setting["cond_number"]
            }
        )
        N, C, T = x_train.shape
        M = y_train.shape[1]
        mixing_mat = metadata["mixing_mat"]
        tasks = ["regression"] * n_tasks

        W_init = None
        
        seed = setting["seed"]
        lam = setting["lam"]
        lr_unmix = setting["lr_unmix"]
        lr_model = setting["lr_model"]
        weight_decay = 0.01
        density = "huber"
        optim = setting["optim"]

        models = [
            SpectrogramMLP("regression", 1, 26, 41, None),
            SpectrogramMLP("regression", 1, 26, 41, None),
            SpectrogramMLP("regression", 1, 26, 41, None),
            SpectrogramMLP("regression", 1, 26, 41, None),
            SpectrogramMLP("regression", 1, 26, 41, None),
        ]
        models = models[:n_tasks]

        max_iter = 10000
        eval_iter = 400
        batch_size_trials = 128
        batch_size_samples = 128
        verbose = False

        test_func = sisr

        try:
            output = test_func(
                x_train, 
                density=density, 
                lr_unmix=lr_unmix, 
                lr_model=lr_model, 
                tasks=tasks,
                models=models,
                lam=lam,
                labels=y_train,
                batch_size_trials=batch_size_trials, 
                batch_size_samples=batch_size_samples, 
                weight_decay=weight_decay,
                seed=seed, 
                max_iter=max_iter,
                eval_iter=eval_iter,
                x_test=x_test,
                labels_test=y_test,
                mixing_mat=mixing_mat,
                optim=optim,
                verbose=verbose,
                W_init=W_init
            )
            out = {
                'metrics': output["metrics"],
                'setting': setting,
                'unmixing_matrix': output['unmixing_matrix']
            }
            return out
        except OptimizationError:
            out = {
                'metrics': None,
                'setting': setting,
                'unmixing_matrix': None
            }
            return out
    elif experiment == "reach_multitarget_icassp":
        dataset = setting["dataset"]
        task = "multioutput"
        x_train, y_train, x_test, y_test, metadata = load_dataset(
            dataset=dataset, 
            task=task,
        )
        tasks = ["classification", "classification"]
        N, C, T = x_train.shape

        y_train = y_train[:, 2:4]
        y_test = y_test[:, 2:4]

        W_init = None
        
        
        seed = setting["seed"]
        lam = setting["lam"]
        lr_unmix = setting["lr_unmix"]
        lr_model = setting["lr_model"]
        weight_decay = 0.01
        density = "huber"
        optim = setting["optim"]

        models = [
            SpectrogramMLP(tasks[0], 1, 26, 81, n_classes=len(np.unique(y_train[:, 0])), n_layers=setting["n_layers"]),
            SpectrogramMLP(tasks[1], 1, 26, 81, n_classes=len(np.unique(y_train[:, 1])), n_layers=setting["n_layers"]),
        ]

        max_iter = 5000
        eval_iter = 200
        batch_size_trials = 32
        batch_size_samples = 64
        verbose = False

        try:
            output = sisr(
                x_train,
                density=density,
                lr_unmix=lr_unmix,
                lr_model=lr_model,
                tasks=tasks,
                models=models,
                lam=lam,
                labels=y_train,
                batch_size_trials=batch_size_trials,
                batch_size_samples=batch_size_samples,
                weight_decay=weight_decay,
                seed=seed,
                max_iter=max_iter,
                eval_iter=eval_iter,
                x_test=x_test,
                labels_test=y_test,
                mixing_mat=None,
                optim=optim,
                verbose=verbose,
                W_init=W_init
            )
            out = {
                'metrics': output["metrics"],
                'setting': setting,
                'unmixing_matrix': output['unmixing_matrix']
            }
            return out
        except OptimizationError:
            out = {
                'metrics': None,
                'setting': setting,
                'unmixing_matrix': None
            }
            return out
    elif experiment == "amari_distance_tensorial":
        method = setting["method"] 
        seed = setting["seed"]
        abs_tol = setting["abs_tol"]
        max_iter = setting["max_iter"]
        dataset = setting["dataset"]

        x_train, x_test, metadata = load_dataset(dataset=dataset, data_path="data/", seed=seed,  params={"noise_scale": 0.3})
        mixing_mat = torch.from_numpy(metadata["mixing_mat"])

        tic = time.time()
        if method == "jade":
            result_tensorial = jade_ica(x_train, max_iter=max_iter, abs_tol=abs_tol)
        elif method == "fobi":
            result_tensorial = fobi_ica(x_train)
        toc = time.time()
        elapsed = toc - tic

        amari_distances_tensorial  = []
        for unmixing_mat in result_tensorial["unmixing_matrices"]:
            amari_distances_tensorial.append(compute_amari_distance(unmixing_mat,mixing_mat))
        
        result_sisr = sisr(x_train,
                                    x_test=x_test,
                                    max_iter=max_iter,
                                    batch_size_trials=10,
                                    batch_size_samples=64,
                                    lr_unmix=0.1,
                                    mixing_mat=mixing_mat.float(),
                                    density="huber",
                                    seed=seed,
                                    eval_iter=5,
                                    lam=0.0,
                                    tasks=None)
        unmixing_mat_sisr = torch.from_numpy(result_sisr["unmixing_matrix"]).double()
        amari_distance_sisr = compute_amari_distance(unmixing_mat_sisr, mixing_mat)

        out = {
            "elapsed_tensorial": elapsed,
            "amari_distances_tensorial": amari_distances_tensorial,
            "amari_distance_multi_ica": amari_distance_sisr,
            "metrics": result_sisr["metrics"]
        }
        return out

    elif experiment == "amari_distance_tensorial_concatenation":
        method = setting["method"] 
        seed = setting["seed"]
        abs_tol = setting["abs_tol"]
        shift_factor = setting["shift_factor"]
        max_iter = setting["max_iter"]
        dataset = setting["dataset"]

        x_train, x_test, metadata = load_dataset(dataset=dataset, data_path="data/", seed=seed, params={"noise_scale": 0.3})
        mixing_mat = torch.from_numpy(metadata["mixing_mat"])

        N, C, T = x_train.shape

        shifts = shift_factor * x_train[0].std(axis=-1)[0] * np.random.uniform(low=-1, high=1, size=(N, 1, 1))
        x_train_shifted = x_train + shifts

        x_train_concat = np.moveaxis(x_train_shifted, 1, 0).reshape(C, N * T)
        x_train_concat_demean = process_multi_trial(x_train_concat, T)

        tic = time.time()
        if method == "jade":
            unmixing_mat_tensorial = jade(x_train_concat_demean, max_iter=max_iter, abs_tol=abs_tol)
        elif method == "fobi":
            unmixing_mat_tensorial = fobi(x_train_concat_demean)
        toc = time.time()
        elapsed = toc - tic

        amari_distances_tensorial  = [compute_amari_distance(torch.from_numpy(unmixing_mat_tensorial), mixing_mat)]
        result_sisr = sisr(x_train_shifted,
                                    x_test=x_test,
                                    max_iter=max_iter,
                                    batch_size_trials=10,
                                    batch_size_samples=64,
                                    lr_unmix=0.25,
                                    mixing_mat=mixing_mat.float(),
                                    density="huber",
                                    seed=seed,
                                    lam=0.0,
                                    eval_iter=5,
                                    tasks=None)
        unmixing_mat_sisr = torch.from_numpy(result_sisr["unmixing_matrix"]).double()
        amari_distance_sisr = compute_amari_distance(unmixing_mat_sisr, mixing_mat)

        out = {
            "elapsed_tensorial": elapsed,
            "amari_distances_tensorial": amari_distances_tensorial,
            "amari_distance_multi_ica": amari_distance_sisr,
            "metrics": result_sisr["metrics"]
        }
        return out
    else:
        raise NotImplementedError(f"Unrecognized experiment '{experiment}'!")
