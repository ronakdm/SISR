# SISR
Stochastic algorithms for multi-trial, supervised independent component analysis. SISR (Supervised Independent Source Recovery) is the name of the core algorithm, implemented in the `src/sisr_algorithm.py` module.


## Dependencies

We recommend a hardware environment that has at least 32GB of CPU RAM for ease of use. GPUs are not necessary, although for larger-scale models in the supervised term, one with at least 12GB of GPU RAM may be used.
The code runs in Python 3 with the standard Python scientific stack, along with PyTorch and packages built on top of it. What you need to install depends on what you're doing with the repo.

**To use `sisr()` directly and run the `experiments/` scripts** (`run.py`, `run_array.py`, `run_joblib.py`):
```
pip install numpy scipy pandas scikit-learn tqdm joblib
```

**To also regenerate the figures in `notebooks/`**, which additionally need plotting and CSP-baseline packages:
```
pip install numpy scipy pandas scikit-learn tqdm joblib matplotlib seaborn ipykernel mne pyriemann
```

Both cases also need PyTorch and TorchAudio — follow the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Note that a GPU is not necessary to run the method on most examples, so simply installing PyTorch and TorchAudio for CPU with `pip install torch torchaudio` will work for most users.

**`notebooks/figure_eeg_motor_imagery_benchmark.ipynb` needs one more thing, in a separate environment.** Fetching the raw BNCI2014_001 EEG data requires `moabb`, which needs `numpy>=2` — incompatible with the `numpy<2` this project's main environment otherwise uses. Rather than pulling `moabb` into the main environment, that notebook shells out to a second, dedicated Python interpreter for just the data-fetch step (see `MOAB_PYTHON` near the top of its EEG-loading cell). To set that up:
```
conda create -n moab python=3.11
conda activate moab
pip install numpy torch moabb
```
then point `MOAB_PYTHON` in the notebook at that environment's `python3`. This only runs once per subject — results are cached under `notebooks/output/results/.../eeg_per_subject_data/`, so subsequent runs (or re-runs after clearing other caches) won't need this second environment again. Note that `notebooks/output/` is gitignored and not shipped in the repo (see "Reproducing Experiments" below), so a fresh clone *will* need this step at least once.

## Overview

Below is an outline of the code repository.
| Directory      | Description |
| ----------- | ----------- |
| `src`   | Source code. The full SISR algorithm is implemented via the `sisr` function in `sisr_algorithm.py`, used by both the `experiments/` Slurm pipeline below and the notebooks in `notebooks/`, along with the models (`models.py`), data loaders (`data.py`), and utilities (`utils.py`) it depends on. |
| `notebooks`   | Visualizations such as training curves for the simulated datasets (`figure_baselines.ipynb`, `figure_nonconvex.ipynb`), plus the arxiv manuscript's supervision-effect and EEG motor-imagery benchmark figures (`figure_effect_of_supervision.ipynb`, `figure_eeg_motor_imagery_benchmark.ipynb`). |
| `experiments`   | Code for running experiments (whose individual logic is written in `run.py`) either locally (`run_joblib.py`) or on a Slurm cluster (`run_array.py`, `run_array.sbatch`). |
| `data`   |  Empty placeholder directory. `load_dataset()` requires its `data_path` argument to exist, but every dataset it currently supports (`laplace_mini`, `laplace_tall`, `laplace_wide`, `multioutput_simulated`) is generated in-memory, not read from a file. |

## Quickstart

We provide access to the simulated data used for the manuscript, which can be loaded as shown (set `data_path` to the `data` directory of this repository).

```
from src.data import load_dataset

dataset = "multioutput_simulated"
n_tasks = 3

x_train, y_train, x_test, y_test, metadata = load_dataset(
    dataset=dataset, 
    data_path="data"
)
N, C, T = x_train.shape
M = y_train.shape[1]
mixing_mat = metadata["mixing_mat"]
tasks = ["regression"] * n_tasks
```
For a minimal example with default values, you may run the following. Note that for any supervised task, one must specify the supervised models (which are of type `nn.Module` from PyTorch); this example uses `SpectrogramSharedBinPresenceMLP`, the shared per-time-bin presence classifier used in the `notebooks/` figures. Unlike a plain scalar-per-trial regression model, it predicts *presence* (1/0) independently at each of its own output time bins, so `labels` must have shape `(N, n_tasks, n_time_bins)` rather than `(N, n_tasks)` — below, `n_time_bins` is read off the model itself, and the presence labels are only illustrative (thresholding `multioutput_simulated`'s scalar per-trial label to fake a presence signal); substitute your own per-time-bin event/presence labels for real usage. For fully unsupervised tasks (`lam=0.0`) you may simply use `models=None` and `tasks=None`.
```
import numpy as np
from src.sisr_algorithm import sisr
from src.models import SpectrogramSharedBinPresenceMLP

batch_size_trials=64 
batch_size_samples=64 

models = [
    SpectrogramSharedBinPresenceMLP(
        n_channels=1, n_fft=64, window_length=64, hop_length=16,
        n_timepoints=T, bottleneck=True, freq_cutoff=100, sample_rate=1000,
    )
    for _ in range(n_tasks)
]
n_time_bins = models[0].n_time

# Illustrative presence labels: 1 in every time bin of trials whose scalar
# label for that task is above its own median, 0 otherwise. Replace with
# real per-time-bin presence/event labels for your own data.
labels = np.stack([
    (y_train[:, m] > np.median(y_train[:, m])).astype(np.float32)[:, None].repeat(n_time_bins, axis=1)
    for m in range(n_tasks)
], axis=1)

output = sisr(
    x_train, 
    labels=labels,
    batch_size_trials=batch_size_trials, 
    batch_size_samples=batch_size_samples, 
    models=models,
    tasks=tasks,
    mixing_mat=mixing_mat, # used for measuring Amari distance
)
```
For greater control over the hyperparameters, see the more detailed example below.
```
import numpy as np
from src.sisr_algorithm import sisr
from src.models import SpectrogramSharedBinPresenceMLP

W_init = None       # initial value for unmixing matrix
seed = 0            # seed for algorithmic randomness
lam = 3e-05         # balancing parameter
lr_unmix = 1e-3     # learning rate for the unmixing matrix
lr_model = 1e-4     # learning rate for the models used in supervised learning
weight_decay = 0.01 # l2 regularization parameter
density = "huber"   # density used to define the negative log likelihood objective
optim = "adam"      # optimizer used for supervised model, either "sgd" or "adam"

models = [
    SpectrogramSharedBinPresenceMLP(
        n_channels=1, n_fft=64, window_length=64, hop_length=16,
        n_timepoints=T, bottleneck=True, freq_cutoff=100, sample_rate=1000,
    )
    for _ in range(n_tasks)
]
n_time_bins = models[0].n_time

# See the minimal example above for what these presence labels represent.
labels = np.stack([
    (y_train[:, m] > np.median(y_train[:, m])).astype(np.float32)[:, None].repeat(n_time_bins, axis=1)
    for m in range(n_tasks)
], axis=1)
labels_test = np.stack([
    (y_test[:, m] > np.median(y_test[:, m])).astype(np.float32)[:, None].repeat(n_time_bins, axis=1)
    for m in range(n_tasks)
], axis=1)

batch_size_trials = 64 
batch_size_samples = 64
max_iter = 5000
eval_iter = 500

output = sisr(
    x_train, 
    density=density, 
    lr_unmix=lr_unmix, 
    lr_model=lr_model, 
    tasks=tasks,
    models=models,
    lam=lam,
    labels=labels,
    batch_size_trials=batch_size_trials, 
    batch_size_samples=batch_size_samples, 
    weight_decay=weight_decay,
    seed=seed, 
    max_iter=max_iter,
    eval_iter=eval_iter,
    x_test=x_test,
    labels_test=labels_test,
    mixing_mat=mixing_mat,
    optim=optim,
    W_init=W_init
)
```
Here, `output` is a dictionary containing the independence sources `sources`, the supervised model `model` (as a PyTorch module), the unmixing matrix `W` via the key `unmixing_matrix`, and the training metrics in `metrics`.
```
import pandas as pd

W = output['unmixing_matrix']
result = pd.DataFrame(output['metrics'])
result
```

## Reproducing Experiments

This section explains how to regenerate everything the `notebooks/figure_*.ipynb` files plot — both the JSON results the "cheap" notebooks read (`figure_baselines.ipynb`, `figure_nonconvex.ipynb`) and the self-contained fits the two arxiv_v2 notebooks run themselves (`figure_effect_of_supervision.ipynb`, `figure_eeg_motor_imagery_benchmark.ipynb`).

**None of this data is included in the repo.** `experiments/out/` and `notebooks/output/` are both gitignored — a fresh clone has no cached results at all, and every notebook/script below regenerates its own cache from scratch (subsequent runs are fast no-ops as long as that cache still exists).

The time estimates below were measured by timing `sisr()` directly with each pipeline's actual model/data configuration (30 iterations each, single-threaded per fit) on a 12-core CPU machine, then extrapolating to the full iteration/fit counts — they scale roughly with `(total fits) / (CPU cores available)`, so expect proportionally more or less time on different hardware. Treat them as order-of-magnitude planning numbers, not guarantees.

### `experiments/out/` (backs `figure_baselines.ipynb`, `figure_nonconvex.ipynb`)

The experiments used in the paper are defined and run via files in the `experiments/` directory:
- `config.py`: This file simply contains key names for experiments, and any possible parameters that specify them. These are specified by lists, after which every element in the Cartesian product of these lists is a possible experiment setting.
- `run.py`: This file uses the same key names that appear in `config.py`, but specifies the logic needed to convert the hyperparameters into a result. For example, the experiment key `multitarget_simulated_cond_5` corresponds to the experiment used to compute the Amari distance when the mixing matrix is ill-conditioned (Figure 4).
- `run_joblib.py` / `run_array.py` + `run_array.sbatch`: These run every setting for a given experiment key and write one JSON file per setting to `experiments/out/<experiment_key>/`. `run_joblib.py` runs the full Cartesian product locally in parallel (see `run_local.sh` for an example); `run_array.py` runs a single setting (`SLURM_ARRAY_TASK_ID`) and is meant to be launched once per array index by `run_array.sbatch` on a Slurm cluster.

To reproduce an existing experiment locally, run:
```
python experiments/run_joblib.py <experiment_key>
```
To reproduce it on a Slurm cluster instead, specify the key name and the email of the user in `run_array.sbatch` (matching the `--array` range to the number of settings for that key, printed by running `config.py`), and run:
```
sbatch experiments/run_array.sbatch
```
For custom experiments, follow the flow above by:
1. Adding a dictionary to `experiments` in `config.py`.
2. Adding the logic of that experiment (with the same key name) to `run.py`.
3. Running `python experiments/run_joblib.py <experiment_key>` locally, or editing `run_array.sbatch` with the number of array jobs and running it as shown above.

Approximate time to regenerate each key from scratch (12-core CPU machine, `run_joblib.py`):
| Experiment key | Settings | Iterations/setting | Estimated time |
| --- | --- | --- | --- |
| `multitarget_simulated_cond_5` | 480 | 10,000 | ~14 hours |
| `multitarget_simulated_cond_7` | 480 | 10,000 | ~14 hours |
| `amari_distance_tensorial` | 100 | 10,000 (unsupervised, much cheaper per-iteration) | a few minutes |
| `amari_distance_tensorial_concatenation` | 100 | 10,000 (unsupervised) | a few minutes |

### `notebooks/output/` (self-generated by `figure_effect_of_supervision.ipynb` and `figure_eeg_motor_imagery_benchmark.ipynb`)

These two notebooks generate/fetch their own input data and fit everything themselves — there's no separate script to run, just execute the notebook top to bottom. Every `fit_*`/`load_*` helper checks its own cache first (`retrain=False` by default), so a partial or interrupted run only redoes what's missing next time.

- **`figure_effect_of_supervision.ipynb`**: generates its simulated "neural-like" dataset in-notebook (no download needed), then runs at least ~77 supervised `sisr()` fits at 10,000 iterations each (a fixed 3-combo + 73-combo grid sweep), plus up to ~216 more in the worst case from an adaptive 3-stage warm-start chain that only retries combos that diverged. **Estimated: roughly 4–14 hours** on a 12-core machine, depending on how many combos need warm-start retries.
- **`figure_eeg_motor_imagery_benchmark.ipynb`**: fetches real BNCI2014_001 EEG data via MOABB (see the separate `moab` environment setup under Dependencies above; this step itself is comparatively quick), then runs a fixed 45 (subject, seed) combos at 30,000 iterations each. **Estimated: roughly 9 hours** on a 12-core machine for the fits alone.

Both notebooks parallelize their fits across all available CPU cores via `joblib.Parallel` (`n_jobs=-1`), so wall time drops roughly proportionally on a machine with more cores (and rises on one with fewer).
