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
then point `MOAB_PYTHON` in the notebook at that environment's `python3`. This only runs once per subject — results are cached under `notebooks/output/results/.../eeg_per_subject_data/`, so if those caches already exist (as shipped in the repo), you likely won't need this second environment at all.

## Overview

Below is an outline of the code repository.
| Directory      | Description |
| ----------- | ----------- |
| `src`   | Source code. The full SISR algorithm is implemented via the `sisr` function in `sisr_algorithm.py`, used by both the `experiments/` Slurm pipeline below and the notebooks in `notebooks/`, along with the models (`models.py`), data loaders (`data.py`), and utilities (`utils.py`) it depends on. |
| `notebooks`   | Visualizations such as training curves for the simulated datasets (`figure_baselines.ipynb`, `figure_nonconvex.ipynb`), plus the arxiv manuscript's supervision-effect and EEG motor-imagery benchmark figures (`figure_effect_of_supervision.ipynb`, `figure_eeg_motor_imagery_benchmark.ipynb`). |
| `experiments`   | Code for running experiments (whose individual logic is written in `run.py`) either locally (`run_joblib.py`) or on a Slurm cluster (`run_array.py`, `run_array.sbatch`). |
| `data`   |  Includes simulated data used by the `experiments/` pipeline. |

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
For a minimal example with default values, you may run the following. Note that for any supervised task, one must specify the supervised models (which are of type `nn.Module` from PyTorch). For fully unsupervised tasks (`lam=0.0`) you may simply use `models=None` and `tasks=None`.
```
from src.sisr_algorithm import sisr
from src.models import SpectrogramMLP

batch_size_trials=64 
batch_size_samples=64 

models = [SpectrogramMLP("regression", 1, 26, 41, None) for _ in range(n_tasks)]

output = sisr(
    x_train, 
    labels=y_train,
    batch_size_trials=batch_size_trials, 
    batch_size_samples=batch_size_samples, 
    models=models,
    tasks=tasks,
    mixing_mat=mixing_mat, # used for measuring Amari distance
)
```
For greater control over the hyperparameters, see the more detailed example below.
```
from src.sisr_algorithm import sisr
from src.models import SpectrogramMLP

W_init = None       # initial value for unmixing matrix
seed = 0            # seed for algorithmic randomness
lam = 3e-05         # balancing parameter
lr_unmix = 1e-3     # learning rate for the unmixing matrix
lr_model = 1e-4     # learning rate for the models used in supervised learning
weight_decay = 0.01 # l2 regularization parameter
density = "huber"   # density used to define the negative log likelihood objective
optim = "adam"      # optimizer used for supervised model, either "sgd" or "adam"

models = [SpectrogramMLP("regression", 1, 26, 41, None) for _ in range(n_tasks)]

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
The output is saved in JSON format in `experiments/out`. This is the exact output that is used to generate the figures in the `notebook/figure_*.ipynb` files.
