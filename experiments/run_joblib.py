import sys
import os
import numpy as np
import pathlib
import torch

sys.path.extend([".", ".."])
from experiments.config import experiments
from experiments.run import run
from src.utils import to_list_of_dicts
import json
from datetime import datetime

from joblib import Parallel, delayed

experiment = sys.argv[1]
path = pathlib.Path(__file__).parent.resolve()
save_dir = f"{path}/out/{experiment}"
os.makedirs(save_dir, exist_ok=True)
settings = to_list_of_dicts(experiments[experiment])
print(f"total settings for '{experiment}': {len(settings)}")

def worker(job_id):

    tic = datetime.now()
    result = run(experiment, settings[job_id])
    toc = datetime.now()
    elapsed = toc - tic
    print(f"Time elapsed for job {job_id:03d}: {elapsed}")

    if "model" in result:
        torch.save(result["model"].state_dict(), os.path.join(save_dir, f"model_{job_id:03d}.pt"))
        result.pop("model")

    with open(os.path.join(save_dir, f"{job_id:03d}.json"), "w") as f:
        json.dump(result, f, indent=2)

Parallel(n_jobs=-2)(delayed(worker)(i) for i in range(len(settings)))
# for job_id in range(len(settings)):
#     worker(job_id)