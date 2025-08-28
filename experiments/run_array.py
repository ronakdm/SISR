import sys
import os
import pathlib
import torch
import numpy as np

sys.path.extend([".", ".."])
from experiments.config import experiments
from experiments.run import run
from src.utils import to_list_of_dicts
import json
from datetime import datetime

job_id = int(sys.argv[1])
experiment = sys.argv[2]
path = pathlib.Path(__file__).parent.resolve()
save_dir = f"{path}/out/{experiment}"

os.makedirs(save_dir, exist_ok=True)

settings = to_list_of_dicts(experiments[experiment])

tic = datetime.now()
result = run(experiment, settings[job_id])
toc = datetime.now()
elapsed = toc - tic
print(f"Time elapsed: {elapsed}")

if "model" in result:
    torch.save(result["model"].state_dict(), os.path.join(save_dir, f"model_{job_id:03d}.pt"))
    result.pop("model")

if "unmixing_matrix" in result:
    np.save(os.path.join(save_dir, f"unmixing_matrix_{job_id:03d}.npy"), result["unmixing_matrix"])
    result.pop("unmixing_matrix")

with open(os.path.join(save_dir, f"{job_id:03d}.json"), "w") as f:
    json.dump(result, f, indent=2)