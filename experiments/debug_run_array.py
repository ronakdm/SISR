import sys
import os
import pathlib
import torch

sys.path.extend([".", ".."])
from experiments.config import experiments
from experiments.run import run
from src.utils import to_list_of_dicts
import json
from datetime import datetime

job_id = 0
experiment = "ablin19"
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

with open(os.path.join(save_dir, f"{job_id:03d}.json"), "w") as f:
    json.dump(result, f, indent=2)