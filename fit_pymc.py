import sys
import os
from fetch_data import get_pymc_model
from time import time
import pymc as pm

start_year = int(sys.argv[1])
target_dir = sys.argv[2] + "/pymc"
seed = int(sys.argv[3])

os.makedirs(target_dir, exist_ok=True)

model = get_pymc_model(start_year=start_year)

start_time = time()

with model:
    hierarchical_trace = pm.sample(
        1000,
        tune=1000,
        return_inferencedata=True,
        compute_convergence_checks=False,
        random_seed=seed,
    )

runtime = time() - start_time

hierarchical_trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
