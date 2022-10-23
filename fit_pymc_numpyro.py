import sys
import os
import pymc.sampling_jax
from fetch_data import get_pymc_model
from time import time
import pymc as pm

start_year = int(sys.argv[1])
platform = sys.argv[2]
chain_method = sys.argv[3]
base_dir = sys.argv[4]
seed = int(sys.argv[5])

assert platform in ["cpu", "gpu"]

if platform == "cpu":
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

target_dir = f"{base_dir}/pymc_numpyro_{platform}_{chain_method}"

os.makedirs(target_dir, exist_ok=True)

model = get_pymc_model(start_year=start_year)

start_time = time()

with model:
    hierarchical_trace = pymc.sampling_jax.sample_numpyro_nuts(
        chain_method=chain_method, random_seed=seed,
        idata_kwargs={'log_likelihood': False}
    )

runtime = time() - start_time

hierarchical_trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
