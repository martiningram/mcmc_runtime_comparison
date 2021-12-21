import sys
import os
from fetch_data import create_arrays
from time import time
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel

start_year = int(sys.argv[1])
target_dir = sys.argv[2] + "/cmdstanpy"
seed = int(sys.argv[3])

os.makedirs(target_dir, exist_ok=True)

arrays = create_arrays(start_year=start_year)

start_time = time()

winner_ids = arrays["winner_ids"]
loser_ids = arrays["loser_ids"]
player_encoder = arrays["player_encoder"]

stan_data = {
    "n_matches": len(winner_ids),
    "n_players": len(player_encoder.classes_),
    "winner_ids": winner_ids + 1,
    "loser_ids": loser_ids + 1,
}

model = CmdStanModel(stan_file="stan_model_optimised.stan")
model.compile()

fit = model.sample(data=stan_data, parallel_chains=4, seed=seed)

runtime = time() - start_time

arviz_version = az.from_cmdstanpy(posterior=fit)

az.to_netcdf(arviz_version, os.path.join(target_dir, f"samples_{start_year}.netcdf"))
print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
