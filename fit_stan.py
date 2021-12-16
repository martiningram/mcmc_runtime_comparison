import sys
import os
from fetch_data import create_arrays
from time import time
import stan
import numpy as np
import arviz as az

start_year = int(sys.argv[1])
target_dir = sys.argv[2] + "/stan"

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

with open("./stan_model.stan", "r") as f:
    posterior = stan.build(program_code=f.read(), data=stan_data)

fit = posterior.sample(num_chains=4, num_samples=1000)

runtime = time() - start_time

arviz_version = az.from_pystan(fit)

az.to_netcdf(arviz_version, os.path.join(target_dir, f"samples_{start_year}.netcdf"))
print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
