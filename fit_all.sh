# Modify as desired; will be created if it does not exist
target_dir="/media/martin/External Drive/projects/pymc_vs_stan/rerun_09_10_2022/fits"
# target_dir="./fits/"

random_seed=2

for start_year in 2020 2019 2015 2010 2000 1990 1980 1968; do
    echo "Fitting $start_year"
    echo "PyMC JAX GPU parallel" && python fit_pymc_numpyro.py $start_year gpu parallel "$target_dir" $random_seed
    echo "PyMC JAX GPU vectorized" && python fit_pymc_numpyro.py $start_year gpu vectorized "$target_dir" $random_seed
    echo "PyMC JAX CPU parallel" && python fit_pymc_numpyro.py $start_year cpu parallel "$target_dir" $random_seed
    echo "PyMC JAX CPU vectorized" && python fit_pymc_numpyro.py $start_year cpu vectorized "$target_dir" $random_seed
    echo "PyMC BlackJAX CPU" && python fit_pymc_blackjax.py $start_year cpu "$target_dir" $random_seed parallel
    echo "PyMC BlackJAX GPU" && python fit_pymc_blackjax.py $start_year gpu "$target_dir" $random_seed vectorized
    # echo "PyMC BlackJAX GPU Parallel env var" && XLA_FLAGS="--xla_force_host_platform_device_count=1" python fit_pymc_blackjax.py $start_year gpu "$target_dir" $random_seed parallel
    echo "PyMC" && python fit_pymc.py $start_year "$target_dir" $random_seed
    echo "cmdstanpy" && python fit_cmdstanpy.py $start_year "$target_dir" $random_seed
done

# for start_year in 2020 2019 2015 2010 2000 1990 1980 1968; do
#     echo "Fitting $start_year"
#     echo "PyMC JAX CPU vectorized" && python fit_pymc_numpyro.py $start_year cpu vectorized
# done
