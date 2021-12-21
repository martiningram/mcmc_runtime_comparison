# Modify as desired; will be created if it does not exist
target_dir="/media/martin/big_extra_space/pymc_vs_stan/rerun_20_12_2021/fits"

random_seed=2

# for start_year in 2020 2019 2015 2010 2000 1990 1980 1968; do
for start_year in 1980 1968; do
    echo "Fitting $start_year"
    echo "PyMC JAX GPU parallel" && python fit_pymc_jax.py $start_year gpu parallel "$target_dir" $random_seed
    echo "PyMC JAX GPU vectorized" && python fit_pymc_jax.py $start_year gpu vectorized "$target_dir" $random_seed
    echo "PyMC JAX CPU parallel" && python fit_pymc_jax.py $start_year cpu parallel "$target_dir" $random_seed
    # echo "Stan" && python fit_stan.py $start_year "$target_dir"
    echo "PyMC" && python fit_pymc.py $start_year "$target_dir" $random_seed
    echo "cmdstanpy" && python fit_cmdstanpy.py $start_year "$target_dir" $random_seed
done

# for start_year in 2020 2019 2015 2010 2000 1990 1980 1968; do
#     echo "Fitting $start_year"
#     echo "PyMC JAX CPU vectorized" && python fit_pymc_jax.py $start_year cpu vectorized
# done
