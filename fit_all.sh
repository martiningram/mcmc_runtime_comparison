# Modify as desired; will be created if it does not exist
base_target_dir="/media/martin/External Drive/projects/pymc_vs_stan/multi_run/fits"
n_runs=10

for cur_run in `seq 1 $n_runs`; do

    echo "Running $cur_run"

    random_seed=$cur_run
    target_dir="$base_target_dir"/"$cur_run"

    for start_year in 2020 2019 2015 2010 2000 1990 1980 1968; do
	echo "Fitting $start_year"
	echo "PyMC JAX GPU parallel" && python fit_pymc_numpyro.py $start_year gpu parallel "$target_dir" $random_seed
	echo "PyMC JAX GPU vectorized" && python fit_pymc_numpyro.py $start_year gpu vectorized "$target_dir" $random_seed
	echo "PyMC JAX CPU parallel" && python fit_pymc_numpyro.py $start_year cpu parallel "$target_dir" $random_seed
	echo "PyMC JAX CPU vectorized" && python fit_pymc_numpyro.py $start_year cpu vectorized "$target_dir" $random_seed
	echo "PyMC BlackJAX CPU" && python fit_pymc_blackjax.py $start_year cpu "$target_dir" $random_seed parallel
	echo "PyMC BlackJAX GPU" && python fit_pymc_blackjax.py $start_year gpu "$target_dir" $random_seed vectorized
	echo "PyMC" && python fit_pymc.py $start_year "$target_dir" $random_seed
	echo "cmdstanpy" && python fit_cmdstanpy.py $start_year "$target_dir" $random_seed
    done

done
