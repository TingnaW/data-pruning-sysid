#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "Running plot_predition.py...whbm...delay7"
python plot_prediction.py --dataset whbm --delay 7

echo "Running plot_atom.py...whbm...random10"
python plot_atom.py --dataset whbm --n_random 10

echo "Running plot_errorbar.py...random10"
python plot_errorbar.py --dataset whbm --n_random 10

echo "Running plot_box.py...random10"
python plot_box.py --dataset whbm --n_random 10

echo "Running plot_pca.py..."
python plot_pca.py --dataset whbm

echo "Running plot_batch.py...random10"
python plot_batch.py --dataset whbm --n_random 10

echo "Running plot_sample.py...random10"
python plot_sample.py --dataset whbm --n_random 10

echo "Running plot_density.py...random10"
python plot_density.py --dataset whbm

echo "All scripts completed."
