#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "Running plot_predition.py...dsed...delay3"
python plot_prediction.py --dataset dsed --delay 3

echo "Running plot_atom.py...dsed...random20"
python plot_atom.py --dataset dsed --n_random 20

echo "Running plot_errorbar.py...random20"
python plot_errorbar.py --dataset dsed --n_random 20

echo "Running plot_box.py...random20"
python plot_box.py --dataset dsed --n_random 20

echo "Running plot_pca.py..."
python plot_pca.py --dataset dsed

echo "Running plot_batch.py...random20"
python plot_batch.py --dataset dsed --n_random 20

echo "Running plot_sample.py...random20"
python plot_sample.py --dataset dsed --n_random 20

echo "All scripts completed."
