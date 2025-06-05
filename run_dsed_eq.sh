#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "Running plot_pca.py..."
python plot_pca.py --dataset dsed-eq

echo "Running plot_errorbar.py..."
python plot_errorbar.py --dataset dsed-eq

echo "Running plot_atom.py..."
python plot_atom.py --dataset dsed-eq

echo "Running plot_batch.py..."
python plot_batch.py --dataset dsed-eq

echo "Running plot_box.py..."
python plot_box.py --dataset dsed-eq

echo "All scripts completed."