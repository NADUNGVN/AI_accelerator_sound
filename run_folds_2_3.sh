#!/bin/bash
# Script to run Fold 2 and Fold 3 for both CrossEntropy and MSLE sequentially

echo "=== RUNNING CROSSENTROPY FOLD 2 ==="
python train.py --fold 2 --config configs/reproduce_crossentropy.json --exp_name crossentropy

echo "=== RUNNING CROSSENTROPY FOLD 3 ==="
python train.py --fold 3 --config configs/reproduce_crossentropy.json --exp_name crossentropy

echo "=== RUNNING MSLE FOLD 2 ==="
python train.py --fold 2 --config configs/reproduce_msle.json --exp_name msle

echo "=== RUNNING MSLE FOLD 3 ==="
python train.py --fold 3 --config configs/reproduce_msle.json --exp_name msle

echo "=== ALL RUNS FOR FOLDS 2 & 3 COMPLETED! ==="
