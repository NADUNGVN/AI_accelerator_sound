#!/bin/bash
# Script to automate running 10-fold Cross-Validation for paper reproduction

START_FOLD=${1:-1}

for fold in $(seq $START_FOLD 10)
do
    echo "================================================================="
    echo "  STARTING TCAM1DCNN MSLE REPRODUCTION ON FOLD $fold / 10"
    echo "================================================================="
    
    # Run training with MSLE config and save logs to a file
    python train.py --fold $fold --config configs/reproduce_msle.json --exp_name paper9_msle > "train_fold_${fold}.log" 2>&1
    
    # Check if run succeeded
    if [ $? -eq 0 ]; then
        echo "Fold $fold completed successfully!"
    else
        echo "Fold $fold failed! Exiting script."
        exit 1
    fi
done

echo "All folds from $START_FOLD to 10 have completed successfully!"
