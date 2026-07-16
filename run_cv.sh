#!/bin/bash
# Script to automate running remaining folds for UrbanSound8K 10-fold CV

START_FOLD=${1:-1}

for fold in $(seq $START_FOLD 10)
do
    echo "================================================================="
    echo "  STARTING TCAM1DCNN TRAINING & EVALUATION ON FOLD $fold / 10"
    echo "================================================================="
    
    # Run training and save logs to a file for easy monitoring
    python train.py --fold $fold --config configs/rtx3090_config.json > "train_fold_${fold}.log" 2>&1
    
    # Check if run succeeded
    if [ $? -eq 0 ]; then
        echo "Fold $fold completed successfully!"
    else
        echo "Fold $fold failed! Exiting script."
        exit 1
    fi
done

echo "All folds from $START_FOLD to 10 have completed successfully!"
