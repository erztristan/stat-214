#!/bin/bash

conda activate env_214
jupyter nbconvert --to notebook --execute --inplace lab3_3_lora_training.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3_3_lora_training_best_parameters.ipynb
jupyter nbconvert --to notebook --execute --inplace lab3_3_ridge_finetuned.ipynb
