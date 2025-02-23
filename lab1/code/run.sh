#!/bin/bash
conda activate stat214
jupyter nbconvert --to notebook --execute --inplace data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace prediction.ipynb