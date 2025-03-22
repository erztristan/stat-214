#!/bin/bash
conda activate stat214
python run.py
cd feature_engineering
python run_feature_engineering.py
cd ..
#jupyter nbconvert --to notebook --execute --inplace lab2.ipynb
#jupyter nbconvert --to notebook --execute --inplace modeling.ipynb
