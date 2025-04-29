#!/bin/bash

conda activate env_214
jupyter nbconvert --to notebook --execute --inplace lab3_2.ipynb
