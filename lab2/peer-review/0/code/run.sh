#!/bin/bash

conda activate stat214
jupyter nbconvert --to notebook --execute --inplace EDA.ipynb

<<<<<<< HEAD
jupyter nbconvert --to notebook --execute --inplace model.ipynb
=======
jupyter nbconvert --to notebook --execute --inplace model_final.ipynb
>>>>>>> e0af12cd7874fdf44845931676ca99571c1cf611



