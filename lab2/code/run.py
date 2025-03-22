import os
import shutil
import re
import numpy as np
import pandas as pd

# Split our images into three subfolders like in the code needed
source_dir = '../data/image_data'
labeled_dir = '../data/labeled_data'
unlabeled_dir = '../data/unlabeled_data'
unlabeled_pred_dir = '../data/unlabeled_data_prediction'
transfer_data = '../data/transfer_data'

os.makedirs(labeled_dir, exist_ok=True)
os.makedirs(unlabeled_dir, exist_ok=True)
os.makedirs(unlabeled_pred_dir, exist_ok=True)
os.makedirs(transfer_data, exist_ok=True)
os.makedirs("../data/data_with_features_and_pca", exist_ok=True)

labeled_files = {"O012791.npz", "O013257.npz", "O013490.npz"}

pattern = re.compile(r'^O0\d{5}\.npz$')

for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    if not os.path.isfile(file_path):
        continue

    if filename in labeled_files:
        destination = os.path.join(labeled_dir, filename)
    elif pattern.match(filename):
        destination = os.path.join(unlabeled_dir, filename)
    else:
        destination = os.path.join(unlabeled_pred_dir, filename)

    shutil.copy(file_path, destination)
    print(f"Copied {filename} to {destination}")


def add_noise(df, noise_ratio=0.1, random_state=42):
    np.random.seed(random_state)
    df_noisy = df.copy()
    for i in range(2, 9):
        variance = np.var(df.iloc[:, i])
        std_dev = np.sqrt(variance)

        mask = np.random.rand(len(df)) < noise_ratio
        noise = np.random.normal(0, std_dev, size=mask.sum())
        df_noisy.iloc[mask, i] += noise
    return df_noisy


# Loading the 3 expert label files
files = ['O013257.npz', 'O013490.npz', 'O012791.npz']
data = []

for i in range(len(files)):
    file_path = os.path.join("../data/labeled_data", files[i])
    npz_data = np.load(file_path)
    key = list(npz_data.files)[0]
    data.append(npz_data[key])
    print("loaded", i, "with shape", npz_data[key].shape)
    data_original = pd.DataFrame(data[i])
    df_noisy = add_noise(data_original)
    np.savez(f"../data/labeled_data/N{files[i][1:7]}.npz", data=df_noisy.to_numpy())