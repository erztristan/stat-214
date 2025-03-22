import numpy as np
import os
import pandas as pd
import glob
import run_autoencoder
import get_embedding
import yaml

def load_npz_files(path, remove_label=False):
    """Load NPZ files from the specified path and optionally remove the label column. Returns image IDs, full data arrays, patch data arrays, and file paths."""
    image_ids, data_full, data_patch = [], [], []
    filepaths = glob.glob(path)
    for fp in filepaths:
        image_ids.append(os.path.basename(fp))
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        data_full.append(data)
        data_patch.append(data[:, :-1] if remove_label and data.shape[1] == 11 else data)
    return image_ids, data_full, data_patch, filepaths

def compute_global_grid(data_list):
    """Compute the global grid parameters (min, max, height, width) from a list of data arrays. Returns global minimum y, maximum y, minimum x, maximum x, grid height, and grid width."""
    all_y = np.concatenate([img[:, 0] for img in data_list]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in data_list]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height, width = int(global_maxy - global_miny + 1), int(global_maxx - global_minx + 1)
    return global_miny, global_maxy, global_minx, global_maxx, height, width

def reshape_images(data_list, global_miny, global_minx, height, width):
    """Reshape each image array onto a common grid based on global coordinates. Returns a numpy array of reshaped images."""
    nchannels = data_list[0].shape[1] - 2
    images = []
    for img in data_list:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    return np.array(images)

def normalize_images(images, keepdims=True):
    """Normalize images per channel using global means and standard deviations. Returns the normalized images."""
    if keepdims:
        means = np.mean(images, axis=(0, 2, 3), keepdims=True)
        stds = np.std(images, axis=(0, 2, 3), keepdims=True)
    else:
        means = np.mean(images, axis=(0, 2, 3))[:, None, None]
        stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    return (images - means) / stds

def compute_surrounding_features(patch, exclude_center=False):
    """Compute the minimum, mean, and maximum for each channel of a patch; optionally excluding the center pixel. Returns a numpy array of computed features."""
    nchannels, patch_size, _ = patch.shape
    features = []
    center = patch_size // 2  # Works for odd patch sizes.
    for c in range(nchannels):
        channel_data = patch[c]
        if exclude_center:
            flat = channel_data.flatten()
            center_index = center * patch_size + center
            values = np.delete(flat, center_index)
        else:
            values = channel_data.flatten()
        features.append(values.min())
        features.append(values.mean())
        features.append(values.max())
    return np.array(features)

def make_dataframe_surrounding(path, patch_sizes=[3, 5, 9]):
    """Generate a DataFrame with surrounding features for each patch in labeled data using different patch sizes. Returns a concatenated DataFrame containing min, mean, and max features per channel."""
    data_columns = ['y', 'x', 'NDAI', 'SD', 'CORR', 'DF', 'CF', 'BF', 'AF', 'AN', 'expert_label']
    channel_names = data_columns[2:-1]
    print("path: ", path)
    image_ids, images_long_full, images_long_patch, _ = load_npz_files(f"{path}/*.npz", remove_label=True)
    global_miny, _, global_minx, _, height, width = compute_global_grid(images_long_full)
    images = reshape_images(images_long_patch, global_miny, global_minx, height, width)
    images = normalize_images(images, keepdims=True)
    df_chunks = []
    for i, orig_data in enumerate(images_long_full):
        padded_images = {p: np.pad(images[i], ((0, 0), (p // 2, p // 2), (p // 2, p // 2)), mode="reflect")
                         for p in patch_sizes}
        chunk_rows = []
        for row in orig_data:
            row_dict = {"image_id": image_ids[i]}
            row_dict.update(dict(zip(data_columns, row)))
            for p in patch_sizes:
                pad_len = p // 2
                y_idx = int(row[0]) - global_miny + pad_len
                x_idx = int(row[1]) - global_minx + pad_len
                patch = padded_images[p][:, y_idx - pad_len:y_idx + pad_len + 1,
                        x_idx - pad_len:x_idx + pad_len + 1].astype(np.float32)
                features = compute_surrounding_features(patch)
                for ch in range(len(channel_names)):
                    row_dict[f"{channel_names[ch]}_{p}_min"] = features[ch * 3]
                    row_dict[f"{channel_names[ch]}_{p}_mean"] = features[ch * 3 + 1]
                    row_dict[f"{channel_names[ch]}_{p}_max"] = features[ch * 3 + 2]
            chunk_rows.append(row_dict)
        df_chunks.append(pd.DataFrame(chunk_rows))
    return pd.concat(df_chunks, ignore_index=True)

def make_data(data_path, patch_size=9):
    """Load image data from NPZ files and create patches from each image with the specified patch size. Returns the original image arrays, patches, and file paths."""
    filepaths = glob.glob(f"{data_path}/*.npz")
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)

    # Compute global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    # Reshape each image onto the common grid.
    nchannels = images_long[0].shape[1] - 2
    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        # Use global minimums to get relative coordinates.
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    # Now that all images have the same shape, convert to a 4D array.
    images = np.array(images)
    pad_len = patch_size // 2

    # Global normalization across images.
    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        # Pad the image by reflecting across the border.
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        # Use global min values to compute relative indices.
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)

    return images_long, patches, filepaths

def feature_engineering(ae_train, path_labeled, path_unlabeled, path_config, path_output):
    df_pca = make_dataframe_surrounding(path_labeled, patch_sizes=[3, 5, 9, 13])
    config = yaml.safe_load(open(path_config, "r"))
    
    if ae_train == True:
        _, patches, _ = make_data(path_unlabeled, patch_size=config["data"]["patch_size"])
        run_autoencoder.run_autoencoder(path_unlabeled, path_config, patches)

    images_long, patches, filepaths = make_data(path_labeled, patch_size=config["data"]["patch_size"])
    get_embedding.get_embedding(path_config, images_long, patches, filepaths)

    csv_files = glob.glob("../../data/transfer_data/*.csv")
    dfs = []
    for file in csv_files:
        print(file)
        temp_df = pd.read_csv(file)
        temp_path = os.path.basename(file).replace("_ae.csv", ".npz").split("/transfer_data/")[-1]
        print(temp_path)
        temp_df["image_id"] = temp_path
        dfs.append(temp_df)

    df_ae = pd.concat(dfs, ignore_index=True)
    df = pd.merge(df_pca, df_ae, on=["image_id", "x", "y"], how="inner")

    chunks = np.array_split(df, 12)

    for i, chunk in enumerate(chunks, start=1):
        chunk.to_csv(f"{path_output}/part_{i}.csv", index=False)